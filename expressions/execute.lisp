(in-package #:cl-gpu/expressions)

(defstruct task kernel arguments metadata grid group)

(defun own-view (context shape element)
  ;; TODO: scratch grows with expression depth; reuse dead buffers and tile producers (#83).
  (let ((data (gpu:make-gpu-array (gpu::shape-size shape) :element-type element
                                 :device (context-device context))))
    (push data (context-owned context))
    (gpu:make-strided-view data shape)))

(defun input-indices (body)
  (let ((indices nil))
    (labels ((walk (node)
               (when (consp node)
                 (if (eq (car node) :input) (pushnew (second node) indices)
                     (mapc #'walk (cdr node))))))
      (walk body))
    (sort indices #'<)))

(defun remap-inputs (body indices)
  (if (atom body) body
      (if (eq (car body) :input) (list :input (position (second body) indices))
          (cons (car body) (mapcar (lambda (child) (remap-inputs child indices)) (cdr body))))))

(defun append-operand (context view)
  (let ((index (length (context-operands context))))
    (setf (context-operands context) (concatenate 'vector (context-operands context) (vector view)))
    (list :input index)))

(defun schedule-region (context body demand element &key predicatep raw out shape)
  (let* ((shape (or shape (context-shape context)))
         (result (or out (own-view context shape element)))
         (indices (input-indices body))
         (inputs (mapcar (lambda (i) (aref (context-operands context) i)) indices))
         (types (mapcar (lambda (input) (if (gpu:strided-view-p input) (gpu::view-element input) :scalar)) inputs))
         (body (remap-inputs body indices))
         (all-inputs (append inputs (when demand (list demand))))
         (kernel (cached-kernel context (list :region body types element (not (null demand)) predicatep raw)
                                (lambda () (region-kernel body types element demand predicatep raw))))
         (metadata (ew::operation-metadata shape result all-inputs))
         (arguments (loop for input in all-inputs
                          append (if (gpu:strided-view-p input)
                                     (list (gpu::view-root input) (gpu::view-base input)) (list input)))))
    (loop for start from 0 below (gpu::shape-size shape) by 16776960
          for count = (min 16776960 (- (gpu::shape-size shape) start))
          do (push (make-task :kernel kernel :metadata metadata
                              :arguments (append (list (gpu::view-root result) :metadata (length shape)
                                                       (gpu::shape-size shape) start (gpu::view-base result)) arguments)
                              :grid count :group 256)
                   (context-tasks context)))
    result))

(defun reduction-layout (context input)
  (let* ((shape (context-shape context)) (axes (context-axes context))
         (kept (loop for a below (length shape) unless (member a axes) collect a))
         (strides (ew::broadcast-strides input shape))
         (metadata (make-array (* 2 (length shape)) :element-type '(signed-byte 64))))
    (loop for axis in (append (reverse kept) (reverse axes)) for i from 0
          do (setf (aref metadata (* i 2)) (aref shape axis)
                   (aref metadata (1+ (* i 2))) (aref strides axis)))
    (values metadata (length kept))))

(defun schedule-reduction (context operation input demand)
  ;; TODO: one group underuses the GPU for long reductions; split across groups (#74).
  (let* ((shape (red::result-shape (context-shape context) (context-axes context) t))
         (element (cond ((eq operation :any) 'gpu:u8)
                        ((member operation '(:argmin :argmax)) 'gpu:i64) (t 'gpu:f32)))
         (result (own-view context shape element))
         (kernel (cached-kernel context (list :reduction operation (not (null demand)))
                                (lambda () (reduction-kernel operation demand)))))
    (multiple-value-bind (metadata kept) (reduction-layout context input)
      (loop for first from 0 below (context-groups context) by 65535
            for groups = (min 65535 (- (context-groups context) first))
            do (push (make-task :kernel kernel :metadata metadata
                                :arguments (append (list (gpu::view-root input) (gpu::view-root result)
                                                         :metadata kept (length (context-shape context))
                                                         (context-width context) (gpu::view-base input) first)
                                                   (when demand (list (gpu::view-root demand))))
                                :grid (* groups 256) :group 256)
                     (context-tasks context))))
    result))

(defun contains-reduction-p (body)
  (and (consp body) (or (reducer-p (car body)) (some #'contains-reduction-p (cdr body)))))

(defun resolve-reductions (context body demand)
  (cond
    ((atom body) body)
    ((eq (car body) :input) body)
    ((reducer-p (car body)) (append-operand context (evaluate-reducer context body demand)))
    ((and (eq (car body) :select) (contains-reduction-p body))
     (let* ((predicate (evaluate-region context (second body) demand 'gpu:u8 :predicatep t))
            (leaf (append-operand context predicate))
            (yes (schedule-region context `(:select (:ne ,leaf 0f0) 1f0 0f0) demand 'gpu:u8))
            (no (schedule-region context `(:select (:eq ,leaf 0f0) 1f0 0f0) demand 'gpu:u8)))
       `(:select (:ne ,leaf 0f0)
                 ,(resolve-reductions context (third body) yes)
                 ,(resolve-reductions context (fourth body) no))))
    (t (cons (car body) (mapcar (lambda (child) (resolve-reductions context child demand)) (cdr body))))))

(defun evaluate-region (context body demand element &key predicatep)
  (schedule-region context (resolve-reductions context body demand) demand element :predicatep predicatep))

(defun evaluate-reducer (context body demand)
  (let ((key (list body demand)))
    (or (gethash key (context-memo context))
        (setf (gethash key (context-memo context))
              (let* ((group-demand (when demand (schedule-reduction context :any demand nil)))
                     (input (if (zerop (context-width context))
                                (own-view context (context-shape context) 'gpu:f32)
                                (evaluate-region context (second body) group-demand 'gpu:f32))))
                (schedule-reduction context (car body) input group-demand))))))

(defun prepare-tasks (context tasks)
  (dolist (task tasks)
    (let ((kernel (task-kernel task)) (device (context-device context)))
      (gpu::check-shared-memory kernel nil device)
      (bt:with-lock-held ((gpu::gpu-kernel-lock kernel))
        (unless (gethash device (gpu::gpu-kernel-pipelines kernel))
          (let* ((ir (gpu:kernel-ir kernel)) (target (gpu:device-target device))
                 (source (concatenate 'string (format nil "#pragma clang fp contract(off)~%")
                                      (gpu:kernel-source ir target))))
            (setf (gethash device (gpu::gpu-kernel-pipelines kernel))
                  (gpu:device-prepare device source (gpu::kernel-entry-name ir target) :precise-math t))))))))

(defun upload-task-metadata (context tasks)
  (let ((uploads (make-hash-table :test #'eq)))
    (dolist (task tasks)
      (let* ((metadata (task-metadata task))
             (data (or (gethash metadata uploads)
                       (let ((data (gpu:to-device metadata :element-type 'gpu:i64 :device (context-device context))))
                         (push data (context-owned context))
                         (setf (gethash metadata uploads) data)))))
        (setf (task-arguments task) (substitute data :metadata (task-arguments task)))))))

(defun launch-tasks (context tasks)
  (dolist (task tasks)
    (gpu::launch (task-kernel task) (task-arguments task) :grid (task-grid task)
                 :group (task-group task) :device (context-device context))))

(defun run-expression (plan operands out axis keepdims)
  (let* ((context (expression-context plan operands axis keepdims))
         (shape (context-result-shape context)) (element (context-result-element context))
         (body (expression-plan-body plan)) (result nil) (success nil) (launched nil))
    (when out (red::check-output out shape element (context-device context)))
    (unwind-protect
         (progn
           (setf result
                 (cond
                   ((zerop (gpu::shape-size shape)) (own-view context shape element))
                   ((and (consp body) (reducer-p (car body)))
                    (let* ((reduced (evaluate-reducer context body nil))
                           (reshaped (gpu:make-strided-view (gpu:strided-view-data reduced) shape)))
                      (if (eq element (gpu::view-element reduced)) reshaped
                          (schedule-region context (append-operand context reshaped) nil element :raw t :shape shape))))
                   (t (evaluate-region context body nil element))))
           (let* ((computation (nreverse (context-tasks context)))
                  (commit (when (and out (plusp (gpu::shape-size shape)))
                            (setf (context-tasks context) nil)
                            (schedule-region context (append-operand context result) nil element
                                             :raw t :out out :shape shape)
                            (nreverse (context-tasks context))))
                  (tasks (append computation commit)))
             (prepare-tasks context tasks)
             (upload-task-metadata context tasks)
             (when computation
               (setf launched t)
               (launch-tasks context computation)
               (gpu:synchronize (context-device context)))
             (when commit
               (setf launched t)
               (launch-tasks context commit)
               (gpu:synchronize (context-device context))))
           (setf success t)
           (or out result))
      (when (and launched (not success)) (ignore-errors (gpu:synchronize (context-device context))))
      (dolist (data (context-owned context))
        (unless (and success (null out) (eq data (gpu:strided-view-data result))) (gpu:free data))))))

(defun execute-expression (plan operands &key axis keepdims)
  "Execute PLAN synchronously; return an independently owned contiguous result view."
  (run-expression plan operands nil axis keepdims))

(defun execute-expression! (out plan operands &key axis keepdims)
  "Compute into scratch, then synchronously copy into OUT. Return OUT."
  (gpu::check-view out)
  (run-expression plan operands out axis keepdims))
