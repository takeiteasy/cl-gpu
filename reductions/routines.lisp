(in-package #:cl-gpu/reductions)

(defun result-shape (shape axes keepdims)
  (coerce (loop for dimension across shape for axis from 0
                when (or keepdims (not (member axis axes)))
                  collect (if (member axis axes) 1 dimension)) 'vector))

(defun allocate-view (shape element device)
  (make-strided-view (make-gpu-array (shape-size shape) :element-type element :device device) shape))

(defun check-output (out shape element device)
  (check-view out)
  (unless (and (equalp shape (strided-view-shape out))
               (eq (parse-type element) (gpu-array-element-type (strided-view-data out))))
    (gpu-fail "output shape or element type does not match"))
  (unless (eq device (gpu-array-device (strided-view-data out)))
    (gpu-fail "output is on another device"))
  (unless (unique-view-p out) (gpu-fail "output elements overlap"))
  out)

(defun weight-strides (input weights element device)
  (when (strided-view-p weights)
    (check-view weights)
    (unless (and (eq element (float-view-element weights))
                 (eq device (gpu-array-device (strided-view-data weights))))
      (gpu-fail "weights must have the input element type and device"))
    (let* ((shape (strided-view-shape input)) (ws (strided-view-shape weights))
           (rank (length shape)) (wrank (length ws)) (strides (make-array rank :initial-element 0)))
      (unless (<= wrank rank) (gpu-fail "weights expand the input shape"))
      (dotimes (axis wrank)
        (let* ((target (+ (- rank wrank) axis)) (dimension (aref ws axis)))
          (unless (or (= dimension 1) (= dimension (aref shape target)))
            (gpu-fail "weights do not broadcast to the input shape"))
          (unless (= dimension 1)
            (setf (aref strides target) (aref (strided-view-strides weights) axis)))))
      strides)))

(defun f32-scalar (value name)
  (unless (realp value) (gpu-fail "~a must be real" name))
  (handler-case (coerce value 'single-float)
    (arithmetic-error () (gpu-fail "~a does not fit f32" name))))

(defun layout-metadata (input output axes keepdims normalization weights)
  ;; Rows are (dimension input-stride output-stride weight-stride), fastest kept/reduced axes first.
  (let* ((shape (strided-view-shape input)) (rank (length shape))
         (kept (loop for a below rank unless (member a axes) collect a))
         (order (append (reverse kept) (reverse axes)))
         (ys (strided-view-strides output))
         (metadata (make-array (* 4 rank) :element-type '(signed-byte 64) :initial-element 0)))
    (loop for axis in order for i from 0
          for output-axis = (if (or normalization keepdims) axis (position axis kept))
          do (setf (aref metadata (* i 4)) (aref shape axis)
                   (aref metadata (+ (* i 4) 1)) (aref (strided-view-strides input) axis)
                   (aref metadata (+ (* i 4) 2))
                   (if (and output-axis (or normalization (not (member axis axes))))
                       (aref ys output-axis) 0)
                   (aref metadata (+ (* i 4) 3)) (if weights (aref weights axis) 0)))
    (values metadata (length kept))))

(defun scatter-metadata (output)
  (let* ((shape (strided-view-shape output)) (rank (length shape))
         (metadata (make-array (* 4 rank) :element-type '(signed-byte 64) :initial-element 0)))
    (dotimes (i rank metadata)
      (let ((axis (- rank i 1)))
        (setf (aref metadata (* 4 i)) (aref shape axis)
              (aref metadata (+ (* 4 i) 2)) (aref (strided-view-strides output) axis))))))

(defun scatter-result (input output metadata)
  (let ((n (shape-size (strided-view-shape output))))
    (loop for start from 0 below n by 16776960
          for count = (min (- n start) 16776960)
          do (funcall (kernel-name 'scatter (view-element output))
                      (view-root input) (view-root output) metadata
                      (length (strided-view-shape output)) n (view-base output) start
                      :grid count :group 256 :device (gpu-array-device (strided-view-data output))))))

(defun execute-operation (operation input out axis keepdims weights epsilon)
  (check-view input)
  (unless (and (realp epsilon) (>= epsilon 0)) (gpu-fail "epsilon must be nonnegative and real"))
  (let* ((element (float-view-element input))
         (device (gpu-array-device (strided-view-data input)))
         (input-shape (strided-view-shape input))
         (rank (length input-shape)) (axes (normalize-axes axis rank))
         (normalization (member operation '(softmax rmsnorm)))
         (indexp (member operation '(argmin argmax)))
         (shape (if normalization input-shape (result-shape input-shape axes keepdims)))
         (result-element (if indexp 'i64 element))
         (width (reduce #'* axes :key (lambda (a) (aref input-shape a)) :initial-value 1))
         (groups (reduce #'* (loop for a below rank unless (member a axes) collect (aref input-shape a))
                         :initial-value 1))
         (wstrides (when (eq operation 'rmsnorm) (weight-strides input weights element device)))
         (weight (if (or (null weights) wstrides) 1f0 (f32-scalar weights "weights")))
         (epsilon (f32-scalar epsilon "epsilon")))
    (unless (and (typep width '(integer 0 #.(- (expt 2 63) 257)))
                 (typep groups '(integer 0 #.(1- (expt 2 63)))))
      (gpu-fail "reduction dimensions exceed signed 64-bit indexing"))
    (when (and (plusp (shape-size shape)) (zerop width)
               (member operation '(mean minimum maximum argmin argmax)))
      (gpu-fail "empty groups have no ~a" operation))
    (when out (check-output out shape result-element device))
    (let* ((result (or out (allocate-view shape result-element device)))
           (alias (and out (or (views-overlap-p input out)
                               (and wstrides (views-overlap-p weights out)))))
           (scratch nil) (metadata nil) (scatter nil) (success nil))
      (unwind-protect
           (progn
             (unless (zerop (shape-size shape))
               (when alias (setf scratch (allocate-view shape result-element device)))
               (let ((target (or scratch result)))
                 (multiple-value-bind (layout kept)
                     (layout-metadata input target axes keepdims normalization wstrides)
                   (setf metadata (to-device layout :element-type 'i64 :device device))
                   (when alias
                     (setf scatter (to-device (scatter-metadata out) :element-type 'i64 :device device)))
                   ;; TODO: one group underuses the GPU for a few very long groups; add multi-pass (#74).
                   (loop for first-group from 0 below groups by 65535
                         for count = (min (- groups first-group) 65535)
                         do (if normalization
                                (funcall (kernel-name operation element)
                                         (view-root input) (view-root target)
                                         (if wstrides (view-root weights) (view-root input))
                                         metadata kept rank width (view-base input) (view-base target)
                                         (if wstrides (view-base weights) 0) first-group
                                         (if wstrides 1 0) weight epsilon
                                         :grid (* count 256) :group 256 :device device)
                                (funcall (kernel-name operation element)
                                         (view-root input) (view-root target) metadata kept rank width
                                         (view-base input) (view-base target) first-group
                                         :grid (* count 256) :group 256 :device device)))
                   (when alias (scatter-result scratch out scatter)))))
             (setf success t)
             result)
        (when metadata (free metadata))
        (when scatter (free scatter))
        (when scratch (free (strided-view-data scratch)))
        (unless (or success out) (free (strided-view-data result)))))))

(defmacro define-reduction-routines (name)
  (let ((out-name (intern (format nil "~a!" name))))
    `(progn
       (defun ,name (input &key axis keepdims)
         ,(format nil "Reduce INPUT with ~a over AXIS; return an independent strided view." name)
         (execute-operation ',name input nil axis keepdims nil 0f0))
       (defun ,out-name (out input &key axis keepdims)
         ,(format nil "Reduce INPUT with ~a over AXIS into OUT; return OUT." name)
         (execute-operation ',name input out axis keepdims nil 0f0)))))

(define-reduction-routines sum)
(define-reduction-routines mean)
(define-reduction-routines prod)
(define-reduction-routines minimum)
(define-reduction-routines maximum)
(define-reduction-routines argmin)
(define-reduction-routines argmax)

(defun softmax (input &key (axis -1))
  "Normalize INPUT by exponentials over AXIS; return independent storage."
  (execute-operation 'softmax input nil axis nil nil 0f0))

(defun softmax! (out input &key (axis -1))
  "Normalize INPUT by exponentials over AXIS into OUT; return OUT."
  (execute-operation 'softmax input out axis nil nil 0f0))

(defun rmsnorm (input &key (axis -1) weights (epsilon 1e-6))
  "Normalize INPUT by its root mean square over AXIS, then apply WEIGHTS."
  (execute-operation 'rmsnorm input nil axis nil weights epsilon))

(defun rmsnorm! (out input &key (axis -1) weights (epsilon 1e-6))
  "Normalize INPUT by its root mean square over AXIS into OUT; return OUT."
  (execute-operation 'rmsnorm input out axis nil weights epsilon))
