(in-package #:cl-gpu)

(defstruct (gpu-kernel (:constructor %make-gpu-kernel (name ir shared-bytes precise-math)))
  name
  ir
  shared-bytes
  precise-math
  ;; TODO: entries of released devices stay (#18)
  (pipelines (make-hash-table :test 'eq))
  (lock (bt:make-lock "gpu-kernel pipelines")))

(defun static-shared-bytes (ir)
  "Bytes of the shared arrays of IR, not counting alignment."
  (loop for array in (shared-arrays ir)
        sum (* (ir-array-size array)
               (pointee-bytes (pointer-type-pointee (ir-variable-type (ir-array-variable array)))))))

(defun make-gpu-kernel (name arguments form &key precise-math)
  (let ((ir (lower-kernel form :name name :arguments arguments)))
    (%make-gpu-kernel name ir (static-shared-bytes ir) precise-math)))

(defun find-kernel (name)
  "The kernel that `define-kernel` bound to NAME, or NIL."
  (get name 'gpu-kernel))

(defun kernel-ir (kernel)
  "The lowered IR-KERNEL of KERNEL."
  (gpu-kernel-ir kernel))

(defun lower-form-cst (cst)
  "Lower the `define-kernel` form CST, so that errors carry its source locations."
  (let* ((name (cst:raw (cst:second cst)))
         (arguments (cst:third cst))
         (body (cst:nthrest 3 cst)))
    (when (and (stringp (cst:raw (cst:first body))) (cst:rest body))
      (setf body (cst:rest body)))
    (when (and (cst:consp (cst:first body)) (eq (cst:raw (cst:first (cst:first body))) 'declare))
      (setf body (cst:rest body)))
    (lower-kernel (cst:cons (cst:cst-from-expression 'progn) body :source (cst:source cst))
                  :name name :arguments arguments)))

(defun located-kernel-error (whole original)
  "The error from lowering the file text of WHOLE if it repeats ORIGINAL with a location, or NIL."
  (let ((cst (find-form-cst whole)))
    (when cst
      (handler-case (progn (lower-form-cst cst) nil)
        (kernel-error (located)
          (when (and (eq (type-of located) (type-of original))
                     (string= (kernel-error-message located) (kernel-error-message original)))
            located))))))

(defun check-kernel (whole name arguments form)
  "Lower FORM now. A kernel error is signalled with a source location when WHOLE can be found in its file."
  (handler-case (lower-kernel form :name name :arguments arguments)
    (kernel-error (original)
      (error (or (located-kernel-error whole original) original)))))

(defun split-kernel-declaration (name body)
  "Split a leading (declare (precise-math)) off BODY. Return whether it was there, and the rest."
  (let ((declaration (first body)))
    (if (and (consp declaration) (eq (first declaration) 'declare))
        (progn
          (dolist (specifier (rest declaration))
            (unless (and (consp specifier) (null (rest specifier))
                         (string= (symbol-name (first specifier)) "PRECISE-MATH"))
              (error 'kernel-syntax-error
                     :message (format nil "~(~a~): the only kernel declaration is (precise-math), got ~s"
                                      name specifier)
                     :form declaration)))
          (values t (rest body)))
        (values nil body))))

(defmacro define-kernel (&whole whole name arguments &body body)
  "Define the function NAME, which runs BODY on the GPU.
ARGUMENTS is a list of (parameter type). The function takes one argument per parameter, then
the keywords :grid (required), :group, :wait and :device. The launch is queued: it returns NIL at
once, or with :wait true after the device finishes all its launches (see SYNCHRONIZE). BODY is
lowered now, so type errors are reported when the definition is compiled.
A leading (declare (precise-math)) compiles the kernel without fast math."
  (let* ((documentation (when (and (stringp (first body)) (rest body))
                          (pop body)))
         (precise-math (multiple-value-bind (precise rest) (split-kernel-declaration name body)
                         (setf body rest)
                         precise))
         (form `(progn ,@body))
         (parameters (loop for nil in arguments collect (gensym "ARG")))
         (grid (gensym "GRID"))
         (group (gensym "GROUP"))
         (wait (gensym "WAIT"))
         (device (gensym "DEVICE")))
    (check-kernel whole name arguments form)
    `(progn
       (setf (get ',name 'gpu-kernel) (make-gpu-kernel ',name ',arguments ',form :precise-math ,precise-math))
       (defun ,name (,@parameters &key ((:grid ,grid)) ((:group ,group)) ((:wait ,wait) nil)
                                    ((:device ,device)))
         ,@(when documentation (list documentation))
         (launch (find-kernel ',name) (list ,@parameters)
                 :grid ,grid :group ,group :wait ,wait :device ,device)))))

;;; Arguments

(defun pointee-bytes (type)
  (if (atomic-type-p type)
      (pointee-bytes (atomic-type-element type))
      (element-bytes type)))

(defun scalar-value-p (type value)
  (if (eq (scalar-type-kind type) :float)
      (realp value)
      (or (eq (scalar-type-kind type) :bool)
          (typep value (integer-range type)))))

(defun value-p (type value)
  (etypecase type
    (scalar-type (scalar-value-p type value))
    (vector-type (and (typep value 'sequence)
                      (= (length value) (vector-type-length type))
                      (every (lambda (component) (scalar-value-p (vector-type-element type) component))
                             value)))))

(defun launch-argument (kernel parameter value device)
  "The backend argument that passes VALUE as the IR-VARIABLE PARAMETER of KERNEL."
  (let ((type (ir-variable-type parameter)))
    (flet ((refuse (control &rest arguments)
             (gpu-fail "~(~a~), parameter ~(~a~) ~(~a~): ~?"
                       (gpu-kernel-name kernel) (ir-variable-name parameter)
                       (type-description type) control arguments)))
      (etypecase type
        (pointer-type
         (ecase (pointer-type-space type)
           ((:global :constant)
            (unless (typep value 'gpu-array)
              (refuse "expected a gpu-array, got ~s" value))
            (unless (eq (gpu-array-device value) device)
              (refuse "the array is on another device"))
            (unless (eq (gpu-array-element-type value) (pointer-type-pointee type))
              (refuse "the array holds ~a" (type-description (gpu-array-element-type value))))
            (list :buffer (or (array-buffer value) (refuse "the array was freed"))))
           (:shared
            (unless (typep value '(integer 0))
              (refuse "expected an element count, got ~s" value))
            (list :shared (* value (pointee-bytes (pointer-type-pointee type)))))
           (:private (refuse "cannot be passed"))))
        ((or scalar-type vector-type)
         (unless (value-p type value)
           (refuse "expected a value of that type, got ~s" value))
         (list :bytes type value))))))

(defun grid-dimensions (kernel keyword value)
  (let ((dimensions (if (listp value) value (list value))))
    (unless (and (<= 1 (length dimensions) 3)
                 (every (lambda (size) (typep size '(integer 1))) dimensions))
      (gpu-fail "~(~a~): ~(~s~) is one to three positive integers, got ~s"
                (gpu-kernel-name kernel) keyword value))
    (append dimensions (make-list (- 3 (length dimensions)) :initial-element 1))))

;;; Launching

(defun prepared-kernel (kernel device)
  (bt:with-lock-held ((gpu-kernel-lock kernel))
    (or (gethash device (gpu-kernel-pipelines kernel))
        (let* ((ir (gpu-kernel-ir kernel))
               (target (device-target device)))
          (setf (gethash device (gpu-kernel-pipelines kernel))
                (device-prepare device (kernel-source ir target) (kernel-entry-name ir target)
                                :precise-math (gpu-kernel-precise-math kernel)))))))

(defun check-shared-memory (kernel arguments device)
  (let ((limit (device-shared-memory-limit device))
        (needed (+ (gpu-kernel-shared-bytes kernel)
                   (loop for argument in arguments
                         when (eq (first argument) :shared) sum (second argument)))))
    (when (and limit (> needed limit))
      (gpu-fail "~(~a~): needs ~d bytes of shared memory, the device allows ~d"
                (gpu-kernel-name kernel) needed limit))))

(defun launch (kernel arguments &key grid group wait device)
  "Queue KERNEL on ARGUMENTS over GRID threads and return NIL. With WAIT, then wait for the device."
  (unless grid
    (gpu-fail "~(~a~): :grid is required" (gpu-kernel-name kernel)))
  (let* ((device (or device (current-device)))
         (parameters (ir-kernel-parameters (gpu-kernel-ir kernel)))
         (grid (grid-dimensions kernel :grid grid))
         (group (and group (grid-dimensions kernel :group group)))
         (converted (loop for parameter in parameters
                          for value in arguments
                          collect (launch-argument kernel parameter value device))))
    (check-shared-memory kernel converted device)
    (device-launch device (prepared-kernel kernel device) converted grid group)
    (when wait
      (device-synchronize device))
    nil))
