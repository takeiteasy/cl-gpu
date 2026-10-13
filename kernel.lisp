(in-package #:cl-gpu)

(defstruct (gpu-kernel (:constructor %make-gpu-kernel (name ir shared-bytes)))
  name
  ir
  shared-bytes
  ;; TODO: entries of released devices stay (#18)
  (pipelines (make-hash-table :test 'eq :synchronized t)))

(defun static-shared-bytes (ir)
  "Bytes of the shared arrays of IR, not counting alignment."
  (loop for array in (shared-arrays ir)
        sum (* (ir-array-size array)
               (pointee-bytes (pointer-type-pointee (ir-variable-type (ir-array-variable array)))))))

(defun make-gpu-kernel (name arguments form)
  (let ((ir (lower-kernel form :name name :arguments arguments)))
    (%make-gpu-kernel name ir (static-shared-bytes ir))))

(defun find-kernel (name)
  "The kernel that `define-kernel` bound to NAME, or NIL."
  (get name 'gpu-kernel))

(defun kernel-ir (kernel)
  "The lowered IR-KERNEL of KERNEL."
  (gpu-kernel-ir kernel))

(defmacro define-kernel (name arguments &body body)
  "Define the function NAME, which runs BODY on the GPU.
ARGUMENTS is a list of (parameter type). The function takes one argument per parameter, then
the keywords :grid (required), :group, :wait (default true) and :device. BODY is lowered now,
so type errors are reported when the definition is compiled."
  (let ((documentation (when (and (stringp (first body)) (rest body))
                         (pop body)))
        (form `(progn ,@body))
        (parameters (loop for nil in arguments collect (gensym "ARG")))
        (grid (gensym "GRID"))
        (group (gensym "GROUP"))
        (wait (gensym "WAIT"))
        (device (gensym "DEVICE")))
    (lower-kernel form :name name :arguments arguments)
    `(progn
       (setf (get ',name 'gpu-kernel) (make-gpu-kernel ',name ',arguments ',form))
       (defun ,name (,@parameters &key ((:grid ,grid)) ((:group ,group)) ((:wait ,wait) t)
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
  (or (gethash device (gpu-kernel-pipelines kernel))
      (let* ((ir (gpu-kernel-ir kernel))
             (target (device-target device)))
        (setf (gethash device (gpu-kernel-pipelines kernel))
              (device-prepare device (kernel-source ir target) (kernel-entry-name ir target))))))

(defun check-shared-memory (kernel arguments device)
  (let ((limit (device-shared-memory-limit device))
        (needed (+ (gpu-kernel-shared-bytes kernel)
                   (loop for argument in arguments
                         when (eq (first argument) :shared) sum (second argument)))))
    (when (and limit (> needed limit))
      (gpu-fail "~(~a~): needs ~d bytes of shared memory, the device allows ~d"
                (gpu-kernel-name kernel) needed limit))))

(defun launch (kernel arguments &key grid group (wait t) device)
  "Run KERNEL on ARGUMENTS over GRID threads. Return NIL, or with WAIT NIL the command to pass to WAIT."
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
    (let ((command (device-launch device (prepared-kernel kernel device) converted grid group wait)))
      (unless wait
        command))))
