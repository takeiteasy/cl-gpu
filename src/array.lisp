(in-package #:cl-gpu)

(defclass gpu-array ()
  ((device :initarg :device :reader gpu-array-device)
   (buffer :initarg :buffer :accessor array-buffer)
   (element-type :initarg :element-type :reader gpu-array-element-type)
   (length :initarg :length :reader gpu-array-length))
  (:documentation "A device buffer of LENGTH numbers of one scalar ELEMENT-TYPE."))

(defmethod print-object ((array gpu-array) stream)
  (print-unreadable-object (array stream :type t)
    (format stream "~a x ~a~:[~; (freed)~]"
            (gpu-array-length array)
            (type-description (gpu-array-element-type array))
            (null (array-buffer array)))))

(defun scalar-element-type (spec)
  (let ((type (parse-type spec)))
    (unless (scalar-type-p type)
      (gpu-fail "array elements are scalars, got ~a" (type-description type)))
    type))

(defun scalar-bytes (type)
  (/ (scalar-type-bits type) 8))

(defun live-buffer (array)
  (or (array-buffer array)
      (gpu-fail "the array was freed")))

(defun finalizer (device buffer)
  "A function freeing BUFFER that holds no reference to the array that owns it."
  (lambda () (device-free device buffer)))

(defun make-gpu-array (length &key (element-type 'f32) device initial-contents)
  "A zeroed array of LENGTH ELEMENT-TYPE numbers on DEVICE, holding INITIAL-CONTENTS if given.
The array is freed when garbage collected, or sooner by FREE."
  (let* ((device (or device (current-device)))
         (type (scalar-element-type element-type))
         (buffer (device-allocate device (max 1 (* length (scalar-bytes type)))))
         (array (make-instance 'gpu-array :device device :buffer buffer
                                          :element-type type :length length)))
    (tg:finalize array (finalizer device buffer))
    (when initial-contents
      (handler-bind ((error (lambda (condition)
                              (declare (ignore condition))
                              (free array))))
        (store-contents array initial-contents)))
    array))

(defun store-contents (array sequence)
  (unless (<= (length sequence) (gpu-array-length array))
    (gpu-fail "~d elements do not fit an array of ~d"
              (length sequence) (gpu-array-length array)))
  (device-store (gpu-array-device array) (live-buffer array) sequence
                (gpu-array-element-type array)))

(defun to-device (sequence &key (element-type 'f32) device)
  "A new array holding the numbers of SEQUENCE."
  (make-gpu-array (length sequence) :element-type element-type :device device
                                    :initial-contents sequence))

(defun from-device (array &optional sequence)
  "The contents of ARRAY in a new vector, or in the first elements of SEQUENCE if given."
  (let ((sequence (or sequence (make-array (gpu-array-length array)))))
    (unless (<= (length sequence) (gpu-array-length array))
      (gpu-fail "~d elements do not fit from an array of ~d"
                (length sequence) (gpu-array-length array)))
    (device-load (gpu-array-device array) (live-buffer array) sequence
                 (gpu-array-element-type array))
    sequence))

(defun free (array)
  "Release ARRAY's device memory now. Freeing twice is harmless."
  (let ((buffer (array-buffer array)))
    (when buffer
      (setf (array-buffer array) nil)
      (tg:cancel-finalization array)
      (device-free (gpu-array-device array) buffer)))
  nil)

(defmacro with-gpu-array ((variable form) &body body)
  "Bind VARIABLE to the array FORM returns and free it on exit."
  `(let ((,variable ,form))
     (unwind-protect (progn ,@body)
       (free ,variable))))
