(in-package #:cl-gpu)

(defclass gpu-array ()
  ((device :initarg :device :reader gpu-array-device)
   (buffer :initarg :buffer :accessor array-buffer)
   (element-type :initarg :element-type :reader gpu-array-element-type)
   (length :initarg :length :reader gpu-array-length))
  (:documentation "A device buffer of LENGTH elements of one scalar or vector ELEMENT-TYPE."))

(defmethod print-object ((array gpu-array) stream)
  (print-unreadable-object (array stream :type t)
    (format stream "~a x ~a~:[~; (freed)~]"
            (gpu-array-length array)
            (type-description (gpu-array-element-type array))
            (null (array-buffer array)))))

(defun parse-element-type (spec)
  (let ((type (parse-type spec)))
    (unless (or (scalar-type-p type) (vector-type-p type))
      (gpu-fail "array elements are scalars or vectors, got ~a" (type-description type)))
    type))

(defun element-bytes (type)
  "Bytes one element of an array of TYPE takes in a buffer."
  (or (type-bytes type) (/ (scalar-type-bits type) 8)))

(defun component-count (type)
  (if (vector-type-p type) (vector-type-length type) 1))

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
         (type (parse-element-type element-type))
         (buffer (device-allocate device (max 1 (* length (element-bytes type)))))
         (array (make-instance 'gpu-array :device device :buffer buffer
                                          :element-type type :length length)))
    (tg:finalize array (finalizer device buffer))
    (when initial-contents
      (handler-bind ((error (lambda (condition)
                              (declare (ignore condition))
                              (free array))))
        (store-contents array initial-contents)))
    array))

(defun flat-components (type sequence)
  "The components in SEQUENCE: numbers, or for a vector TYPE a sequence per element, which is
flattened into a vector specialised to the component type."
  (if (and (vector-type-p type)
           (plusp (length sequence))
           (typep (elt sequence 0) 'sequence))
      (let* ((count (component-count type))
             (lisp-type (lisp-element-type type))
             (floatp (float-type-p type))
             (components (make-array (* (length sequence) count) :element-type lisp-type))
             (index 0))
        (map nil (lambda (element)
                   (unless (and (typep element 'sequence) (= (length element) count))
                     (gpu-fail "~s is not ~d components of ~a"
                               element count (type-description type)))
                   (map nil (lambda (value)
                              (setf (aref components index) (if floatp (coerce value lisp-type) value))
                              (incf index))
                        element))
             sequence)
        components)
      sequence))

(defun store-contents (array sequence)
  (let* ((type (gpu-array-element-type array))
         (components (flat-components type sequence))
         (count (component-count type)))
    (unless (zerop (mod (length components) count))
      (gpu-fail "~d components are not whole elements of ~a"
                (length components) (type-description type)))
    (unless (<= (/ (length components) count) (gpu-array-length array))
      (gpu-fail "~d elements do not fit an array of ~d"
                (/ (length components) count) (gpu-array-length array)))
    (device-store (gpu-array-device array) (live-buffer array) components type)))

(defun to-device (sequence &key (element-type 'f32) device)
  "A new array holding SEQUENCE: numbers, or for a vector ELEMENT-TYPE a sequence of components per
element or all the components in one flat sequence."
  (let* ((type (parse-element-type element-type))
         (components (flat-components type sequence)))
    (make-gpu-array (ceiling (length components) (component-count type))
                    :element-type type :device device :initial-contents components)))

(defun from-device (array &key into flat)
  "The contents of ARRAY in a new vector specialised to the element type, or in the first elements of
INTO if given. A vector element is a vector of its components, or with FLAT the components are in one sequence."
  (let* ((type (gpu-array-element-type array))
         (count (component-count type))
         (limit (gpu-array-length array)))
    (flet ((load-into (components)
             (device-load (gpu-array-device array) (live-buffer array) components type)))
      (if (and (vector-type-p type) (not flat))
          (let* ((elements (or into (make-array limit)))
                 (components (make-array (* count (length elements))
                                         :element-type (lisp-element-type type))))
            (unless (<= (length elements) limit)
              (gpu-fail "~d elements do not fit from an array of ~d" (length elements) limit))
            (load-into components)
            (dotimes (index (length elements) elements)
              (setf (elt elements index)
                    (subseq components (* index count) (* (1+ index) count)))))
          (let ((components (or into (make-array (* count limit)
                                                  :element-type (lisp-element-type type)))))
            (unless (and (zerop (mod (length components) count))
                         (<= (length components) (* count limit)))
              (gpu-fail "~d components do not fit from an array of ~d ~a"
                        (length components) limit (type-description type)))
            (load-into components)
            components)))))

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
