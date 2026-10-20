(in-package #:cl-gpu)

(defclass gpu-array ()
  ((device :initarg :device :reader gpu-array-device)
   (buffer :initarg :buffer :accessor array-buffer)
   (element-type :initarg :element-type :reader gpu-array-element-type)
   (length :initarg :length :reader gpu-array-length))
  (:documentation "A device buffer of LENGTH elements of one scalar, vector or complex ELEMENT-TYPE."))

(defclass gpu-array-view (gpu-array)
  ((owner :initarg :owner :reader gpu-array-view-owner)
   (offset :initarg :offset :reader gpu-array-view-offset))
  (:documentation "A window onto the buffer of OWNER, an array that owns its buffer, starting OFFSET
elements in. It keeps OWNER alive and is freed with it."))

(defmethod array-buffer ((view gpu-array-view))
  (array-buffer (gpu-array-view-owner view)))

(defmethod print-object ((array gpu-array) stream)
  (print-unreadable-object (array stream :type t)
    (format stream "~a x ~a~@[ at ~a~]~:[~; (freed)~]"
            (gpu-array-length array)
            (type-description (gpu-array-element-type array))
            (and (gpu-array-view-p array) (gpu-array-view-offset array))
            (null (array-buffer array)))))

(defun gpu-array-view-p (object)
  (typep object 'gpu-array-view))

(defun array-root-and-offset (array)
  "The array that owns the buffer of ARRAY, and the element offset of ARRAY in it."
  (if (gpu-array-view-p array)
      (values (gpu-array-view-owner array) (gpu-array-view-offset array))
      (values array 0)))

(defun parse-element-type (spec)
  (let ((type (parse-type spec)))
    (unless (or (scalar-type-p type) (vector-type-p type) (complex-type-p type))
      (gpu-fail "array elements are scalars, vectors or complex values, got ~a" (type-description type)))
    type))

(defun element-bytes (type)
  "Bytes one element of an array of TYPE takes in a buffer."
  (or (type-bytes type) (/ (scalar-type-bits type) 8)))

(defun component-count (type)
  (cond ((complex-type-p type) 2) ((vector-type-p type) (vector-type-length type)) (t 1)))

(defun live-buffer (array)
  (or (array-buffer array)
      (gpu-fail "the array was freed")))

(defun array-byte-offset (array)
  "Bytes from the start of the buffer of ARRAY to its first element."
  (* (nth-value 1 (array-root-and-offset array))
     (element-bytes (gpu-array-element-type array))))

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
        (copy-into array initial-contents)))
    array))

(defun make-gpu-array-view (array length &key (offset 0))
  "A view of LENGTH elements of ARRAY from element OFFSET. It shares the memory of ARRAY, which a
view of a view shares too, and keeps it alive. Freeing a view does nothing."
  (unless (typep array 'gpu-array)
    (gpu-fail "expected a gpu-array, got ~s" array))
  (unless (and (typep length '(integer 0)) (typep offset '(integer 0))
               (<= (+ offset length) (gpu-array-length array)))
    (gpu-fail "a view of ~s elements from ~s does not fit an array of ~d"
              length offset (gpu-array-length array)))
  (live-buffer array)
  (multiple-value-bind (root base) (array-root-and-offset array)
    (make-instance 'gpu-array-view :device (gpu-array-device array)
                                   :element-type (gpu-array-element-type array)
                                   :length length :owner root :offset (+ base offset))))

(defun gpu-array-storage-view (array)
  "Borrow ARRAY with its scalar/vector storage type, retaining its owner and window offset."
  (live-buffer array)
  (let ((storage (storage-type (gpu-array-element-type array))))
    (if (eq storage (gpu-array-element-type array)) array
        (multiple-value-bind (root offset) (array-root-and-offset array)
          (make-instance 'gpu-array-view :owner root :offset offset
                         :device (gpu-array-device array) :length (gpu-array-length array)
                         :element-type storage)))))

(defun flat-components (type sequence)
  "The components in SEQUENCE: numbers, or for a vector TYPE a sequence per element, which is
flattened into a vector specialised to the component type."
  ;; TODO: generic map and a coerce per value, ~20x slower than flat input (#62)
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

(defun transfer-type (type raw)
  (cond ((not raw) (storage-type type))
        ((complex-type-p type)
         (parse-type `(vec ,(if (eq (type-spec type) 'c32) 'u32 'u64) 2)))
        (t
         (unless (and (scalar-type-p type)
                      (member (scalar-type-kind type) '(:float :signed :unsigned)))
           (gpu-fail "raw transfers require numeric scalar or complex elements"))
         (if (member (scalar-type-name type) '(f16 bf16)) (parse-type 'u16) type))))

(defun check-raw-vector (sequence type)
  (unless (typep sequence `(simple-array ,(lisp-element-type type) (*)))
    (gpu-fail "raw transfer requires a simple vector of ~s" (lisp-element-type type)))
  sequence)

(defun complex-components (type sequence flat raw)
  (cond (raw (check-raw-vector sequence (transfer-type type t)))
        (flat sequence)
        (t (let* ((component (lisp-element-type type))
                  (result (make-array (* 2 (length sequence)) :element-type component)))
             (loop for value across (coerce sequence 'vector) for i from 0
                   do (unless (numberp value) (gpu-fail "complex transfer requires numbers"))
                      (setf (aref result (* 2 i)) (coerce (realpart value) component)
                            (aref result (1+ (* 2 i))) (coerce (imagpart value) component)))
             result))))

(defun inferred-upload-type (sequence)
  (let ((declared (and (vectorp sequence) (array-element-type sequence))))
    (if (or (some #'complexp sequence)
            (and declared (subtypep declared '(complex float))))
        (if (or (and declared (subtypep declared '(complex double-float)))
                (some (lambda (x) (and (complexp x)
                                       (or (typep (realpart x) 'double-float)
                                           (typep (imagpart x) 'double-float)))) sequence))
            'c64 'c32)
        'f32)))

(defun copy-into (array sequence &key raw flat)
  "Store SEQUENCE in the first elements of ARRAY, in the form TO-DEVICE takes. Return ARRAY.
FLAT supplies interleaved complex components; RAW supplies specialized component bits."
  (let* ((type (gpu-array-element-type array))
         (transfer (transfer-type type raw))
         (components (if (complex-type-p type) (complex-components type sequence flat raw)
                         (if raw (check-raw-vector sequence transfer) (flat-components type sequence))))
         (count (component-count type)))
    (unless (zerop (mod (length components) count))
      (gpu-fail "~d components are not whole elements of ~a"
                (length components) (type-description type)))
    (unless (<= (/ (length components) count) (gpu-array-length array))
      (gpu-fail "~d elements do not fit an array of ~d"
                (/ (length components) count) (gpu-array-length array)))
    (device-store (gpu-array-device array) (live-buffer array) components transfer
                  (array-byte-offset array))
    array))

(defun to-device (sequence &key (element-type nil supplied-p) device raw flat)
  "Upload values, flat components, or raw bits into independent device storage."
  (when (and (or raw flat) (not supplied-p))
    (gpu-fail "flat/raw uploads require an explicit element type"))
  (let* ((type (parse-element-type (if supplied-p element-type (inferred-upload-type sequence))))
         (transfer (transfer-type type raw))
         (components (if (complex-type-p type) (complex-components type sequence flat raw)
                         (if raw (check-raw-vector sequence transfer) (flat-components type sequence))))
         (count (component-count type)))
    (unless (zerop (mod (length components) count)) (gpu-fail "incomplete element components"))
    (let ((array (make-gpu-array (/ (length components) count) :element-type type :device device)))
      (handler-case (copy-into array components :raw raw :flat t)
        (error (condition) (free array) (error condition))))))

(defun from-device (array &key into flat raw)
  "The contents of ARRAY in a new vector specialised to the element type, or in the first elements of
INTO if given. RAW transfers scalar bit representations. A vector element is a vector of its
components, or with FLAT the components are in one sequence."
  (let* ((type (gpu-array-element-type array))
         (transfer (transfer-type type raw))
         (count (component-count type))
         (limit (gpu-array-length array)))
    (when (and raw into) (check-raw-vector into transfer))
    (flet ((load-into (components)
             (device-load (gpu-array-device array) (live-buffer array) components transfer
                         (array-byte-offset array))))
      (if (and (or (vector-type-p type) (complex-type-p type)) (not flat) (not raw))
          (let* ((elements (or into (make-array limit)))
                 (components (make-array (* count (length elements))
                                         :element-type (lisp-element-type type))))
            (unless (<= (length elements) limit)
              (gpu-fail "~d elements do not fit from an array of ~d" (length elements) limit))
            (load-into components)
            (dotimes (index (length elements) elements)
              (setf (elt elements index)
                    (if (complex-type-p type)
                        (complex (aref components (* index count))
                                 (aref components (1+ (* index count))))
                        (subseq components (* index count) (* (1+ index) count))))))
          (let ((components (or into (make-array (* count limit)
                                                  :element-type (lisp-element-type transfer)))))
            (unless (and (zerop (mod (length components) count))
                         (<= (length components) (* count limit)))
              (gpu-fail "~d components do not fit from an array of ~d ~a"
                        (length components) limit (type-description type)))
            (load-into components)
            components)))))

(defun free (array)
  "Release ARRAY's device memory now. Freeing twice is harmless. Freeing a view does nothing."
  (let ((buffer (and (not (gpu-array-view-p array)) (array-buffer array))))
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
