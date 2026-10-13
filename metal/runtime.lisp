(in-package #:cl-gpu/metal)

;;; Conditions

(define-condition metal-error (gpu:gpu-error) ())

(defun metal-error-message (condition)
  (gpu:gpu-error-message condition))

(define-condition metal-compile-error (metal-error) ())

(defun fail (control &rest arguments)
  (error 'metal-error :message (apply #'format nil control arguments)))

(defun take-string (pointer)
  "The Lisp string at the malloc'd C string POINTER, which is freed. NIL for a null pointer."
  (unless (cffi:null-pointer-p pointer)
    (unwind-protect (cffi:foreign-string-to-lisp pointer)
      (%free-string pointer))))

(defmacro with-error-slot ((slot) &body body)
  "Run BODY with SLOT bound to a char** and return the message the callee stored, or NIL, as a second value."
  `(cffi:with-foreign-object (,slot :pointer)
     (setf (cffi:mem-ref ,slot :pointer) (cffi:null-pointer))
     (let ((result (progn ,@body)))
       (values result (take-string (cffi:mem-ref ,slot :pointer))))))

;;; Objects

(defclass metal-object ()
  ((handle :initarg :handle :accessor handle)))

(defgeneric release (object)
  (:documentation "Free OBJECT's Metal resources. Releasing twice is harmless.")
  (:method ((object metal-object))
    (when (handle object)
      (%release (handle object))
      (setf (handle object) nil))
    nil))

(defun live-handle (object)
  (or (handle object) (fail "~(~a~) was released" (type-of object))))

(defclass metal-device (metal-object)
  ((queue :initarg :queue :reader device-queue)))

(defclass metal-buffer (metal-object) ())
(defclass metal-library (metal-object)
  ((device :initarg :device :reader library-device)))
(defclass metal-pipeline (metal-object)
  ((device :initarg :device :reader pipeline-device)))
(defclass metal-command (metal-object) ())

;;; Device

(defvar *default-device* nil)

(defun default-device ()
  "The system's default Metal device, opened on first use."
  (or *default-device*
      (let ((handle (%device-default)))
        (when (cffi:null-pointer-p handle)
          (fail "no Metal device"))
        (setf *default-device*
              (make-instance 'metal-device :handle handle :queue (%queue-new handle))))))

(defun device-name (device)
  (take-string (%device-name (live-handle device))))

(defun device-max-threads (device)
  (%device-max-threads (live-handle device)))

(defmethod release ((device metal-device))
  (when (handle device)
    (%release (device-queue device))
    (when (eq device *default-device*)
      (setf *default-device* nil)))
  (call-next-method))

;;; Element encodings

(defun element-name (type)
  (let ((type (gpu:parse-type type)))
    (if (typep type 'gpu:vector-type)
        (gpu:scalar-type-name (gpu:vector-type-element type))
        (gpu:scalar-type-name type))))

(defun element-layout (spec)
  "The scalar name, component count and bytes per element of the array element type SPEC."
  (let ((type (gpu:parse-type spec)))
    (unless (typep type '(or gpu:scalar-type gpu:vector-type))
      (fail "expected a scalar or vector type, got ~s" spec))
    (let* ((name (element-name type))
           (count (if (typep type 'gpu:vector-type) (gpu:vector-type-length type) 1)))
      (values name count (* (element-bytes name) (if (= count 3) 4 count))))))

(defun element-ctype (name)
  (ecase name
    ((gpu:f16 gpu:bf16 gpu:u16) :uint16)
    (gpu:f32 :float) (gpu:f64 :double)
    (gpu:i8 :int8) (gpu:i16 :int16) (gpu:i32 :int32) (gpu:i64 :int64)
    ((gpu:u8 gpu:bool) :uint8) (gpu:u32 :uint32) (gpu:u64 :uint64)))

(defun element-bytes (name)
  (cffi:foreign-type-size (element-ctype name)))

(defun single-bits (value)
  (ldb (byte 32 0) (sb-kernel:single-float-bits (coerce value 'single-float))))

(defun bits-single (bits)
  (sb-kernel:make-single-float (if (logbitp 31 bits) (- bits (ash 1 32)) bits)))

(defun shift-round (integer shift)
  "INTEGER shifted right by SHIFT bits, rounding to nearest even."
  (let* ((quotient (ash integer (- shift)))
         (remainder (ldb (byte shift 0) integer))
         (half (ash 1 (1- shift))))
    (if (or (> remainder half) (and (= remainder half) (oddp quotient)))
        (1+ quotient)
        quotient)))

(defun bf16-bits (value)
  (shift-round (single-bits value) 16))

(defun bits-bf16 (bits)
  (bits-single (ash bits 16)))

(defun f16-bits (value)
  (let* ((bits (single-bits value))
         (sign (ash (ldb (byte 1 31) bits) 15))
         (biased (ldb (byte 8 23) bits))
         (exponent (- biased 127))
         (mantissa (ldb (byte 23 0) bits)))
    (logior sign
            (cond ((= biased 255) (if (zerop mantissa) #x7c00 #x7e00))
                  ((> exponent 15) #x7c00)
                  ((>= exponent -14)
                   (shift-round (logior (ash (+ exponent 15) 23) mantissa) 13))
                  ((>= exponent -25)
                   (shift-round (logior #x800000 mantissa) (- -1 exponent)))
                  (t 0)))))

(defun bits-f16 (bits)
  (let ((sign (if (logbitp 15 bits) -1 1))
        (exponent (ldb (byte 5 10) bits))
        (mantissa (ldb (byte 10 0) bits)))
    (cond ((= exponent 31)
           (if (zerop mantissa)
               (if (plusp sign)
                   sb-ext:single-float-positive-infinity
                   sb-ext:single-float-negative-infinity)
               (fail "half-precision NaN has no Lisp value")))
          ((zerop exponent) (* sign (scale-float (coerce mantissa 'single-float) -24)))
          (t (* sign (scale-float (coerce (+ 1024 mantissa) 'single-float) (- exponent 25)))))))

(defun encode-element (name value)
  (ecase name
    (gpu:f16 (f16-bits value))
    (gpu:bf16 (bf16-bits value))
    (gpu:f32 (coerce value 'single-float))
    (gpu:f64 (coerce value 'double-float))
    (gpu:bool (if value 1 0))
    ((gpu:i8 gpu:i16 gpu:i32 gpu:i64 gpu:u8 gpu:u16 gpu:u32 gpu:u64) value)))

(defun decode-element (name raw)
  (ecase name
    (gpu:f16 (bits-f16 raw))
    (gpu:bf16 (bits-bf16 raw))
    (gpu:bool (/= raw 0))
    ((gpu:f32 gpu:f64 gpu:i8 gpu:i16 gpu:i32 gpu:i64 gpu:u8 gpu:u16 gpu:u32 gpu:u64) raw)))

(defun write-element (pointer name value offset)
  (setf (cffi:mem-ref pointer (element-ctype name) offset) (encode-element name value)))

(defun read-element (pointer name offset)
  (decode-element name (cffi:mem-ref pointer (element-ctype name) offset)))

;;; Buffers

(defun make-buffer (device bytes)
  "A zeroed BYTES-long buffer in memory shared by the host and device."
  ;; TODO: private storage with blit copies for discrete GPUs (#15)
  (let ((handle (%buffer-new (live-handle device) bytes)))
    (when (cffi:null-pointer-p handle)
      (fail "could not allocate a ~d-byte buffer" bytes))
    (let ((buffer (make-instance 'metal-buffer :handle handle)))
      (cffi:foreign-funcall "memset" :pointer (%buffer-contents handle) :int 0
                                     :unsigned-long bytes :pointer)
      buffer)))

(defun buffer-size (buffer)
  (%buffer-length (live-handle buffer)))

(defun buffer-pointer (buffer)
  "The host address of BUFFER's contents. Valid until BUFFER is released."
  (%buffer-contents (live-handle buffer)))

(defun check-fits (buffer elements stride)
  (when (> (* elements stride) (buffer-size buffer))
    (fail "~d elements of ~d bytes do not fit a ~d-byte buffer" elements stride (buffer-size buffer))))

(defun transfer-elements (buffer sequence count stride)
  "The number of elements SEQUENCE's components make, checked against BUFFER."
  (unless (zerop (mod (length sequence) count))
    (fail "~d components are not whole elements of ~d components" (length sequence) count))
  (let ((elements (/ (length sequence) count)))
    (check-fits buffer elements stride)
    elements))

(defun component-offset (index count stride width)
  "Byte offset of component INDEX of a flat sequence of elements of COUNT components."
  (multiple-value-bind (element component) (floor index count)
    (+ (* element stride) (* component width))))

(defun copy-to-buffer (buffer sequence &key (element-type 'gpu:f32))
  "Store the numbers in SEQUENCE in BUFFER from its start as ELEMENT-TYPE, a scalar or vector type.
A vector's components are consecutive in SEQUENCE."
  (multiple-value-bind (name count stride) (element-layout element-type)
    (transfer-elements buffer sequence count stride)
    (let ((pointer (buffer-pointer buffer))
          (width (element-bytes name))
          (index -1))
      (map nil (lambda (value)
                 (write-element pointer name value
                                (component-offset (incf index) count stride width)))
           sequence))
    buffer))

(defun copy-from-buffer (buffer sequence &key (element-type 'gpu:f32))
  "Fill SEQUENCE with the components of elements of ELEMENT-TYPE read from the start of BUFFER."
  (multiple-value-bind (name count stride) (element-layout element-type)
    (transfer-elements buffer sequence count stride)
    (let ((pointer (buffer-pointer buffer))
          (width (element-bytes name)))
      (dotimes (index (length sequence))
        (setf (elt sequence index)
              (read-element pointer name (component-offset index count stride width)))))
    sequence))

;;; Libraries and pipelines

(defun compile-library (device source)
  "Compile Metal SOURCE. Signals METAL-COMPILE-ERROR with the compiler's log on failure."
  (multiple-value-bind (handle message)
      (with-error-slot (slot)
        (%library-new (live-handle device) source slot))
    (when (cffi:null-pointer-p handle)
      (error 'metal-compile-error :message (or message "library compilation failed")))
    (make-instance 'metal-library :handle handle :device device)))

(defun make-pipeline (library name)
  "The compute pipeline for the kernel function NAME of LIBRARY."
  (multiple-value-bind (handle message)
      (with-error-slot (slot)
        (%pipeline-new (live-handle (library-device library)) (live-handle library) name slot))
    (when (cffi:null-pointer-p handle)
      (fail "~a" (or message "pipeline creation failed")))
    (make-instance 'metal-pipeline :handle handle :device (library-device library))))

(defun pipeline-max-threads (pipeline)
  (%pipeline-max-threads (live-handle pipeline)))

(defun pipeline-execution-width (pipeline)
  (%pipeline-execution-width (live-handle pipeline)))

;;; Dispatch

(defun grid-sizes (grid)
  (let ((sizes (if (listp grid) grid (list grid))))
    (unless (<= 1 (length sizes) 3)
      (fail "a grid has one to three dimensions, got ~s" grid))
    (append sizes (make-list (- 3 (length sizes)) :initial-element 1))))

(defun default-group (pipeline sizes)
  (let ((width (min (pipeline-execution-width pipeline)
                    (pipeline-max-threads pipeline)
                    (max 1 (first sizes)))))
    (list width 1 1)))

(defun pack-bytes (type value)
  "Foreign memory holding VALUE as TYPE, and its size. The caller frees the memory."
  (let* ((type (gpu:parse-type type))
         (vectorp (typep type 'gpu:vector-type))
         (name (element-name type))
         (stride (element-bytes name))
         (count (if vectorp (gpu:vector-type-length type) 1))
         (size (* stride (if (= count 3) 4 count)))
         (pointer (cffi:foreign-alloc :uint8 :count size :initial-element 0)))
    (handler-bind ((error (lambda (condition)
                            (declare (ignore condition))
                            (cffi:foreign-free pointer))))
      (if vectorp
          (progn
            (unless (= (length value) count)
              (fail "~s needs ~d components, got ~s" (gpu:type-spec type) count value))
            (dotimes (index count)
              (write-element pointer name (elt value index) (* stride index))))
          (write-element pointer name value 0)))
    (values pointer size)))

(defun dispatch (pipeline arguments grid &key group (wait t))
  "Run PIPELINE over GRID threads. Returns a command, after it has finished when WAIT.
ARGUMENTS are in kernel parameter order: a buffer, (:bytes type value) or (:threadgroup bytes)."
  (let* ((sizes (grid-sizes grid))
         (group (grid-sizes (or group (default-group pipeline sizes))))
         (count (length arguments))
         (owned '()))
    (cffi:with-foreign-objects ((kinds :int (max count 1))
                                (values :pointer (max count 1))
                                (lengths :unsigned-long (max count 1))
                                (grid-array :unsigned-long 3)
                                (group-array :unsigned-long 3))
      (unwind-protect
           (progn
             (loop for argument in arguments
                   for index from 0
                   do (etypecase argument
                        (metal-buffer
                         (setf (cffi:mem-aref kinds :int index) 0
                               (cffi:mem-aref values :pointer index) (live-handle argument)
                               (cffi:mem-aref lengths :unsigned-long index) 0))
                        (cons
                         (ecase (first argument)
                           (:bytes
                            (multiple-value-bind (pointer size)
                                (pack-bytes (second argument) (third argument))
                              (push pointer owned)
                              (setf (cffi:mem-aref kinds :int index) 1
                                    (cffi:mem-aref values :pointer index) pointer
                                    (cffi:mem-aref lengths :unsigned-long index) size)))
                           (:threadgroup
                            (setf (cffi:mem-aref kinds :int index) 2
                                  (cffi:mem-aref values :pointer index) (cffi:null-pointer)
                                  (cffi:mem-aref lengths :unsigned-long index) (second argument)))))))
             (loop for size in sizes for index from 0
                   do (setf (cffi:mem-aref grid-array :unsigned-long index) size))
             (loop for size in group for index from 0
                   do (setf (cffi:mem-aref group-array :unsigned-long index) size))
             (multiple-value-bind (handle message)
                 (with-error-slot (slot)
                   (%dispatch (device-queue (pipeline-device pipeline)) (live-handle pipeline)
                              count kinds values lengths grid-array group-array slot))
               (when (cffi:null-pointer-p handle)
                 (fail "~a" (or message "dispatch failed")))
               (let ((command (make-instance 'metal-command :handle handle)))
                 (when wait
                   (wait command))
                 command)))
        (mapc #'cffi:foreign-free owned)))))

(defun command-done-p (command)
  (or (null (handle command))
      (/= 0 (%command-done-p (handle command)))))

(defmethod wait ((command metal-command))
  "Block until COMMAND finishes. Signals METAL-ERROR if it failed. The command is released."
  (when (handle command)
    (unwind-protect
         (multiple-value-bind (status message)
             (with-error-slot (slot)
               (%command-wait (handle command) slot))
           (unless (zerop status)
             (fail "~a" (or message "command failed"))))
      (release command)))
  nil)

;;; Scoped resources

(defmacro define-with-macro (name function)
  `(defmacro ,name ((variable &rest arguments) &body body)
     `(let ((,variable (,',function ,@arguments)))
        (unwind-protect (progn ,@body)
          (release ,variable)))))

(define-with-macro with-buffer make-buffer)
(define-with-macro with-library compile-library)
(define-with-macro with-pipeline make-pipeline)
