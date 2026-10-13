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
  ((queue :initarg :queue :reader device-queue)
   (stream :initarg :stream :accessor device-stream)
   (mps-support :initform :unknown :accessor device-mps-support)
   (staging :initform nil :accessor device-staging)
   (staging-lock :initform (bt:make-lock "metal staging") :reader device-staging-lock)))

(defclass metal-buffer (metal-object)
  ((device :initarg :device :reader buffer-device)
   (storage :initarg :storage :reader buffer-storage)
   (size :initarg :size :reader buffer-bytes)))
(defclass metal-library (metal-object)
  ((device :initarg :device :reader library-device)))
(defclass metal-pipeline (metal-object)
  ((device :initarg :device :reader pipeline-device)))

;;; Device

(defvar *default-device* nil)

(defun default-device ()
  "The system's default Metal device, opened on first use."
  (or *default-device*
      (let ((handle (%device-default)))
        (when (cffi:null-pointer-p handle)
          (fail "no Metal device"))
        (let ((queue (%queue-new handle)))
          (setf *default-device*
                (make-instance 'metal-device :handle handle :queue queue
                                             :stream (%stream-new queue)))))))

(defun device-name (device)
  (take-string (%device-name (live-handle device))))

(defun device-unified-memory-p (device)
  "True if the host and DEVICE share memory, as on Apple silicon."
  (/= 0 (%device-unified-memory (live-handle device))))

(defun device-max-threads (device)
  (%device-max-threads (live-handle device)))

(defun device-max-threadgroup-memory (device)
  (%device-max-threadgroup-memory (live-handle device)))

(defun synchronize-device (device)
  "Block until every dispatch on DEVICE has finished. Signals METAL-ERROR for the first that failed."
  (when (device-stream device)
    (multiple-value-bind (status message)
        (with-error-slot (slot)
          (%stream-wait (device-stream device) slot))
      (unless (zerop status)
        (fail "~a" (or message "command failed")))))
  nil)

(defmethod release ((device metal-device))
  (unwind-protect (when (handle device)
                    (synchronize-device device))
    (when (handle device)
      (when (device-staging device)
        (release (device-staging device))
        (setf (device-staging device) nil))
      (%release (device-stream device))
      (setf (device-stream device) nil)
      (%release (device-queue device))
      (when (eq device *default-device*)
        (setf *default-device* nil)))
    (call-next-method))
  nil)

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

(defmacro with-constant-ctype ((variable name) &body body)
  "Run BODY with VARIABLE a literal C type keyword, so CFFI:MEM-REF compiles to a direct access."
  `(ecase (element-ctype ,name)
     ,@(loop for ctype in '(:uint8 :int8 :uint16 :int16 :uint32 :int32 :uint64 :int64 :float :double)
             collect `(,ctype ,@(subst ctype variable body)))))

(defun element-bytes (name)
  (cffi:foreign-type-size (element-ctype name)))

(defun single-bits (value)
  (ldb (byte 32 0) (float-features:single-float-bits (coerce value 'single-float))))

(defun bits-single (bits)
  (float-features:bits-single-float bits))

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
                   float-features:single-float-positive-infinity
                   float-features:single-float-negative-infinity)
               (fail "half-precision NaN has no Lisp value")))
          ((zerop exponent) (* sign (scale-float (coerce mantissa 'single-float) -24)))
          (t (* sign (scale-float (coerce (+ 1024 mantissa) 'single-float) (- exponent 25)))))))

(defun element-encoder (name)
  "A function turning a Lisp number into the bits a buffer holds for scalar NAME."
  (ecase name
    (gpu:f16 #'f16-bits)
    (gpu:bf16 #'bf16-bits)
    (gpu:f32 (lambda (value) (coerce value 'single-float)))
    (gpu:f64 (lambda (value) (coerce value 'double-float)))
    (gpu:bool (lambda (value) (if value 1 0)))
    ((gpu:i8 gpu:i16 gpu:i32 gpu:i64 gpu:u8 gpu:u16 gpu:u32 gpu:u64) #'identity)))

(defun element-decoder (name)
  (ecase name
    (gpu:f16 #'bits-f16)
    (gpu:bf16 #'bits-bf16)
    (gpu:bool (lambda (raw) (/= raw 0)))
    ((gpu:f32 gpu:f64 gpu:i8 gpu:i16 gpu:i32 gpu:i64 gpu:u8 gpu:u16 gpu:u32 gpu:u64) #'identity)))

(defun write-element (pointer name value offset)
  (setf (cffi:mem-ref pointer (element-ctype name) offset) (funcall (element-encoder name) value)))

(defun read-element (pointer name offset)
  (funcall (element-decoder name) (cffi:mem-ref pointer (element-ctype name) offset)))

;;; Buffers

(defvar *buffer-storage* :auto
  "The storage MAKE-BUFFER uses unless told otherwise: :shared, :private, or :auto to share memory
only where the host and device already do.")

(defun resolve-storage (device storage)
  (case storage
    (:auto (if (device-unified-memory-p device) :shared :private))
    ((:shared :private) storage)
    (t (fail "storage is :auto, :shared or :private, got ~s" storage))))

(defun align-4 (bytes)
  "BYTES rounded up to a multiple of 4, which macOS requires of blit sizes."
  (* 4 (ceiling bytes 4)))

(defun check-blit (status message what)
  (unless (zerop status)
    (fail "~a" (or message what))))

(defun blit-copy (device source source-offset target target-offset length)
  "Queue a copy of LENGTH bytes between buffers of DEVICE."
  (multiple-value-bind (status message)
      (with-error-slot (slot)
        (%stream-copy (device-stream device) (live-handle source) source-offset
                      (live-handle target) target-offset length slot))
    (check-blit status message "copy failed")))

(defun blit-zero (device buffer length)
  (multiple-value-bind (status message)
      (with-error-slot (slot)
        (%stream-fill (device-stream device) (live-handle buffer) length slot))
    (check-blit status message "fill failed")))

(defun make-buffer (device bytes &key (storage *buffer-storage*))
  "A zeroed BYTES-long buffer. Shared storage is memory the host and device share; private storage
is the device's own, reached by COPY-TO-BUFFER and COPY-FROM-BUFFER. STORAGE is :shared, :private
or :auto, which is private unless the device shares memory with the host."
  (let* ((storage (resolve-storage device storage))
         (privatep (eq storage :private))
         (handle (%buffer-new (live-handle device) (if privatep (align-4 bytes) bytes)
                              (if privatep 1 0))))
    (when (cffi:null-pointer-p handle)
      (fail "could not allocate a ~d-byte buffer" bytes))
    (let ((buffer (make-instance 'metal-buffer :handle handle :device device
                                               :storage storage :size bytes))
          (done nil))
      (unwind-protect
           (progn
             (if privatep
                 (blit-zero device buffer (align-4 bytes))
                 (cffi:foreign-funcall "memset" :pointer (%buffer-contents handle) :int 0
                                                :unsigned-long bytes :pointer))
             (setf done t))
        (unless done
          (release buffer)))
      buffer)))

(defun buffer-size (buffer)
  (live-handle buffer)
  (buffer-bytes buffer))

(defun buffer-pointer (buffer)
  "The host address of the shared BUFFER's contents, once the device has finished its dispatches.
Valid until BUFFER is released. A private buffer has no host address."
  (when (eq (buffer-storage buffer) :private)
    (fail "a private buffer has no host address; use copy-to-buffer and copy-from-buffer"))
  ;; TODO: waits for the whole device, not only the dispatches that use BUFFER (#49)
  (synchronize-device (buffer-device buffer))
  (%buffer-contents (live-handle buffer)))

(defun check-fits (buffer elements stride offset)
  (when (> (+ offset (* elements stride)) (buffer-size buffer))
    (fail "~d elements of ~d bytes from byte ~d do not fit a ~d-byte buffer"
          elements stride offset (buffer-size buffer))))

(defun transfer-elements (buffer sequence count stride offset)
  "The number of elements SEQUENCE's components make, checked against BUFFER from byte OFFSET."
  (unless (zerop (mod (length sequence) count))
    (fail "~d components are not whole elements of ~d components" (length sequence) count))
  (let ((elements (/ (length sequence) count)))
    (check-fits buffer elements stride offset)
    elements))

(defun component-offsets (count stride width)
  "A function returning the byte offset of each component in turn of a flat sequence of elements of
COUNT components, STRIDE bytes apart."
  (let ((base 0)
        (component 0))
    (lambda ()
      (prog1 (+ base (* component width))
        (when (= (incf component) count)
          (setf component 0)
          (incf base stride))))))

;;; Private buffers are reached through the device's shared staging buffer.
;;; TODO: one staging buffer under one lock, and every upload waits for the device first; a ring of
;;; staging buffers would let uploads overlap pending work (#51)

(defun staging-buffer (device bytes)
  "The device's staging buffer, grown to hold BYTES. The caller holds the staging lock."
  (let ((staging (device-staging device)))
    (if (and staging (>= (buffer-size staging) bytes))
        staging
        (progn
          (when staging
            (release staging))
          (setf (device-staging device)
                (make-buffer device (max 4096 (ash 1 (integer-length (1- bytes))))
                             :storage :shared))))))

(defun upload-private (buffer offset bytes store)
  "Fill BYTES of private BUFFER from byte OFFSET by calling STORE on the host address in the staging
buffer where they start. Returns once the copy is queued. Blits move whole 4-byte words, so the
bytes around the range are read back and written unchanged."
  (let* ((device (buffer-device buffer))
         (start (* 4 (floor offset 4)))
         (end (align-4 (+ offset bytes)))
         (window (- end start)))
    (when (plusp bytes)
      (bt:with-lock-held ((device-staging-lock device))
        (synchronize-device device)
        (let* ((staging (staging-buffer device window))
               (pointer (%buffer-contents (live-handle staging))))
          (cffi:foreign-funcall "memset" :pointer pointer :int 0 :unsigned-long window :pointer)
          (when (or (/= offset start) (< (+ offset bytes) end))
            (blit-copy device buffer start staging 0 4)
            (blit-copy device buffer (- end 4) staging (- window 4) 4)
            (synchronize-device device))
          (funcall store (cffi:inc-pointer pointer (- offset start)))
          (blit-copy device staging 0 buffer start window))))))

(defun download-private (buffer offset bytes load)
  "Call LOAD on the host address in the staging buffer of the BYTES of private BUFFER from byte OFFSET."
  (let* ((device (buffer-device buffer))
         (start (* 4 (floor offset 4)))
         (window (- (align-4 (+ offset bytes)) start)))
    (when (plusp bytes)
      (bt:with-lock-held ((device-staging-lock device))
        (let ((staging (staging-buffer device window)))
          (blit-copy device buffer start staging 0 window)
          (synchronize-device device)
          (funcall load (cffi:inc-pointer (%buffer-contents (live-handle staging))
                                          (- offset start))))))))

(defun raw-lisp-type (name)
  "The Lisp type whose vectors hold scalar NAME exactly as a buffer does, or NIL if values need
encoding or the implementation stores that type boxed."
  (let ((type (case name
                (gpu:f32 'single-float)
                (gpu:f64 'double-float)
                (gpu:i8 '(signed-byte 8)) (gpu:i16 '(signed-byte 16))
                (gpu:i32 '(signed-byte 32)) (gpu:i64 '(signed-byte 64))
                (gpu:u8 '(unsigned-byte 8)) (gpu:u16 '(unsigned-byte 16))
                (gpu:u32 '(unsigned-byte 32)) (gpu:u64 '(unsigned-byte 64)))))
    (when (and type (equal (upgraded-array-element-type type) type))
      type)))

(defun raw-vector-p (sequence name)
  "True if SEQUENCE is a simple vector whose bytes are the buffer encoding of scalar NAME."
  (let ((type (raw-lisp-type name)))
    (and type (typep sequence `(simple-array ,type (*))))))

(defun sequence-decoder (sequence name)
  "A function turning the bits of scalar NAME into a value SEQUENCE can hold."
  (let ((decode (element-decoder name))
        (type (if (vectorp sequence) (array-element-type sequence) t)))
    (if (eq type t)
        decode
        (lambda (raw) (coerce (funcall decode raw) type)))))

(defun copy-bytes (target source length)
  (cffi:foreign-funcall "memcpy" :pointer target :pointer source :unsigned-long length :pointer))

(defun bulk-copy (pointer vector count stride width tobuffer)
  "Copy the components of the raw VECTOR to or from the buffer memory at POINTER, a flat sequence
of elements of COUNT components laid out STRIDE bytes apart."
  (cffi:with-pointer-to-vector-data (host vector)
    (flet ((copy (buffer-offset host-offset length)
             (let ((buffer (cffi:inc-pointer pointer buffer-offset))
                   (host (cffi:inc-pointer host host-offset)))
               (if tobuffer
                   (copy-bytes buffer host length)
                   (copy-bytes host buffer length)))))
      (if (= count 3)
          (dotimes (element (floor (length vector) 3))
            (copy (* element stride) (* element 3 width) (* 3 width)))
          (copy 0 0 (* (length vector) width))))))

(defun copy-to-buffer (buffer sequence &key (element-type 'gpu:f32) (offset 0))
  "Store the numbers in SEQUENCE in BUFFER from byte OFFSET as ELEMENT-TYPE, a scalar or vector type.
A vector's components are consecutive in SEQUENCE. A simple vector specialised to the Lisp type of
the elements' scalar (e.g. single-float for f32) is copied in bulk."
  (multiple-value-bind (name count stride) (element-layout element-type)
    (let ((elements (transfer-elements buffer sequence count stride offset))
          (width (element-bytes name)))
      (flet ((store (pointer)
               (if (raw-vector-p sequence name)
                   (bulk-copy pointer sequence count stride width t)
                   ;; TODO: one funcall of the encoder per value; loops typed on the source
                   ;; element would be ~25x faster (#60), and f16/bf16 need a converting loop (#61)
                   (with-constant-ctype (ctype name)
                     (let ((encode (element-encoder name))
                           (next (component-offsets count stride width)))
                       (map nil (lambda (value)
                                  (setf (cffi:mem-ref pointer ctype (funcall next))
                                        (funcall encode value)))
                            sequence))))))
        (if (eq (buffer-storage buffer) :private)
            (upload-private buffer offset (* elements stride) #'store)
            (store (cffi:inc-pointer (buffer-pointer buffer) offset))))))
  buffer)

(defun copy-from-buffer (buffer sequence &key (element-type 'gpu:f32) (offset 0))
  "Fill SEQUENCE with the components of elements of ELEMENT-TYPE read from byte OFFSET of BUFFER.
A simple vector specialised as for COPY-TO-BUFFER is filled in bulk; any other specialised vector
receives the values coerced to its element type."
  (multiple-value-bind (name count stride) (element-layout element-type)
    (let ((elements (transfer-elements buffer sequence count stride offset))
          (width (element-bytes name)))
      (flet ((fetch (pointer)
               (if (raw-vector-p sequence name)
                   (bulk-copy pointer sequence count stride width nil)
                   (with-constant-ctype (ctype name)
                     (let ((decode (sequence-decoder sequence name))
                           (next (component-offsets count stride width)))
                       (map-into sequence
                                 (lambda ()
                                   (funcall decode (cffi:mem-ref pointer ctype (funcall next))))))))))
        (if (eq (buffer-storage buffer) :private)
            (download-private buffer offset (* elements stride) #'fetch)
            (fetch (cffi:inc-pointer (buffer-pointer buffer) offset))))))
  sequence)

;;; Libraries and pipelines

(defun compile-library (device source &key precise-math)
  "Compile Metal SOURCE, without fast math when PRECISE-MATH. Signals METAL-COMPILE-ERROR with the
compiler's log on failure."
  (multiple-value-bind (handle message)
      (with-error-slot (slot)
        (%library-new (live-handle device) source (if precise-math 1 0) slot))
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

(defun dispatch (pipeline arguments grid &key group wait)
  "Queue PIPELINE over GRID threads on its device and return NIL; with WAIT, then wait for the
device's dispatches. Dispatches run in order. A failure in one is signalled by the next wait.
ARGUMENTS are in kernel parameter order: a buffer, (:buffer buffer byte-offset), (:bytes type value)
or (:threadgroup bytes)."
  (let* ((device (pipeline-device pipeline))
         (sizes (grid-sizes grid))
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
                           (:buffer
                            (setf (cffi:mem-aref kinds :int index) 0
                                  (cffi:mem-aref values :pointer index) (live-handle (second argument))
                                  (cffi:mem-aref lengths :unsigned-long index) (third argument)))
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
             (multiple-value-bind (status message)
                 (with-error-slot (slot)
                   (%stream-dispatch (device-stream device) (live-handle pipeline)
                                     count kinds values lengths grid-array group-array slot))
               (unless (zerop status)
                 (fail "~a" (or message "dispatch failed")))))
        (mapc #'cffi:foreign-free owned)))
    (when wait
      (synchronize-device device))
    nil))

;;; Scoped resources

(defmacro define-with-macro (name function)
  `(defmacro ,name ((variable &rest arguments) &body body)
     `(let ((,variable (,',function ,@arguments)))
        (unwind-protect (progn ,@body)
          (release ,variable)))))

(define-with-macro with-buffer make-buffer)
(define-with-macro with-library compile-library)
(define-with-macro with-pipeline make-pipeline)
