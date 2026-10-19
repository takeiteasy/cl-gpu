(defpackage #:cl-gpu/transfer
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu))
  (:export
   #:element-name #:element-layout #:element-ctype #:with-constant-ctype #:element-bytes
   #:single-bits #:bits-single #:shift-round #:bf16-bits #:bits-bf16
   #:f16-bits #:bits-f16 #:element-encoder #:element-decoder #:write-element
   #:read-element #:raw-lisp-type #:raw-vector-p #:sequence-decoder #:copy-bytes
   #:bulk-copy #:pack-bytes #:component-offsets))

(in-package #:cl-gpu/transfer)

(defun fail (control &rest arguments)
  (error 'gpu:gpu-error :message (apply #'format nil control arguments)))

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
