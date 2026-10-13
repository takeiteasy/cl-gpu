(in-package #:cl-gpu)

(defstruct (gpu-type (:constructor nil)))

(defstruct (scalar-type (:include gpu-type) (:constructor make-scalar-type (name kind bits)))
  name kind bits)

(defstruct (vector-type (:include gpu-type) (:constructor make-vector-type (element length)))
  element length)

(defstruct (pointer-type (:include gpu-type) (:constructor make-pointer-type (pointee space)))
  pointee space)

(defstruct (atomic-type (:include gpu-type) (:constructor make-atomic-type (element)))
  element)

;;; Types are interned, so equal types are EQ.
(defvar *types* (make-hash-table :test 'equal))

(defun intern-type (key constructor)
  (or (gethash key *types*)
      (setf (gethash key *types*) (funcall constructor))))

(defparameter *scalar-definitions*
  '((f16 :float 16) (bf16 :float 16) (f32 :float 32) (f64 :float 64)
    (i8 :signed 8) (i16 :signed 16) (i32 :signed 32) (i64 :signed 64)
    (u8 :unsigned 8) (u16 :unsigned 16) (u32 :unsigned 32) (u64 :unsigned 64)
    (bool :bool 8)))

(loop for (name kind bits) in *scalar-definitions*
      do (intern-type (list :scalar name) (lambda () (make-scalar-type name kind bits))))

(defparameter *address-spaces* '(:global :shared :constant :private))

(defun scalar (name)
  (or (gethash (list :scalar name) *types*)
      (error "unknown scalar type ~s" name)))

(defun type-name-symbol-p (symbol)
  (and (gethash (list :scalar symbol) *types*) t))

(defun type-spec (type)
  "The specifier that PARSE-TYPE reads back as TYPE."
  (etypecase type
    (scalar-type (scalar-type-name type))
    (vector-type `(vec ,(type-spec (vector-type-element type)) ,(vector-type-length type)))
    (pointer-type `(ptr ,(type-spec (pointer-type-pointee type)) ,(pointer-type-space type)))
    (atomic-type `(atomic ,(type-spec (atomic-type-element type))))))

(defun type-description (type)
  "TYPE spelled for messages."
  (if type
      (let ((*package* (find-package '#:cl-gpu)))
        (format nil "~(~s~)" (type-spec type)))
      "nothing"))

(defmethod print-object ((type gpu-type) stream)
  (print-unreadable-object (type stream :type t)
    (write-string (type-description type) stream)))

(defun type-failure (spec control &rest args)
  (error 'kernel-type-error
         :message (apply #'format nil control args)
         :form spec
         :location *current-source*))

(defun parse-type (spec)
  "The type that SPEC names: a scalar name, (vec T N), (ptr T SPACE) or (atomic T)."
  (cond
    ((gpu-type-p spec) spec)
    ((symbolp spec)
     (or (gethash (list :scalar spec) *types*)
         (type-failure spec "unknown type ~s" spec)))
    ((and (consp spec) (null (cdr (last spec))))
     (case (first spec)
       (vec (parse-vector-type spec))
       (ptr (parse-pointer-type spec))
       (atomic (parse-atomic-type spec))
       (t (type-failure spec "unknown type ~s" spec))))
    (t (type-failure spec "unknown type ~s" spec))))

(defun parse-vector-type (spec)
  (unless (= (length spec) 3)
    (type-failure spec "a vector type is (vec element length)"))
  (let ((element (parse-type (second spec)))
        (length (third spec)))
    (unless (and (scalar-type-p element) (numeric-type-p element))
      (type-failure spec "vector elements must be numeric scalars, got ~a"
                    (type-description element)))
    (unless (member length '(2 3 4))
      (type-failure spec "vector length must be 2, 3 or 4, got ~s" length))
    (intern-type (list :vec element length)
                 (lambda () (make-vector-type element length)))))

(defun parse-pointer-type (spec)
  (unless (= (length spec) 3)
    (type-failure spec "a pointer type is (ptr pointee address-space)"))
  (let ((pointee (parse-type (second spec)))
        (space (third spec)))
    (when (pointer-type-p pointee)
      (type-failure spec "pointers to pointers are not supported"))
    (unless (member space *address-spaces*)
      (type-failure spec "unknown address space ~s, expected one of ~{~s~^, ~}"
                    space *address-spaces*))
    (intern-type (list :ptr pointee space)
                 (lambda () (make-pointer-type pointee space)))))

(defun parse-atomic-type (spec)
  (unless (= (length spec) 2)
    (type-failure spec "an atomic type is (atomic element)"))
  (let ((element (parse-type (second spec))))
    (unless (member element (list (scalar 'i32) (scalar 'u32)))
      (type-failure spec "atomic elements must be i32 or u32, got ~a"
                    (type-description element)))
    (intern-type (list :atomic element)
                 (lambda () (make-atomic-type element)))))

;;; Predicates look through vectors to the element type.
(defun element-kind (type)
  (typecase type
    (scalar-type (scalar-type-kind type))
    (vector-type (scalar-type-kind (vector-type-element type)))))

(defun float-type-p (type) (eq (element-kind type) :float))
(defun signed-type-p (type) (eq (element-kind type) :signed))
(defun unsigned-type-p (type) (eq (element-kind type) :unsigned))
(defun integer-type-p (type) (and (member (element-kind type) '(:signed :unsigned)) t))
(defun numeric-type-p (type) (and (member (element-kind type) '(:float :signed :unsigned)) t))
(defun bool-type-p (type) (eq type (scalar 'bool)))

(defun integer-range (type)
  "The CL type of the values of the integer scalar TYPE."
  (let ((bits (scalar-type-bits type)))
    (if (eq (scalar-type-kind type) :signed)
        `(signed-byte ,bits)
        `(unsigned-byte ,bits))))

(defun type-bytes (type)
  "Size of a numeric scalar or vector in bytes, or NIL. A 3-vector takes the space of a 4-vector."
  (typecase type
    (scalar-type (unless (eq (scalar-type-kind type) :bool)
                   (/ (scalar-type-bits type) 8)))
    (vector-type (* (type-bytes (vector-type-element type))
                    (if (= (vector-type-length type) 3) 4 (vector-type-length type))))))

(defun type-scalars (type)
  "The scalar types that TYPE is built from."
  (etypecase type
    (scalar-type (list type))
    (vector-type (list (vector-type-element type)))
    (pointer-type (type-scalars (pointer-type-pointee type)))
    (atomic-type (list (atomic-type-element type)))))

;;; T if supported, NIL if not, or the name of the extension to enable.
(defparameter *scalar-support*
  '((f16 :opencl "cl_khr_fp16")
    (bf16 :opencl nil)
    (f64 :metal nil :opencl "cl_khr_fp64")))

(defun type-support (type target)
  "Whether the scalar TYPE exists on TARGET (:metal, :cuda or :opencl): T, NIL or an extension name."
  (let ((entry (member target (rest (assoc (scalar-type-name type) *scalar-support*)))))
    (if entry (second entry) t)))
