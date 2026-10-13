(in-package #:cl-gpu)

;;; Metal Shading Language. Every parameter is bound by position: pointers and
;;; scalars take the next [[buffer(i)]], and :shared pointers take the next
;;; [[threadgroup(k)]]. Thread positions are extra uint3 parameters.

(defclass msl-printer (c-printer)
  ((positions :initform '() :accessor printer-positions)
   (copies :initform '() :accessor printer-copies)))

(defmethod printer-target ((printer msl-printer)) :metal)

(defparameter *msl-reserved-words*
  '("alignas" "alignof" "and" "asm" "auto" "bool" "break" "case" "catch" "char" "class"
    "const" "constant" "constexpr" "continue" "default" "delete" "device" "do" "double"
    "else" "enum" "explicit" "extern" "false" "float" "for" "fragment" "friend" "goto"
    "half" "bfloat" "if" "inline" "int" "kernel" "long" "main" "mutable" "namespace"
    "new" "nullptr" "operator" "or" "private" "protected" "public" "register" "return"
    "short" "signed" "sizeof" "static" "struct" "switch" "template" "this" "thread"
    "threadgroup" "throw" "true" "try" "typedef" "typename" "uchar" "uint" "ulong"
    "union" "unsigned" "ushort" "using" "vertex" "virtual" "void" "volatile" "while"
    "atomic_int" "atomic_uint" "metal" "as_type" "static_cast"))

(defmethod reserved-words ((printer msl-printer))
  (append *msl-reserved-words*
          (loop for base in '("float" "half" "bfloat" "int" "uint" "char" "uchar"
                              "short" "ushort" "long" "ulong" "bool")
                append (loop for n from 2 to 4 collect (format nil "~a~d" base n)))))

(defmethod scalar-text ((printer msl-printer) type)
  (ecase (scalar-type-name type)
    (f16 "half") (bf16 "bfloat") (f32 "float")
    (i8 "char") (i16 "short") (i32 "int") (i64 "long")
    (u8 "uchar") (u16 "ushort") (u32 "uint") (u64 "ulong")
    (bool "bool")))

(defmethod atomic-text ((printer msl-printer) type)
  (if (eq (atomic-type-element type) (scalar 'i32)) "atomic_int" "atomic_uint"))

(defmethod space-text ((printer msl-printer) space)
  (ecase space
    (:global "device") (:shared "threadgroup") (:constant "constant") (:private "thread")))

(defmethod cast-text ((printer msl-printer) type operand)
  (format nil "static_cast<~a>(~a)" (type-text printer type) operand))

(defmethod bitcast-text ((printer msl-printer) type operand)
  (format nil "as_type<~a>(~a)" (type-text printer type) operand))

(defun position-name (printer attribute)
  (pushnew attribute (printer-positions printer) :test #'string=)
  (format nil "cl_gpu_~a" attribute))

(defmethod parameter-text ((printer msl-printer) attribute axis)
  (format nil "~a.~a" (position-name printer attribute) (char "xyz" axis)))

;;; Metal's compare-exchange writes the old value to its expected operand and may fail spuriously.
(defun atomic-cas-helpers ()
  (with-output-to-string (out)
    (dolist (space '("device" "threadgroup"))
      (loop for (atomic type) in '(("atomic_int" "int") ("atomic_uint" "uint"))
            do (format out "inline ~a cl_gpu_atomic_cas(~a ~a* p, ~a expected, ~a desired) {
    ~a seen = expected;
    while (!atomic_compare_exchange_weak_explicit(p, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
        if (expected != seen) return expected;
    }
    return seen;
}
"
                       type space atomic type type type)))))

;;; Floored remainder: the truncated one, moved by B when its sign differs from B's. The
;;; add only happens then, so it cannot overflow. select makes it work on vectors too.
(defparameter *floor-mod-helper* "template <typename T>
inline T cl_gpu_mod(T a, T b) {
    T r = a % b;
    return select(r, r + b, (r != T(0)) & ((r < T(0)) != (b < T(0))));
}
")

(defparameter *floor-mod-float-helper* "template <typename T>
inline T cl_gpu_fmod(T a, T b) {
    T r = fmod(a, b);
    return select(r, r + b, (r != T(0)) & ((r < T(0)) != (b < T(0))));
}
")

(defmethod helper-definition ((printer msl-printer) name)
  (cond ((string= name "cl_gpu_atomic_cas") (atomic-cas-helpers))
        ((string= name "cl_gpu_mod") *floor-mod-helper*)
        ((string= name "cl_gpu_fmod") *floor-mod-float-helper*)
        (t (error "unknown helper ~a" name))))

(defmethod print-prelude ((printer msl-printer) stream)
  (format stream "#include <metal_stdlib>~%using namespace metal;~%~%")
  (dolist (helper (sort (copy-list (printer-helpers printer)) #'string<))
    (format stream "~a~%" (helper-definition printer helper))))

;;; A scalar or vector parameter is a constant reference, so one that the kernel assigns
;;; is copied into a local of its own name.
(defun assigned-parameters (kernel)
  (let ((found '()))
    (dolist (statement (ir-kernel-body kernel))
      (map-ir (lambda (node)
                (when (ir-assign-p node)
                  (let ((target (ir-assign-target node)))
                    (loop while (ir-component-p target)
                          do (setf target (ir-component-base target)))
                    (when (ir-var-p target)
                      (pushnew (ir-var-variable target) found)))))
              statement))
    (remove-if (lambda (parameter) (pointer-type-p (ir-variable-type parameter)))
               (intersection found (ir-kernel-parameters kernel)))))

;;; Metal's fmod returns float for a bfloat.
(defun check-bfloat-remainders (kernel)
  (dolist (statement (ir-kernel-body kernel))
    (map-ir (lambda (node)
              (when (and (ir-intrinsic-p node)
                         (member (intrinsic-name (ir-intrinsic-intrinsic node)) '(%fmod %floor-mod))
                         (eq (scalar 'bf16)
                             (first (type-scalars (ir-expr-type node)))))
                (target-fail "rem and mod are not available on bf16 on metal")))
            statement)))

(defmethod begin-kernel :after ((printer msl-printer) kernel)
  (check-bfloat-remainders kernel)
  (setf (printer-positions printer) '()
        (printer-copies printer) (assigned-parameters kernel)))

;;; Threadgroup arrays are declared at kernel scope. A cleared one is filled by the whole
;;; group before any thread continues, ahead of any return that would skip the barrier.
(defun element-zero-statement (printer array)
  (let* ((storage (array-storage-name printer (ir-array-variable array)))
         (element (pointer-type-pointee (ir-variable-type (ir-array-variable array)))))
    (if (atomic-type-p element)
        (format nil "atomic_store_explicit(&~a[cl_gpu_i], 0, memory_order_relaxed);" storage)
        (format nil "~a[cl_gpu_i] = ~a(0);" storage (type-text printer element)))))

(defun print-shared-arrays (printer kernel)
  (let ((arrays (shared-arrays kernel)))
    (dolist (array arrays)
      (emit-line printer "~a;" (array-storage-text printer array)))
    (let ((cleared (remove-if-not #'ir-array-zero arrays)))
      (when cleared
        (let ((index (position-name printer "thread_index_in_threadgroup"))
              (size (position-name printer "threads_per_threadgroup")))
          (dolist (array cleared)
            (emit-line printer "for (uint cl_gpu_i = ~a; cl_gpu_i < ~d; cl_gpu_i += ~a.x * ~a.y * ~a.z) {"
                       index (ir-array-size array) size size size)
            (emit-line printer "    ~a" (element-zero-statement printer array))
            (emit-line printer "}")))
        (emit-line printer "threadgroup_barrier(mem_flags::mem_threadgroup);")))))

(defmethod print-prologue ((printer msl-printer) kernel)
  (dolist (parameter (ir-kernel-parameters kernel))
    (when (member parameter (printer-copies printer))
      (let ((name (variable-name printer parameter)))
        (emit-line printer "~a ~a = cl_gpu_~a;"
                   (type-text printer (ir-variable-type parameter)) name name))))
  (print-shared-arrays printer kernel))

(defun msl-parameters (printer kernel)
  (let ((buffers -1)
        (groups -1))
    (loop for parameter in (ir-kernel-parameters kernel)
          for type = (ir-variable-type parameter)
          for name = (variable-name printer parameter)
          collect (if (member parameter (printer-copies printer))
                      (format nil "constant ~a& cl_gpu_~a [[buffer(~d)]]"
                              (type-text printer type) name (incf buffers))
                      (typecase type
                        (pointer-type
                         (ecase (pointer-type-space type)
                           ((:global :constant)
                            (format nil "~a ~a [[buffer(~d)]]"
                                    (type-text printer type) name (incf buffers)))
                           (:shared
                            (format nil "~a ~a [[threadgroup(~d)]]"
                                    (type-text printer type) name (incf groups)))
                           (:private
                            (target-fail "parameter ~(~a~) points to private memory"
                                         (ir-variable-name parameter)))))
                        (t (format nil "constant ~a& ~a [[buffer(~d)]]"
                                   (type-text printer type) name (incf buffers))))))))

(defmethod print-signature ((printer msl-printer) kernel stream)
  (let ((parameters (append (msl-parameters printer kernel)
                            (loop for attribute in (sort (copy-list (printer-positions printer))
                                                         #'string<)
                                  collect (format nil "~a cl_gpu_~a [[~a]]"
                                                  (if (string= attribute "thread_index_in_threadgroup")
                                                      "uint"
                                                      "uint3")
                                                  attribute attribute)))))
    (format stream "kernel void ~a(~{~%    ~a~^,~}) {~%"
            (sanitised-kernel-name printer kernel) parameters)))
