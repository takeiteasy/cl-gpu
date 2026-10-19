(in-package #:cl-gpu)

(defclass opencl-printer (c-printer)
  ((extensions :initform nil :accessor opencl-extensions)))

(defmethod printer-target ((printer opencl-printer)) :opencl)

(defmethod reserved-words ((printer opencl-printer))
  (append *msl-reserved-words*
          '("restrict" "typedef" "_Bool" "__kernel" "__global" "__local" "__constant" "__private")
          (loop for base in '("float" "double" "half" "int" "uint" "char" "uchar" "short" "ushort" "long" "ulong")
                append (loop for n in '(2 3 4 8 16) collect (format nil "~a~d" base n)))))

(defmethod scalar-text ((printer opencl-printer) type)
  (case (scalar-type-name type)
    (f64 "double")
    (bf16 (target-fail "bf16 is not available on opencl"))
    (t (scalar-text (make-instance 'msl-printer) type))))

(defmethod atomic-text ((printer opencl-printer) type)
  (format nil "volatile ~a" (scalar-text printer (atomic-type-element type))))

(defmethod space-text ((printer opencl-printer) space)
  (ecase space (:global "__global") (:constant "__constant")
         (:shared "__local") (:private "__private")))

(defmethod cast-text ((printer opencl-printer) type operand)
  (if (vector-type-p type)
      (format nil "convert_~a(~a)" (type-text printer type) operand)
      (call-next-method)))

(defmethod bitcast-text ((printer opencl-printer) type operand)
  (format nil "as_~a(~a)" (type-text printer type) operand))

(defmethod begin-kernel :after ((printer opencl-printer) kernel)
  (setf (opencl-extensions printer)
        (remove-duplicates
         (loop for type in (ir-kernel-scalar-types kernel)
               for support = (type-support type :opencl)
               when (stringp support) collect support) :test #'equal))
  (dolist (parameter (ir-kernel-parameters kernel))
    (let ((type (ir-variable-type parameter)))
      (when (or (member (scalar 'bool) (type-scalars type))
                (and (pointer-type-p type) (eq :private (pointer-type-space type))))
        (target-fail "invalid OpenCL parameter type ~a" (type-description type))))))

(defmethod print-prelude ((printer opencl-printer) stream)
  (dolist (extension (sort (copy-list (opencl-extensions printer)) #'string<))
    (format stream "#ifndef ~a~%#error Required extension ~a~%#endif~%#pragma OPENCL EXTENSION ~a : enable~%"
            extension extension extension))
  (dolist (helper (reverse (printer-helpers printer)))
    (write-string helper stream)))

(defmethod print-signature ((printer opencl-printer) kernel stream)
  (format stream "__kernel void ~a(~{~a~^, ~}) {~%"
          (sanitised-kernel-name printer kernel)
          (loop for parameter in (ir-kernel-parameters kernel)
                collect (format nil "~a ~a" (type-text printer (ir-variable-type parameter))
                                (variable-name printer parameter)))))

(defmethod render ((printer opencl-printer) (expr ir-construct))
  (values (format nil "(~a)(~{~a~^, ~})" (type-text printer (ir-expr-type expr))
                  (mapcar (lambda (arg) (expr-text printer arg)) (ir-construct-args expr)))
          +postfix-precedence+))

(defmethod render ((printer opencl-printer) (expr ir-intrinsic))
  (let ((axis (getf (ir-intrinsic-options expr) :axis)))
    (when axis
      (return-from render
        (values (format nil "~a(~d)"
                        (nth-value 1 (intrinsic-spelling (ir-intrinsic-intrinsic expr) :opencl)) axis)
                +postfix-precedence+))))
  (if (eq '%floor-mod (intrinsic-name (ir-intrinsic-intrinsic expr)))
      (let* ((type (ir-expr-type expr))
             (text (type-text printer type))
             (name (format nil "cl_gpu_mod_~a" text))
             (floatp (eq :float (scalar-type-kind (first (type-scalars type)))))
             (helper (format nil "inline ~a ~a(~a a, ~a b) {~%    ~a r = ~a;~%    return ~a;~%}~%"
                             text name text text text
                             (if floatp "fmod(a, b)" "a % b")
                             (if (vector-type-p type)
                                 (format nil "select(r, r + b, (r != (~a)(0)) & ((r < (~a)(0)) != (b < (~a)(0))))" text text text)
                                 "r != 0 && ((r < 0) != (b < 0)) ? r + b : r"))))
        (pushnew helper (printer-helpers printer) :test #'equal)
        (values (format nil "~a(~{~a~^, ~})" name
                        (mapcar (lambda (arg) (expr-text printer arg)) (ir-intrinsic-args expr)))
                +postfix-precedence+))
      (call-next-method)))

(defmethod print-prologue ((printer opencl-printer) kernel)
  (let* ((arrays (shared-arrays kernel))
         (cleared (remove-if-not #'ir-array-zero arrays)))
    (dolist (array arrays)
      (emit-line printer "~a;" (array-storage-text printer array)))
    (dolist (array cleared)
      (emit-line printer "for (size_t cl_gpu_i = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2)); cl_gpu_i < ~d; cl_gpu_i += get_local_size(0) * get_local_size(1) * get_local_size(2)) {"
                 (ir-array-size array))
      (emit-line printer "    ~a[cl_gpu_i] = 0;" (array-storage-name printer (ir-array-variable array)))
      (emit-line printer "}"))
    (when cleared (emit-line printer "barrier(CLK_LOCAL_MEM_FENCE);"))))

(defmethod print-statement ((printer opencl-printer) (statement ir-array))
  (let ((variable (ir-array-variable statement)))
    (unless (shared-array-p statement)
      (emit-line printer "~a~:[~; = {0}~];" (array-storage-text printer statement) (ir-array-zero statement)))
    (emit-line printer "~a ~a = ~a;" (type-text printer (ir-variable-type variable))
               (variable-name printer variable) (array-storage-name printer variable))))

(setf (getf *printer-classes* :opencl) 'opencl-printer)
