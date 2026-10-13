(in-package #:cl-gpu)

;;; A printer writes an IR-KERNEL as C-like source. C-PRINTER holds what the targets
;;; share; a target subclasses it and fills in the generic functions below.

(defclass c-printer ()
  ((stream :accessor printer-stream)
   (indent :initform 0 :accessor printer-indent)
   (names :initform (make-hash-table :test 'eq) :accessor printer-names)
   (helpers :initform '() :accessor printer-helpers)))

(defgeneric printer-target (printer)
  (:documentation "The keyword naming the target: :metal, :cuda or :opencl."))

(defgeneric reserved-words (printer)
  (:documentation "Names a parameter cannot take on this target."))

(defgeneric scalar-text (printer type))
(defgeneric atomic-text (printer type))
(defgeneric space-text (printer space))

(defgeneric cast-text (printer type operand)
  (:documentation "OPERAND, text of a value, converted to TYPE.")
  (:method ((printer c-printer) type operand)
    (format nil "(~a)(~a)" (type-text printer type) operand)))

(defgeneric bitcast-text (printer type operand)
  (:documentation "OPERAND, text of a value, reinterpreted as TYPE."))

(defgeneric parameter-text (printer attribute axis)
  (:documentation "A thread position that the target takes as a kernel parameter."))

(defgeneric helper-definition (printer name)
  (:documentation "The source of the helper function NAME."))

(defgeneric print-prelude (printer stream)
  (:documentation "Headers and the helpers the kernel used."))

(defgeneric print-signature (printer kernel stream)
  (:documentation "The kernel's header line, ending with an opening brace."))

(defgeneric begin-kernel (printer kernel)
  (:documentation "Reset PRINTER and name the parameters of KERNEL.")
  (:method ((printer c-printer) kernel)
    (setf (printer-indent printer) 1
          (printer-names printer) (make-hash-table :test 'eq)
          (printer-helpers printer) '())
    (name-parameters printer kernel)))

(defgeneric print-prologue (printer kernel)
  (:documentation "Statements ahead of the kernel body.")
  (:method ((printer c-printer) kernel)
    (declare (ignore printer kernel))))

(defgeneric type-text (printer type)
  (:method (printer (type scalar-type)) (scalar-text printer type))
  (:method (printer (type atomic-type)) (atomic-text printer type))
  (:method (printer (type vector-type))
    (format nil "~a~d" (scalar-text printer (vector-type-element type)) (vector-type-length type)))
  (:method (printer (type pointer-type))
    (format nil "~a ~a*"
            (space-text printer (pointer-type-space type))
            (type-text printer (pointer-type-pointee type)))))

(defun target-fail (control &rest args)
  (error 'kernel-target-error :message (apply #'format nil control args)))

;;; Names

(defun sanitise-name (name)
  "NAME as a C identifier of lowercase letters, digits and single underscores."
  (let ((text (with-output-to-string (out)
                (let ((previous nil))
                  (loop for char across (string name)
                        for ascii = (and (< (char-code char) 128) (alphanumericp char))
                        for next = (if ascii (char-downcase char) #\_)
                        unless (and (char= next #\_) (eql previous #\_))
                          do (write-char next out)
                        do (setf previous next))))))
    (if (and (plusp (length text)) (alpha-char-p (char text 0)))
        text
        (concatenate 'string "v" text))))

(defun intrinsic-names (target)
  "The function names the intrinsics call on TARGET."
  (let ((names '()))
    (maphash (lambda (key intrinsic)
               (declare (ignore key))
               (destructuring-bind (&optional kind text &rest options)
                   (getf (intrinsic-spellings intrinsic) target)
                 (declare (ignore kind))
                 (when text (push text names))
                 (loop for (nil value) on options by #'cddr
                       when (stringp value) do (push value names))))
             *intrinsics*)
    names))

(defun reserved-name-p (printer name)
  (or (eql 0 (search "cl_gpu_" name))
      (member name (reserved-words printer) :test #'string=)
      (member name (intrinsic-names (printer-target printer)) :test #'string=)))

(defun local-name-p (name)
  "Whether NAME ends in an underscore and a number, as a local's name does."
  (let ((underscore (position #\_ name :from-end t)))
    (and underscore
         (< (1+ underscore) (length name))
         (every #'digit-char-p (subseq name (1+ underscore))))))

(defun parameter-name (printer variable)
  (let ((name (sanitise-name (ir-variable-name variable))))
    (if (or (reserved-name-p printer name) (local-name-p name))
        (concatenate 'string name "_")
        name)))

(defun name-parameters (printer kernel)
  (let ((taken '()))
    (dolist (parameter (ir-kernel-parameters kernel))
      (let ((name (parameter-name printer parameter)))
        (when (member name taken :test #'string=)
          (target-fail "parameters ~(~a~) clash as ~a" (ir-variable-name parameter) name))
        (push name taken)
        (setf (gethash parameter (printer-names printer)) name)))))

(defun variable-name (printer variable)
  (or (gethash variable (printer-names printer))
      (setf (gethash variable (printer-names printer))
            (format nil "~a_~d" (sanitise-name (ir-variable-name variable))
                    (ir-variable-id variable)))))

(defun sanitised-kernel-name (printer kernel)
  (let ((name (sanitise-name (ir-kernel-name kernel))))
    (if (reserved-name-p printer name)
        (concatenate 'string name "_")
        name)))

;;; Validation

(defun check-target-support (printer kernel)
  (let ((target (printer-target printer)))
    (dolist (type (ir-kernel-scalar-types kernel))
      (let ((support (type-support type target)))
        (unless (eq support t)
          (target-fail "~a is not available on ~(~a~)~@[ without ~a~]"
                       (type-description type) target support))))
    (dolist (statement (ir-kernel-body kernel))
      (map-ir (lambda (node)
                (when (and (ir-intrinsic-p node)
                           (null (intrinsic-spelling (ir-intrinsic-intrinsic node) target)))
                  (target-fail "~(~a~) is not available on ~(~a~)"
                               (intrinsic-name (ir-intrinsic-intrinsic node)) target)))
              statement))))

;;; Expressions. RENDER returns the text and its precedence, which is higher for tighter binding.

(defparameter *binary-operators*
  '((:or "||" 2) (:and "&&" 3) (:bit-or "|" 4) (:bit-xor "^" 5) (:bit-and "&" 6)
    (:eq "==" 7) (:ne "!=" 7) (:lt "<" 8) (:le "<=" 8) (:gt ">" 8) (:ge ">=" 8)
    (:shl "<<" 9) (:shr ">>" 9) (:add "+" 10) (:sub "-" 10)
    (:mul "*" 11) (:div "/" 11) (:rem "%" 11)))

(defparameter *unary-operators* '((:neg "-") (:not "!") (:bit-not "~")))

(defconstant +ternary-precedence+ 1)
(defconstant +unary-precedence+ 12)
(defconstant +postfix-precedence+ 13)
(defconstant +atom-precedence+ 14)

;;; C promotes these to int, so a result that can leave the type's range is cast back.
(defparameter *narrowing-operators* '(:add :sub :mul :div :neg :shl :bit-not))

(defun narrow-type-p (type)
  (and (scalar-type-p type)
       (member (scalar-type-kind type) '(:signed :unsigned))
       (< (scalar-type-bits type) 32)))

(defun expr-text (printer expr &optional (minimum 0) parent)
  "The text of EXPR, in parentheses if it binds looser than MINIMUM or mixes with PARENT's operator."
  (multiple-value-bind (text precedence) (render printer expr)
    (if (or (< precedence minimum)
            (and parent
                 (ir-op-p expr)
                 (not (eq (ir-op-operator expr) parent))
                 (<= 4 precedence 9)))
        (format nil "(~a)" text)
        text)))

(defgeneric render (printer expr))

(defmethod render ((printer c-printer) (expr ir-var))
  (values (variable-name printer (ir-var-variable expr)) +atom-precedence+))

(defmethod render ((printer c-printer) (expr ir-op))
  (let* ((operator (ir-op-operator expr))
         (operands (ir-op-operands expr))
         (type (ir-expr-type expr)))
    (multiple-value-bind (text precedence)
        (let ((unary (second (assoc operator *unary-operators*))))
          (if unary
              (let ((operand (expr-text printer (first operands) +unary-precedence+)))
                (values (format nil "~a~a" unary
                                (if (and (eq operator :neg) (char= (char operand 0) #\-))
                                    (format nil "(~a)" operand)
                                    operand))
                        +unary-precedence+))
              (destructuring-bind (spelling precedence) (rest (assoc operator *binary-operators*))
                (let ((mixing (and (<= 4 precedence 9) operator)))
                  (values (format nil "~a ~a ~a"
                                  (expr-text printer (first operands) precedence mixing)
                                  spelling
                                  (expr-text printer (second operands) (1+ precedence) mixing))
                          precedence)))))
      (if (and (narrow-type-p type) (member operator *narrowing-operators*))
          (values (cast-text printer type text) +unary-precedence+)
          (values text precedence)))))

(defmethod render ((printer c-printer) (expr ir-ternary))
  (values (format nil "~a ? ~a : ~a"
                  (expr-text printer (ir-ternary-test expr) (1+ +ternary-precedence+))
                  (expr-text printer (ir-ternary-then expr) (1+ +ternary-precedence+))
                  (expr-text printer (ir-ternary-else expr) (1+ +ternary-precedence+)))
          +ternary-precedence+))

(defmethod render ((printer c-printer) (expr ir-index))
  (values (format nil "~a[~a]"
                  (expr-text printer (ir-index-base expr) +postfix-precedence+)
                  (expr-text printer (ir-index-index expr)))
          +postfix-precedence+))

(defmethod render ((printer c-printer) (expr ir-component))
  (values (format nil "~a.~a"
                  (expr-text printer (ir-component-base expr) +postfix-precedence+)
                  (char "xyzw" (ir-literal-value (ir-component-index expr))))
          +postfix-precedence+))

(defmethod render ((printer c-printer) (expr ir-swizzle))
  (values (format nil "~a.~{~a~}"
                  (expr-text printer (ir-swizzle-base expr) +postfix-precedence+)
                  (mapcar (lambda (index) (char "xyzw" index)) (ir-swizzle-components expr)))
          +postfix-precedence+))

(defmethod render ((printer c-printer) (expr ir-construct))
  (values (format nil "~a(~{~a~^, ~})" (type-text printer (ir-expr-type expr))
                  (mapcar (lambda (arg) (expr-text printer arg)) (ir-construct-args expr)))
          +postfix-precedence+))

(defmethod render ((printer c-printer) (expr ir-convert))
  (let ((operand (expr-text printer (ir-convert-operand expr)))
        (type (ir-expr-type expr)))
    (values (ecase (ir-convert-kind expr)
              (:convert (cast-text printer type operand))
              (:bitcast (bitcast-text printer type operand)))
            +unary-precedence+)))

(defmethod render ((printer c-printer) (expr ir-intrinsic))
  (let* ((intrinsic (ir-intrinsic-intrinsic expr))
         (options (ir-intrinsic-options expr))
         (args (ir-intrinsic-args expr))
         (arg-texts (mapcar (lambda (arg) (expr-text printer arg)) args)))
    (multiple-value-bind (kind text spelling)
        (intrinsic-spelling intrinsic (printer-target printer)
                            (and args (ir-expr-type (first args))))
      (ecase kind
        (:param
         (values (parameter-text printer text (getf options :axis)) +postfix-precedence+))
        ((:call :helper)
         (when (eq kind :helper)
           (pushnew text (printer-helpers printer) :test #'string=))
         (let ((scope (getf options :scope)))
           (values (format nil "~a(~{~a~^, ~})" text
                           (append arg-texts
                                   (getf spelling :extra)
                                   (and scope (list (getf spelling scope)))))
                   +postfix-precedence+)))))))

;;; Literals

(defun float-spelling (value type)
  "VALUE spelled with a decimal point or exponent, so that it reads back as the same float."
  (when (or (sb-ext:float-infinity-p value) (sb-ext:float-nan-p value))
    (target-fail "~a literal ~a has no spelling" (type-description type) value))
  (let ((*read-default-float-format* (if (typep value 'double-float) 'double-float 'single-float)))
    (prin1-to-string value)))

(defun float-literal-text (printer type value)
  (let ((digits (float-spelling value type)))
    (ecase (scalar-type-name type)
      (f16 (format nil "~ah" digits))
      (f32 (format nil "~af" digits))
      (f64 digits)
      (bf16 (cast-text printer type (format nil "~af" digits))))))

(defun integer-literal-text (printer type value)
  (let* ((name (scalar-type-name type))
         (suffix (ecase name
                   ((i8 i16 u8 u16 i32) "")
                   (u32 "u") (i64 "l") (u64 "ul")))
         (minimum (and (member name '(i32 i64)) (- (expt 2 (1- (scalar-type-bits type))))))
         (text (if (eql value minimum)
                   (format nil "(~d~a - 1)" (1+ value) suffix)
                   (format nil "~d~a" value suffix))))
    (if (narrow-type-p type)
        (cast-text printer type text)
        text)))

(defmethod render ((printer c-printer) (expr ir-literal))
  (let ((type (ir-expr-type expr))
        (value (ir-literal-value expr)))
    (let ((text (case (scalar-type-kind type)
                  (:bool (if value "true" "false"))
                  (:float (float-literal-text printer type value))
                  (t (integer-literal-text printer type value)))))
      (values text (if (and (realp value) (minusp value)) +unary-precedence+ +atom-precedence+)))))

;;; Statements

(defun emit-line (printer control &rest args)
  (let ((stream (printer-stream printer)))
    (dotimes (i (printer-indent printer))
      (write-string "    " stream))
    (apply #'format stream control args)
    (terpri stream)))

(defun print-block (printer statements)
  (incf (printer-indent printer))
  (dolist (statement statements)
    (print-statement printer statement))
  (decf (printer-indent printer)))

(defun assign-text (printer assign)
  (format nil "~a = ~a"
          (expr-text printer (ir-assign-target assign))
          (expr-text printer (ir-assign-value assign))))

(defgeneric print-statement (printer statement))

(defmethod print-statement ((printer c-printer) (statement ir-declare))
  (let ((variable (ir-declare-variable statement)))
    (emit-line printer "~a ~a~@[ = ~a~];"
               (type-text printer (ir-variable-type variable))
               (variable-name printer variable)
               (and (ir-declare-init statement)
                    (expr-text printer (ir-declare-init statement))))))

(defmethod print-statement ((printer c-printer) (statement ir-assign))
  (emit-line printer "~a;" (assign-text printer statement)))

(defun print-if (printer statement head)
  (emit-line printer "~a (~a) {" head (expr-text printer (ir-if-test statement)))
  (print-block printer (ir-if-then statement))
  (let ((else (ir-if-else statement)))
    (cond ((null else)
           (emit-line printer "}"))
          ((and (null (rest else)) (ir-if-p (first else)))
           (print-if printer (first else) "} else if"))
          (t
           (emit-line printer "} else {")
           (print-block printer else)
           (emit-line printer "}")))))

(defmethod print-statement ((printer c-printer) (statement ir-if))
  (print-if printer statement "if"))

(defmethod print-statement ((printer c-printer) (statement ir-for))
  (let ((variable (ir-for-variable statement)))
    (emit-line printer "for (~a ~a = ~a; ~a; ~@[~a~]) {"
               (type-text printer (ir-variable-type variable))
               (variable-name printer variable)
               (expr-text printer (ir-for-init statement))
               (expr-text printer (ir-for-test statement))
               (and (ir-for-step statement) (assign-text printer (ir-for-step statement))))
    (print-block printer (ir-for-body statement))
    (emit-line printer "}")))

(defmethod print-statement ((printer c-printer) (statement ir-while))
  (emit-line printer "while (~a) {" (expr-text printer (ir-while-test statement)))
  (print-block printer (ir-while-body statement))
  (emit-line printer "}"))

(defmethod print-statement ((printer c-printer) (statement ir-break))
  (emit-line printer "break;"))

(defmethod print-statement ((printer c-printer) (statement ir-return))
  (emit-line printer "return;"))

(defmethod print-statement ((printer c-printer) (statement ir-expr-stmt))
  (emit-line printer "~a;" (expr-text printer (ir-expr-stmt-expr statement))))

;;; Kernels

(defun print-kernel (kernel printer &optional (stream *standard-output*))
  "Write KERNEL, an IR-KERNEL, to STREAM as the source PRINTER's target reads."
  (check-target-support printer kernel)
  (begin-kernel printer kernel)
  (let ((body (with-output-to-string (out)
                (setf (printer-stream printer) out)
                (print-prologue printer kernel)
                (dolist (statement (ir-kernel-body kernel))
                  (print-statement printer statement)))))
    (setf (printer-stream printer) stream)
    (print-prelude printer stream)
    (print-signature printer kernel stream)
    (write-string body stream)
    (format stream "}~%")))

(defparameter *printer-classes* '(:metal msl-printer))

(defun make-printer (target)
  (let ((class (getf *printer-classes* target)))
    (unless class
      (target-fail "no printer for target ~s" target))
    (make-instance class)))

(defun kernel-source (kernel target)
  "The source of the IR-KERNEL KERNEL for TARGET, as a string."
  (with-output-to-string (stream)
    (print-kernel kernel (make-printer target) stream)))

(defun kernel-entry-name (kernel target)
  "The name of the entry function that KERNEL's source defines for TARGET."
  (sanitised-kernel-name (make-printer target) kernel))
