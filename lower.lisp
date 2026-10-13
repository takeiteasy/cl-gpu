(in-package #:cl-gpu)

;;; Lowering turns the walker's expression-oriented AST into typed, C-like IR.
;;; Statements are emitted into *EMIT*; expressions are returned. A form in value
;;; position that needs statements (let, progn, block, a complex if) emits them
;;; first and returns an expression over temporaries.

(defvar *emit*)
(defvar *variables*)
(defvar *variable-id*)
(defvar *block-contexts*)
(defvar *loop-stack*)
(defvar *root-block*)

;;; How a block is left. KIND is :root (the kernel, left with a return), :loop or
;;; :once (a block run as a one-shot loop, left with a break). FLAG records that a
;;; return passed through inner loops; PENDING lists the targets whose flag is
;;; tested after this context's statement.
(defstruct (exit-context (:conc-name context-))
  kind want-value hint flag result pending)

(defun type-fail (node control &rest args)
  (error 'kernel-type-error
         :message (apply #'format nil control args)
         :location (or (and node (node-source node)) *current-source*)))

(defun emit (statement)
  (push statement *emit*))

(defmacro capturing (&body body)
  "Run BODY with fresh statement output. Return its value and the statements emitted."
  `(let ((*emit* '()))
     (let ((result (progn ,@body)))
       (values result (reverse *emit*)))))

(defmacro capturing-statements (&body body)
  `(nth-value 1 (capturing ,@body)))

(defun new-variable (name type &key temp)
  (make-ir-variable name type (incf *variable-id*) temp))

(defun var-expr (variable)
  (make-ir-var :variable variable :type (ir-variable-type variable)))

(defun literal-expr (value type)
  (make-ir-literal :value value :type type))

(defun true-literal ()
  (literal-expr t (scalar 'bool)))

(defun declare-variable (variable &optional init)
  (emit (make-ir-declare :variable variable :init init)))

(defun spill (expr &optional (name 'tmp))
  "Bind EXPR to a fresh temporary now. Return the temporary."
  (let ((variable (new-variable name (ir-expr-type expr) :temp t)))
    (declare-variable variable expr)
    (var-expr variable)))

(defun stable-p (expr)
  (or (ir-literal-p expr)
      (and (ir-var-p expr) (ir-variable-temp (ir-var-variable expr)))))

(defun stabilise (expr)
  (if (stable-p expr) expr (spill expr)))

(defun stabilise-place (place)
  "PLACE with its subexpressions bound to temporaries, so later statements cannot change them."
  (etypecase place
    (ir-var place)
    (ir-index (make-ir-index :base (stabilise (ir-index-base place))
                             :index (stabilise (ir-index-index place))
                             :type (ir-expr-type place)))
    (ir-component (make-ir-component :base (stabilise-place (ir-component-base place))
                                     :index (ir-component-index place)
                                     :type (ir-expr-type place)))))

(defun assemble-operands (exprs statement-lists)
  "Emit the statements of operands lowered separately. Operands evaluated before a later
operand's statements are bound to temporaries, keeping left-to-right order."
  (let ((last (position-if #'identity statement-lists :from-end t)))
    (loop for expr in exprs
          for statements in statement-lists
          for index from 0
          do (mapc #'emit statements)
          collect (if (and last (< index last))
                      (stabilise expr)
                      expr))))

(defun check-arity (node args min max)
  (let ((count (length args)))
    (unless (and (>= count min) (or (null max) (<= count max)))
      (type-fail node "~(~a~) takes ~a argument~:p, got ~d"
                 (call-operator node)
                 (cond ((eql min max) min)
                       (max (format nil "~d to ~d" min max))
                       (t (format nil "at least ~d" min)))
                 count))))

;;; Literals

(defun literal-class (node)
  (typecase (literal-value node)
    (integer :integer)
    (float :float)
    (null :bool)
    (t (when (eq (literal-value node) t) :bool))))

(defun default-literal-type (node)
  (let ((value (literal-value node)))
    (ecase (literal-class node)
      (:integer (cond ((typep value '(signed-byte 32)) (scalar 'i32))
                      ((typep value '(signed-byte 64)) (scalar 'i64))
                      (t (type-fail node "integer literal ~d is too large" value))))
      (:float (scalar (if (typep value 'double-float) 'f64 'f32)))
      (:bool (scalar 'bool)))))

(defun coerce-float (value type)
  (coerce value (if (eql (scalar-type-bits type) 64) 'double-float 'single-float)))

(defun adopt-literal (node type)
  "The literal NODE as an IR literal of TYPE, or NIL if its kind cannot take that type."
  (let ((value (literal-value node)))
    (when (scalar-type-p type)
      (ecase (literal-class node)
        (:integer
         (cond ((integer-type-p type)
                (unless (typep value (integer-range type))
                  (type-fail node "literal ~d does not fit in ~a" value (type-description type)))
                (literal-expr value type))
               ((float-type-p type)
                (literal-expr (coerce-float value type) type))))
        (:float
         (when (float-type-p type)
           (literal-expr (coerce-float value type) type)))
        (:bool
         (when (bool-type-p type)
           (literal-expr (and value t) type)))))))

(defun lower-literal (node hint)
  (unless (literal-class node)
    (type-fail node "~s is not a value" (literal-value node)))
  (or (and hint (adopt-literal node hint))
      (adopt-literal node (default-literal-type node))))

;;; Constant numeric forms take the type of what they meet, like literals.
(defparameter *constant-operators*
  '(+ - * / rem mod logand logior logxor lognot shl shr))

(defun constant-node-p (node)
  (typecase node
    (literal (and (member (literal-class node) '(:integer :float)) t))
    (call (and (member (call-operator node) *constant-operators*)
               (call-args node)
               (every #'constant-node-p (call-args node))))))

(defun default-constant-type (nodes)
  (let ((types '()))
    (dolist (node nodes)
      (map-nodes (lambda (child)
                   (when (typep child 'literal)
                     (push (default-literal-type child) types)))
                 node))
    (flet ((has (name) (member (scalar name) types)))
      (scalar (cond ((has 'f64) 'f64) ((has 'f32) 'f32) ((has 'i64) 'i64) (t 'i32))))))

(defun numeric-scalar-p (type)
  (and (scalar-type-p type) (numeric-type-p type)))

(defun broadcast (expr vector-type)
  (make-ir-construct :args (list expr) :type vector-type))

(defun lower-agreeing (nodes hint &key broadcast)
  "Lower NODES to expressions of one type. Return the expressions, the statements each
needed, and the type. Non-constant nodes decide the type; constants adopt it. With
BROADCAST, a vector among the nodes decides the type and a scalar of its element type is widened."
  (let* ((exprs (make-array (length nodes)))
         (statements (make-array (length nodes) :initial-element '()))
         (type nil)
         (lowered-types '()))
    (flet ((lower-at (index node hint)
             (multiple-value-bind (expr node-statements) (capturing (lower-value node hint))
               (setf (aref exprs index) expr
                     (aref statements index) node-statements)
               expr)))
      (loop for node in nodes
            for index from 0
            unless (constant-node-p node)
              do (let ((expr (lower-at index node (or type hint))))
                   (push (ir-expr-type expr) lowered-types)
                   (unless type (setf type (ir-expr-type expr)))))
      (when broadcast
        (setf type (or (find-if #'vector-type-p lowered-types) type)))
      (when (some #'constant-node-p nodes)
        (let ((constant-type (cond ((vector-type-p type) (vector-type-element type))
                                   (type)
                                   ((numeric-scalar-p hint) hint)
                                   (t (default-constant-type
                                       (remove-if-not #'constant-node-p nodes))))))
          (loop for node in nodes
                for index from 0
                when (constant-node-p node)
                  do (lower-at index node constant-type))
          (setf type (or type constant-type)))))
    (loop for node in nodes
          for index from 0
          for expr = (aref exprs index)
          unless (eq (ir-expr-type expr) type)
            do (if (and broadcast (vector-type-p type)
                        (eq (ir-expr-type expr) (vector-type-element type)))
                   (setf (aref exprs index) (broadcast expr type))
                   (type-fail node "expected ~a, got ~a"
                              (type-description type) (type-description (ir-expr-type expr)))))
    (values (coerce exprs 'list) (coerce statements 'list) type)))

(defun lower-sequence (nodes &optional hint-function)
  "Lower NODES in order. HINT-FUNCTION, given an index and the expressions so far, returns that node's hint."
  (let ((exprs '())
        (statements '()))
    (loop for node in nodes
          for index from 0
          do (multiple-value-bind (expr node-statements)
                 (capturing (lower-value node (and hint-function
                                                   (funcall hint-function index (reverse exprs)))))
               (push expr exprs)
               (push node-statements statements)))
    (assemble-operands (nreverse exprs) (nreverse statements))))

;;; Values and statements

(defun binding-variable (binding)
  (or (gethash binding *variables*)
      (error "binding ~s has no variable" binding)))

(defun lower-value (node hint)
  "Lower NODE, whose value is used, to an expression. HINT is the type a literal should take."
  (let* ((*current-source* (or (node-source node) *current-source*))
         (expr (etypecase node
                 (literal (lower-literal node hint))
                 (var-ref (var-expr (binding-variable (var-ref-binding node))))
                 (let-node (lower-let node t hint))
                 (assign (lower-assign node t))
                 (modify (lower-modify node t))
                 (if-node (lower-if node t hint))
                 (progn-node (lower-body (progn-node-body node) t hint))
                 (block-node (lower-block node t hint))
                 (loop-node (type-fail node "a loop has no value"))
                 (return-node (type-fail node "return cannot be used as a value"))
                 (call (lower-call node hint nil)))))
    (unless (ir-expr-type expr)
      (type-fail node "this form has no value"))
    expr))

(defun lower-stmt (node)
  "Lower NODE, whose value is not used, emitting statements."
  (let ((*current-source* (or (node-source node) *current-source*)))
    (etypecase node
      ((or literal var-ref) nil)
      (let-node (lower-let node nil nil))
      (assign (lower-assign node nil))
      (modify (lower-modify node nil))
      (if-node (lower-if node nil nil))
      (progn-node (lower-body (progn-node-body node) nil nil))
      (block-node (lower-block node nil nil))
      (loop-node (type-fail node "a loop must be the body of its block"))
      (return-node (lower-return node))
      (call (let ((expr (lower-call node nil t)))
              (when (and (ir-intrinsic-p expr)
                         (intrinsic-impure-p (ir-intrinsic-intrinsic expr)))
                (emit (make-ir-expr-stmt :expr expr))))))))

(defun lower-body (nodes want-value hint)
  "Lower NODES in sequence. When WANT-VALUE, return the last one's value."
  (when (and want-value (null nodes))
    (type-fail nil "an empty body has no value"))
  (loop for (node . more) on nodes
        do (if (and want-value (null more))
               (return (lower-value node hint))
               (lower-stmt node))))

(defun lower-test (node)
  (let ((expr (lower-value node (scalar 'bool))))
    (unless (bool-type-p (ir-expr-type expr))
      (type-fail node "expected bool, got ~a" (type-description (ir-expr-type expr))))
    expr))

;;; let

(defun declared-types (declarations)
  "Alist from variable name to type, from the (type T name...) declarations."
  (loop for declaration in declarations
        append (loop for specifier in (rest declaration)
                     when (and (consp specifier) (eq (first specifier) 'type))
                       append (let ((type (parse-type (second specifier))))
                                (mapcar (lambda (name) (cons name type))
                                        (cddr specifier))))))

(defun lower-let (node want-value hint)
  (let ((declared (declared-types (let-node-declarations node))))
    (loop for binding in (let-node-bindings node)
          for init in (let-node-inits node)
          do (let* ((declared-type (cdr (assoc (binding-name binding) declared)))
                    (expr (when init (lower-value init declared-type)))
                    (type (or declared-type (and expr (ir-expr-type expr)))))
               (unless type
                 (type-fail binding "variable ~(~a~) needs a type declaration or an initial value"
                            (binding-name binding)))
               (when (atomic-type-p type)
                 (type-fail binding "a variable cannot have an atomic type"))
               (when (and expr (not (eq type (ir-expr-type expr))))
                 (type-fail init "expected ~a, got ~a"
                            (type-description type) (type-description (ir-expr-type expr))))
               (let ((variable (new-variable (binding-name binding) type)))
                 (setf (gethash binding *variables*) variable)
                 (declare-variable variable expr)))))
  (lower-body (let-node-body node) want-value hint))

;;; Assignment

(defun lower-place (node)
  (let ((expr (lower-value node nil)))
    (unless (typep expr '(or ir-var ir-index ir-component))
      (type-fail node "this form is not a place"))
    expr))

(defun emit-store (place value want-value)
  "Emit the assignment of VALUE to PLACE. When WANT-VALUE, return an expression for the new value."
  (cond ((not want-value)
         (emit (make-ir-assign :target place :value value))
         nil)
        ((ir-var-p place)
         (emit (make-ir-assign :target place :value value))
         (var-expr (ir-var-variable place)))
        (t (let ((temp (spill value)))
             (emit (make-ir-assign :target place :value temp))
             temp))))

(defun lower-assign (node want-value)
  (let* ((place (lower-place (assign-target node)))
         (type (ir-expr-type place)))
    (multiple-value-bind (value statements)
        (capturing (lower-value (assign-value node) type))
      (unless (eq (ir-expr-type value) type)
        (type-fail (assign-value node) "expected ~a, got ~a"
                   (type-description type) (type-description (ir-expr-type value))))
      (when statements
        (setf place (stabilise-place place)))
      (mapc #'emit statements)
      (emit-store place value want-value))))

(defun integer-scalar-p (type)
  (and (scalar-type-p type) (integer-type-p type)))

(defun check-offset (node offset)
  "OFFSET, the expression of NODE, which must be an integer scalar."
  (unless (integer-scalar-p (ir-expr-type offset))
    (type-fail node "a pointer offset must be an integer scalar, got ~a"
               (type-description (ir-expr-type offset))))
  offset)

(defun lower-modify (node want-value)
  (let* ((place (stabilise-place (lower-place (modify-target node))))
         (type (ir-expr-type place))
         (operator (modify-operator node))
         (name (if (eq operator '+) 'incf 'decf))
         (ir-operator (if (eq operator '+) :add :sub)))
    (cond
      ((pointer-type-p type)
       (let ((offset (check-offset (modify-delta node) (lower-value (modify-delta node) nil))))
         (emit-store place (make-op ir-operator (list place offset) type) want-value)))
      (t
       (unless (numeric-scalar-p type)
         (type-fail node "~(~a~) needs a numeric or pointer place, got ~a"
                    name (type-description type)))
       (let ((delta (lower-value (modify-delta node) type)))
         (unless (eq (ir-expr-type delta) type)
           (type-fail (modify-delta node) "expected ~a, got ~a"
                      (type-description type) (type-description (ir-expr-type delta))))
         (emit-store place (make-op ir-operator (list place delta) type) want-value))))))

;;; if

(defun lower-if (node want-value hint)
  (let ((test (lower-test (if-node-test node)))
        (then (if-node-then node))
        (else (if-node-else node)))
    (cond
      ((not want-value)
       (emit (make-ir-if :test test
                         :then (capturing-statements (lower-stmt then))
                         :else (when else (capturing-statements (lower-stmt else))))))
      ((null else)
       (type-fail node "an if without an else has no value"))
      (t
       (multiple-value-bind (exprs statements type) (lower-agreeing (list then else) hint)
         (destructuring-bind (then-expr else-expr) exprs
           (destructuring-bind (then-statements else-statements) statements
             (if (and (null then-statements) (null else-statements))
                 (make-ir-ternary :test test :then then-expr :else else-expr :type type)
                 (let ((result (new-variable 'result type :temp t)))
                   (declare-variable result)
                   (emit (make-ir-if
                          :test test
                          :then (append then-statements
                                        (list (make-ir-assign :target (var-expr result)
                                                              :value then-expr)))
                          :else (append else-statements
                                        (list (make-ir-assign :target (var-expr result)
                                                              :value else-expr)))))
                   (var-expr result))))))))))

;;; Loops, blocks and returns

(defun comparison-operand-type (node binding)
  "The type of the variable that NODE compares BINDING against, if it is a plain variable."
  (when (and (typep node 'call)
             (member (call-operator node) '(< <= > >= = /=))
             (= (length (call-args node)) 2))
    (destructuring-bind (a b) (call-args node)
      (flet ((variable-type (ref other)
               (when (and (typep ref 'var-ref) (eq (var-ref-binding ref) binding)
                          (typep other 'var-ref))
                 (let ((variable (gethash (var-ref-binding other) *variables*)))
                   (and variable (ir-variable-type variable))))))
        (or (variable-type a b) (variable-type b a))))))

(defun loop-variable-type (node declared)
  "A declared type, else the type of the variable a constant init is compared with."
  (or declared
      (and (loop-node-test node)
           (constant-node-p (loop-node-init node))
           (let ((type (comparison-operand-type (loop-node-test node) (loop-node-binding node))))
             (and (numeric-scalar-p type) type)))))

(defun lower-loop (node context)
  "The IR statement for the loop NODE. Its variable's declaration is emitted ahead of it."
  (let* ((binding (loop-node-binding node))
         (variable nil)
         (init nil))
    (when binding
      (let* ((declared (cdr (assoc (binding-name binding)
                                   (declared-types (loop-node-declarations node)))))
             (hint (loop-variable-type node declared)))
        (setf init (lower-value (loop-node-init node) hint)
              variable (new-variable (binding-name binding) (ir-expr-type init)))
        (when (and hint (not (eq hint (ir-expr-type init))))
          (type-fail (loop-node-init node) "expected ~a, got ~a"
                     (type-description hint) (type-description (ir-expr-type init))))
        (setf (gethash binding *variables*) variable)))
    (let ((*loop-stack* (cons context *loop-stack*)))
      (multiple-value-bind (test test-statements)
          (if (loop-node-test node)
              (capturing (lower-test (loop-node-test node)))
              (values (true-literal) '()))
        (let ((body (capturing-statements (mapc #'lower-stmt (loop-node-body node))))
              (step-statements (capturing-statements
                                 (when (loop-node-step node)
                                   (lower-stmt (loop-node-step node))))))
          (cond
            ((and (null test-statements)
                  (or (null step-statements)
                      (and (null (rest step-statements))
                           (ir-assign-p (first step-statements)))))
             (if variable
                 (make-ir-for :variable variable :init init :test test
                              :step (first step-statements) :body body)
                 (make-ir-while :test test :body body)))
            (t
             (when variable (declare-variable variable init))
             (make-ir-while
              :test (true-literal)
              :body (append test-statements
                            (list (make-ir-if :test (make-ir-op :operator :not
                                                                :operands (list test)
                                                                :type (scalar 'bool))
                                              :then (list (make-ir-break))))
                            body
                            step-statements)))))))))

(defun emit-exit-statement (context statement)
  "Emit STATEMENT, which leaves CONTEXT's block, after the flag and result declarations."
  (when (context-flag context)
    (declare-variable (context-flag context) (literal-expr nil (scalar 'bool))))
  (when (context-result context)
    (declare-variable (context-result context)))
  (emit statement)
  (dolist (target (context-pending context))
    (emit (make-ir-if :test (var-expr (context-flag target))
                      :then (list (make-ir-break))))
    (let ((parent (first *loop-stack*)))
      (unless (eq parent target)
        (pushnew target (context-pending parent))))))

(defun assign-result (context node expr)
  (let ((type (ir-expr-type expr)))
    (cond ((null (context-result context))
           (setf (context-result context) (new-variable 'result type :temp t)))
          ((not (eq type (ir-variable-type (context-result context))))
           (type-fail node "block value has type ~a, expected ~a"
                      (type-description type)
                      (type-description (ir-variable-type (context-result context))))))
    (make-ir-assign :target (var-expr (context-result context)) :value expr)))

(defun context-result-type (context)
  (and (context-result context) (ir-variable-type (context-result context))))

(defun lower-once (node context hint)
  "Lower the block NODE as a loop that runs once, so `break` can leave it."
  (let* ((forms (block-node-body node))
         (want-value (context-want-value context))
         (statements
           (capturing-statements
             (let ((*loop-stack* (cons context *loop-stack*)))
               (mapc #'lower-stmt (if want-value (butlast forms) forms))
               (when want-value
                 (unless forms
                   (type-fail node "an empty block has no value"))
                 (let ((value (lower-value (car (last forms))
                                           (or (context-result-type context) hint))))
                   (emit (assign-result context node value))))
               (emit (make-ir-break))))))
    (emit-exit-statement context (make-ir-while :test (true-literal) :body statements))
    (when want-value
      (var-expr (context-result context)))))

(defun lower-block (node want-value hint)
  (let* ((body (block-node-body node))
         (root-p (eq node *root-block*))
         (loop-p (and (not root-p) body (null (rest body)) (loop-node-p (first body))))
         (context (make-exit-context :kind (cond (root-p :root) (loop-p :loop) (t :once))
                                     :want-value (and want-value (not loop-p))
                                     :hint hint)))
    (when (and loop-p want-value)
      (type-fail node "a loop has no value"))
    (setf (gethash node *block-contexts*) context)
    (ecase (context-kind context)
      (:root (let ((*loop-stack* (list context)))
               (mapc #'lower-stmt body))
             nil)
      (:loop (emit-exit-statement context (lower-loop (first body) context))
             nil)
      (:once (lower-once node context hint)))))

(defun lower-return (node)
  (let ((target (gethash (return-node-block node) *block-contexts*))
        (value (return-node-value node)))
    (ecase (context-kind target)
      (:root
       (when value (lower-stmt value))
       (emit (make-ir-return)))
      ((:loop :once)
       (cond ((context-want-value target)
              (unless value
                (type-fail node "return without a value from a block whose value is used"))
              (emit (assign-result target node
                                   (lower-value value (or (context-result-type target)
                                                          (context-hint target))))))
             (value (lower-stmt value)))
       (let ((inner (loop for context in *loop-stack*
                          until (eq context target)
                          collect context)))
         (cond ((null inner)
                (emit (make-ir-break)))
               (t
                (unless (context-flag target)
                  (setf (context-flag target) (new-variable 'exit (scalar 'bool))))
                (emit (make-ir-assign :target (var-expr (context-flag target))
                                      :value (true-literal)))
                (emit (make-ir-break))
                (pushnew target (context-pending (first inner))))))))))

;;; Calls

(defvar *operator-lowerers* (make-hash-table :test 'eq))

(defmacro define-operator (names (node args hint) &body body)
  `(let ((lowerer (lambda (,node ,args ,hint)
                    (declare (ignorable ,node ,args ,hint))
                    ,@body)))
     (dolist (name ',(if (listp names) names (list names)))
       (setf (gethash name *operator-lowerers*) lowerer))))

(defun lower-call (node hint statement-p)
  (let* ((operator (call-operator node))
         (lowerer (gethash operator *operator-lowerers*))
         (intrinsic (find-intrinsic operator)))
    (cond (lowerer (funcall lowerer node (call-args node) hint))
          (intrinsic (lower-intrinsic node intrinsic (call-args node) hint statement-p))
          (t (type-fail node "unknown function ~(~a~)" operator)))))

(defun check-operands (node type predicate what)
  (unless (funcall predicate type)
    (type-fail node "~(~a~) needs ~a operands, got ~a"
               (call-operator node) what (type-description type))))

(defun make-op (operator operands type)
  (make-ir-op :operator operator :operands operands :type type))

(defun lower-arith (node operator args hint predicate what)
  "A left fold of the binary OPERATOR over operands of one type."
  (multiple-value-bind (exprs statements type) (lower-agreeing args hint :broadcast t)
    (check-operands node type predicate what)
    (reduce (lambda (a b) (make-op operator (list a b) type))
            (assemble-operands exprs statements))))

(define-operator + (node args hint)
  (check-arity node args 1 nil)
  (lower-arith node :add args hint #'numeric-type-p "numeric"))

(define-operator * (node args hint)
  (check-arity node args 1 nil)
  (lower-arith node :mul args hint #'numeric-type-p "numeric"))

(define-operator - (node args hint)
  (check-arity node args 1 nil)
  (if (rest args)
      (lower-arith node :sub args hint #'numeric-type-p "numeric")
      (let* ((operand (lower-value (first args) hint))
             (type (ir-expr-type operand)))
        (unless (or (float-type-p type) (signed-type-p type))
          (type-fail node "- needs a signed or float operand, got ~a"
                     (type-description type)))
        (make-op :neg (list operand) (ir-expr-type operand)))))

(define-operator / (node args hint)
  (check-arity node args 2 nil)
  (lower-arith node :div args hint #'numeric-type-p "numeric"))

(defun lower-remainder (node args hint floored)
  "REM, or MOD when FLOORED. Floats and floored signed integers go through an intrinsic."
  (check-arity node args 2 2)
  (multiple-value-bind (exprs statements type) (lower-agreeing args hint :broadcast t)
    (check-operands node type #'numeric-type-p "numeric")
    (let ((operands (assemble-operands exprs statements))
          (intrinsic (cond ((and floored (not (unsigned-type-p type))) '%floor-mod)
                           ((float-type-p type) '%fmod))))
      (if intrinsic
          (make-ir-intrinsic :intrinsic (find-intrinsic intrinsic) :args operands :type type)
          (make-op :rem operands type)))))

(define-operator rem (node args hint) (lower-remainder node args hint nil))
(define-operator mod (node args hint) (lower-remainder node args hint t))

(define-operator logand (node args hint)
  (check-arity node args 1 nil)
  (lower-arith node :bit-and args hint #'integer-type-p "integer"))

(define-operator logior (node args hint)
  (check-arity node args 1 nil)
  (lower-arith node :bit-or args hint #'integer-type-p "integer"))

(define-operator logxor (node args hint)
  (check-arity node args 1 nil)
  (lower-arith node :bit-xor args hint #'integer-type-p "integer"))

(define-operator shl (node args hint)
  (check-arity node args 2 2)
  (lower-arith node :shl args hint #'integer-type-p "integer"))

(define-operator shr (node args hint)
  (check-arity node args 2 2)
  (lower-arith node :shr args hint #'integer-type-p "integer"))

(define-operator lognot (node args hint)
  (check-arity node args 1 1)
  (let ((operand (lower-value (first args) hint)))
    (check-operands node (ir-expr-type operand) #'integer-type-p "integer")
    (make-op :bit-not (list operand) (ir-expr-type operand))))

(defun lower-compare (node operator args)
  (check-arity node args 2 nil)
  (multiple-value-bind (exprs statements type) (lower-agreeing args nil)
    (check-operands node type (lambda (type) (or (scalar-type-p type) (pointer-type-p type)))
                    "scalar or pointer")
    (when (and (bool-type-p type) (not (member operator '(:eq :ne))))
      (type-fail node "~(~a~) cannot compare bool values" (call-operator node)))
    (let* ((operands (assemble-operands exprs statements))
           (last (1- (length operands)))
           (operands (loop for operand in operands
                           for index from 0
                           collect (if (< 0 index last) (stabilise operand) operand)))
           (bool (scalar 'bool)))
      (reduce (lambda (a b) (make-op :and (list a b) bool))
              (loop for (a b) on operands
                    while b
                    collect (make-op operator (list a b) bool))))))

(define-operator = (node args hint) (lower-compare node :eq args))
(define-operator /= (node args hint) (lower-compare node :ne args))
(define-operator < (node args hint) (lower-compare node :lt args))
(define-operator <= (node args hint) (lower-compare node :le args))
(define-operator > (node args hint) (lower-compare node :gt args))
(define-operator >= (node args hint) (lower-compare node :ge args))

(define-operator not (node args hint)
  (check-arity node args 1 1)
  (make-op :not (list (lower-test (first args))) (scalar 'bool)))

(defun lower-short-circuit (operator args)
  "AND or OR. An operand that needs statements must run only when the operator reaches it."
  (let ((bool (scalar 'bool)))
    (if (null args)
        (literal-expr (eq operator :and) bool)
        (let ((accumulator (lower-test (first args))))
          (dolist (arg (rest args) accumulator)
            (multiple-value-bind (expr statements) (capturing (lower-test arg))
              (cond
                ((null statements)
                 (setf accumulator (make-op operator (list accumulator expr) bool)))
                (t
                 (let* ((result (new-variable 'result bool :temp t))
                        (reached (if (eq operator :and)
                                     (var-expr result)
                                     (make-op :not (list (var-expr result)) bool))))
                   (declare-variable result accumulator)
                   (emit (make-ir-if
                          :test reached
                          :then (append statements
                                        (list (make-ir-assign :target (var-expr result)
                                                              :value expr)))))
                   (setf accumulator (var-expr result)))))))))))

(define-operator and (node args hint) (lower-short-circuit :and args))
(define-operator or (node args hint) (lower-short-circuit :or args))

;;; Pointer arithmetic

(defun pointer-argument (node expr)
  (unless (pointer-type-p (ir-expr-type expr))
    (type-fail node "expected a pointer, got ~a" (type-description (ir-expr-type expr))))
  expr)

(define-operator ptr+ (node args hint)
  (check-arity node args 2 2)
  (destructuring-bind (base offset) (lower-sequence args)
    (pointer-argument (first args) base)
    (check-offset (second args) offset)
    (make-op :add (list base offset) (ir-expr-type base))))

(define-operator ptr- (node args hint)
  (check-arity node args 2 2)
  (destructuring-bind (a b) (lower-sequence args)
    (pointer-argument (first args) a)
    (pointer-argument (second args) b)
    (unless (eq (ir-expr-type a) (ir-expr-type b))
      (type-fail node "ptr- needs pointers of one type, got ~a and ~a"
                 (type-description (ir-expr-type a)) (type-description (ir-expr-type b))))
    (make-op :sub (list a b) (scalar 'i64))))

;;; aref, vec, convert, bitcast

(defun lower-aref (node args)
  (check-arity node args 2 2)
  (destructuring-bind (base index)
      (lower-sequence args)
    (let ((type (ir-expr-type base)))
      (unless (integer-type-p (ir-expr-type index))
        (type-fail (second args) "an index must be an integer, got ~a"
                   (type-description (ir-expr-type index))))
      (etypecase type
        (pointer-type
         (when (atomic-type-p (pointer-type-pointee type))
           (type-fail node "use the atomic intrinsics on a pointer to an atomic"))
         (make-ir-index :base base :index index :type (pointer-type-pointee type)))
        (vector-type
         (let ((literal (second args)))
           (unless (and (literal-p literal)
                        (typep (literal-value literal) `(integer 0 ,(1- (vector-type-length type)))))
             (type-fail literal "a vector index must be a literal from 0 to ~d"
                        (1- (vector-type-length type)))))
         (make-ir-component :base base :index index :type (vector-type-element type)))
        (scalar-type
         (type-fail node "aref needs a pointer or vector, got ~a" (type-description type)))
        (atomic-type
         (type-fail node "aref needs a pointer or vector, got ~a" (type-description type)))))))

(define-operator aref (node args hint) (lower-aref node args))

(defun node-type-spec (node)
  "The type specifier written by NODE, which the walker read as a literal or a call."
  (typecase node
    (literal (literal-value node))
    (call (cons (call-operator node) (mapcar #'node-type-spec (call-args node))))
    (t (type-fail node "expected a type"))))

(defun type-argument (node)
  (let ((*current-source* (or (node-source node) *current-source*)))
    (parse-type (node-type-spec node))))

(define-operator vec (node args hint)
  (check-arity node args 3 5)
  (let ((element (type-argument (first args))))
    (unless (numeric-scalar-p element)
      (type-fail (first args) "vector elements must be numeric scalars"))
    (let ((exprs (lower-sequence (rest args) (lambda (index previous)
                                               (declare (ignore index previous))
                                               element))))
      (loop for expr in exprs
            for arg in (rest args)
            unless (eq (ir-expr-type expr) element)
              do (type-fail arg "expected ~a, got ~a"
                            (type-description element) (type-description (ir-expr-type expr))))
      (make-ir-construct
       :args exprs
       :type (parse-type `(vec ,(type-spec element) ,(length exprs)))))))

(defun convertible-p (from to)
  (or (and (scalar-type-p from) (scalar-type-p to))
      (and (vector-type-p from) (vector-type-p to)
           (= (vector-type-length from) (vector-type-length to)))))

(define-operator convert (node args hint)
  (check-arity node args 2 2)
  (let* ((target (type-argument (first args)))
         (source-node (second args))
         (operand (lower-value source-node (and (constant-node-p source-node) target)))
         (source (ir-expr-type operand)))
    (cond ((eq source target) operand)
          ((convertible-p source target)
           (make-ir-convert :kind :convert :operand operand :type target))
          (t (type-fail node "cannot convert ~a to ~a"
                        (type-description source) (type-description target))))))

(define-operator bitcast (node args hint)
  (check-arity node args 2 2)
  (let* ((target (type-argument (first args)))
         (operand (lower-value (second args) nil))
         (source (ir-expr-type operand)))
    (unless (and (type-bytes source) (type-bytes target)
                 (= (type-bytes source) (type-bytes target)))
      (type-fail node "cannot reinterpret ~a as ~a"
                 (type-description source) (type-description target)))
    (make-ir-convert :kind :bitcast :operand operand :type target)))

;;; swizzle

(defun swizzle-indices (node spec length)
  "The component indices that the keyword SPEC names in a vector of LENGTH."
  (let* ((letters (string-downcase (symbol-name spec)))
         (set (find-if (lambda (set) (every (lambda (char) (find char set)) letters))
                       '("xyzw" "rgba"))))
    (unless (and set (<= 1 (length letters) 4))
      (type-fail node "a swizzle names 1 to 4 components from xyzw or rgba, got ~(~s~)" spec))
    (map 'list (lambda (char)
                 (let ((index (position char set)))
                   (unless (< index length)
                     (type-fail node "component ~a does not exist in a vector of ~d" char length))
                   index))
         letters)))

(define-operator swizzle (node args hint)
  (check-arity node args 2 2)
  (let* ((base (lower-value (first args) nil))
         (type (ir-expr-type base)))
    (unless (vector-type-p type)
      (type-fail (first args) "swizzle needs a vector, got ~a" (type-description type)))
    (let* ((spec (literal-argument (second args) #'keywordp "a component keyword such as :xy"))
           (indices (swizzle-indices (second args) spec (vector-type-length type)))
           (element (vector-type-element type)))
      (make-ir-swizzle :base base :components indices
                       :type (if (rest indices)
                                 (parse-type `(vec ,(type-spec element) ,(length indices)))
                                 element)))))

;;; Fixed-size arrays

(defun lower-array (node args space)
  "A new array in SPACE, as the pointer to its first element."
  (check-arity node args 2 3)
  (let ((element (type-argument (first args)))
        (size (literal-argument (second args) (lambda (value) (typep value '(integer 1)))
                                "a positive integer size"))
        (zero (and (third args)
                   (literal-argument (third args) (lambda (value) (eq value :zero))
                                     ":zero"))))
    (unless (or (numeric-scalar-p element) (bool-type-p element) (vector-type-p element)
                (and (atomic-type-p element) (eq space :shared)))
      (type-fail (first args) "an array element must be a scalar or vector~:[~; (atomic only in shared)~], got ~a"
                 (eq space :shared) (type-description element)))
    (let ((variable (new-variable 'array (parse-type `(ptr ,(type-spec element) ,space)) :temp t)))
      (emit (make-ir-array :variable variable :size size :zero (and zero t)))
      (var-expr variable))))

(define-operator shared-array (node args hint) (lower-array node args :shared))
(define-operator local-array (node args hint) (lower-array node args :private))

;;; Intrinsics

(defun literal-argument (node predicate what)
  (unless (and (literal-p node) (funcall predicate (literal-value node)))
    (type-fail node "expected ~a" what))
  (literal-value node))

(defun atomic-element (node expr)
  "The element type of the atomic pointer EXPR, which NODE produced."
  (let ((type (ir-expr-type expr)))
    (unless (and (pointer-type-p type)
                 (atomic-type-p (pointer-type-pointee type))
                 (member (pointer-type-space type) '(:global :shared)))
      (type-fail node "expected a pointer to an atomic in :global or :shared memory, got ~a"
                 (type-description type)))
    (atomic-type-element (pointer-type-pointee type))))

(defun lower-atomic (node intrinsic args kind)
  (check-arity node args (ecase kind (:load 1) ((:rmw :store) 2) (:cas 3))
               (ecase kind (:load 1) ((:rmw :store) 2) (:cas 3)))
  (let* ((exprs (lower-sequence args
                                (lambda (index previous)
                                  (when (plusp index)
                                    (atomic-element (first args) (first previous))))))
         (element (atomic-element (first args) (first exprs))))
    (loop for expr in (rest exprs)
          for arg in (rest args)
          unless (eq (ir-expr-type expr) element)
            do (type-fail arg "expected ~a, got ~a"
                          (type-description element) (type-description (ir-expr-type expr))))
    (make-ir-intrinsic :intrinsic intrinsic :args exprs
                       :type (unless (eq kind :store) element))))

(defun lower-intrinsic (node intrinsic args hint statement-p)
  (let* ((signature (intrinsic-signature intrinsic))
         (expr
           (ecase (first signature)
             (:axis
              (check-arity node args 1 1)
              (let ((axis (literal-argument (first args)
                                            (lambda (value) (typep value '(integer 0 2)))
                                            "an axis literal 0, 1 or 2")))
                (make-ir-intrinsic :intrinsic intrinsic :options (list :axis axis)
                                   :type (scalar 'u32))))
             (:barrier
              (check-arity node args 0 1)
              (let ((scope (if args
                               (literal-argument (first args)
                                                 (lambda (value) (member value '(:shared :global :all)))
                                                 "a scope: :shared, :global or :all")
                               :shared)))
                (make-ir-intrinsic :intrinsic intrinsic :options (list :scope scope))))
             ((:float :number)
              (check-arity node args (second signature) (second signature))
              (multiple-value-bind (exprs statements type) (lower-agreeing args hint)
                (if (eq (first signature) :float)
                    (check-operands node type #'float-type-p "float")
                    (check-operands node type #'numeric-type-p "numeric"))
                (make-ir-intrinsic :intrinsic intrinsic
                                   :args (assemble-operands exprs statements)
                                   :type type)))
             (:atomic (lower-atomic node intrinsic args (second signature))))))
    (if (and (intrinsic-impure-p intrinsic) (not statement-p) (ir-expr-type expr))
        (spill expr 'old)
        expr)))

;;; Kernels

(defun parse-parameters (arguments)
  (loop for argument in arguments
        collect (destructuring-bind (name spec) argument
                  (let ((type (parse-type spec)))
                    (when (atomic-type-p type)
                      (type-failure spec "a parameter cannot have an atomic type"))
                    (cons name type)))))

(defun wrap-in-nil-block (form)
  "A CST for (block nil FORM), so that `return` leaves the kernel."
  (let* ((cst (if (typep form 'cst:cst) form (cst:cst-from-expression form)))
         (source (cst:source cst)))
    (cst:cons (cst:cst-from-expression 'block)
              (cst:cons (cst:cst-from-expression nil)
                        (cst:cons cst (cst:cst-from-expression nil))
                        :source source)
              :source source)))

(defun kernel-scalar-types (parameters body)
  (let ((types '()))
    (flet ((note (type)
             (when type
               (dolist (scalar (type-scalars type))
                 (pushnew scalar types)))))
      (dolist (parameter parameters)
        (note (ir-variable-type parameter)))
      (dolist (statement body)
        (map-ir (lambda (node)
                  (typecase node
                    (ir-expr (note (ir-expr-type node)))
                    (ir-declare (note (ir-variable-type (ir-declare-variable node))))
                    (ir-array (note (ir-variable-type (ir-array-variable node))))))
                statement)))
    (sort types #'string< :key (lambda (type) (symbol-name (scalar-type-name type))))))

(defun lower-kernel (form &key (name 'kernel) arguments)
  "Lower FORM, a CST from READ-KERNEL or a plain form, to an IR-KERNEL.
ARGUMENTS is a list of (name type) kernel parameters. `return` leaves the kernel."
  (let* ((*current-source* nil)
         (*emit* '())
         (*variables* (make-hash-table :test 'eq))
         (*variable-id* 0)
         (*block-contexts* (make-hash-table :test 'eq))
         (*loop-stack* '())
         (parameters (parse-parameters arguments)))
    (multiple-value-bind (root bindings)
        (walk-kernel (wrap-in-nil-block form) :variables (mapcar #'car parameters))
      (let* ((*root-block* root)
             (variables (loop for binding in bindings
                              for (nil . type) in parameters
                              collect (let ((variable (new-variable (binding-name binding) type)))
                                        (setf (gethash binding *variables*) variable)
                                        variable)))
             (body (capturing-statements (lower-block root nil nil))))
        (make-ir-kernel :name name :parameters variables :body body
                        :scalar-types (kernel-scalar-types variables body))))))
