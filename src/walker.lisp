(in-package #:cl-gpu)

(defvar *expansion-depth* 0)

(defvar *walkers* (make-hash-table :test 'eq))

(defmacro define-walker (name (cst env args) &body body)
  "Define the walker for the special form NAME. ARGS is the list of argument CSTs."
  `(setf (gethash ',name *walkers*)
         (lambda (,cst ,env ,args)
           (declare (ignorable ,cst ,env ,args))
           ,@body)))

;;; CL macros that stay ordinary calls; the IR maps them onto && and ||.
(defparameter *call-macros* '(and or))

(defun walk-error (cst control &rest args)
  (error 'kernel-syntax-error
         :message (apply #'format nil control args)
         :form (cst:raw cst)
         :location (or (cst:source cst) *current-source*)))

(defun proper-list-p (object)
  (and (listp object) (null (cdr (last object)))))

(defun check-arg-count (cst args min max)
  (let ((count (length args)))
    (unless (and (>= count min) (or (null max) (<= count max)))
      (walk-error cst "wrong number of arguments to ~(~a~): expected ~a, got ~d"
                  (cst:raw (cst:first cst))
                  (cond ((eql min max) min)
                        (max (format nil "~d to ~d" min max))
                        (t (format nil "at least ~d" min)))
                  count))))

(defun variable-name-p (object)
  (and (symbolp object) (not (member object '(nil t))) (not (keywordp object))))

(defun check-variable-name (cst)
  (unless (variable-name-p (cst:raw cst))
    (walk-error cst "~s cannot be used as a variable name" (cst:raw cst))))

(defun cst-list (cst what)
  "The element CSTs of CST, which must be a proper list."
  (unless (proper-list-p (cst:raw cst))
    (walk-error cst "~a must be a proper list, got ~s" what (cst:raw cst)))
  (cst:listify cst))

(defun new-binding (name-cst)
  (check-variable-name name-cst)
  (make-binding :name (cst:raw name-cst)
                :source (or (cst:source name-cst) *current-source*)))

(defun parse-body (csts)
  "Split leading `(declare ...)` forms off CSTS. Return the raw declarations and the rest."
  (let ((declarations '()))
    (loop while (and csts
                     (consp (cst:raw (first csts)))
                     (eq (car (cst:raw (first csts))) 'declare))
          do (push (cst:raw (pop csts)) declarations))
    (values (nreverse declarations) csts)))

(defun body-node (nodes)
  (if (and nodes (null (rest nodes)))
      (first nodes)
      (make-progn-node :body nodes :source *current-source*)))

(defmacro with-expansion ((cst) &body body)
  `(let ((*expansion-depth* (1+ *expansion-depth*)))
     (when (> *expansion-depth* *max-expansion-depth*)
       (walk-error ,cst "macro expansion nested deeper than ~d"
                   *max-expansion-depth*))
     ,@body))

(defun walk-all (csts env)
  (loop for cst in csts collect (walk cst env)))

(defun operator-kind (head env)
  "Classify the operator HEAD as :local-macro, :special, :macro, :unsupported or :call."
  (cond ((env-lookup env :macro head) :local-macro)
        ((gethash head *walkers*) :special)
        ((gethash head *kernel-macros*) :macro)
        ((and (eq (symbol-package head) (find-package '#:common-lisp))
              (not (member head *call-macros*))
              (or (special-operator-p head) (macro-function head)))
         :unsupported)
        (t :call)))

(defun expand-macro-call (cst expander)
  "Expand the macro form CST with EXPANDER into a CST that keeps its source."
  (let ((expansion (handler-case (funcall expander (cst:raw cst))
                     (error (condition)
                       (walk-error cst "cannot expand ~(~a~): ~a"
                                   (car (cst:raw cst)) condition)))))
    (cst:reconstruct nil expansion cst :default-source (cst:source cst))))

(defun walk (cst env)
  (let ((*current-source* (or (cst:source cst) *current-source*))
        (raw (cst:raw cst)))
    (typecase raw
      (null (make-literal :value nil :source *current-source*))
      (symbol (walk-symbol cst env))
      (cons (walk-cons cst env))
      (t (make-literal :value raw :source *current-source*)))))

(defun walk-symbol (cst env)
  (let* ((name (cst:raw cst))
         (entry (and (variable-name-p name) (env-lookup env :variable name))))
    (cond ((not (variable-name-p name))
           (make-literal :value name :source *current-source*))
          ((binding-p entry)
           (make-var-ref :binding entry :source *current-source*))
          (entry
           (with-expansion (cst) (walk (cdr entry) env)))
          ((type-name-symbol-p name)
           (make-literal :value name :source *current-source*))
          (t (walk-error cst "unbound variable ~s" name)))))

(defun walk-cons (cst env)
  (let ((raw (cst:raw cst)))
    (unless (proper-list-p raw)
      (walk-error cst "improper form ~s" raw))
    (unless (symbolp (car raw))
      (walk-error cst "operator must be a symbol, got ~s" (car raw)))
    (let* ((head (car raw))
           (args (cst:listify (cst:rest cst))))
      (ecase (operator-kind head env)
        (:local-macro
         (with-expansion (cst)
           (walk (expand-macro-call cst (env-lookup env :macro head)) env)))
        (:macro
         (with-expansion (cst)
           (walk (expand-macro-call cst (gethash head *kernel-macros*)) env)))
        (:special (funcall (gethash head *walkers*) cst env args))
        (:unsupported
         (walk-error cst "~(~a~) is not supported in kernels" head))
        (:call
         (make-call :operator head :args (walk-all args env)
                    :source *current-source*))))))

(defun walk-place (cst env)
  (let ((*current-source* (or (cst:source cst) *current-source*))
        (raw (cst:raw cst)))
    (cond ((and (symbolp raw) (variable-name-p raw))
           (let ((entry (env-lookup env :variable raw)))
             (cond ((binding-p entry)
                    (make-var-ref :binding entry :source *current-source*))
                   (entry (with-expansion (cst) (walk-place (cdr entry) env)))
                   (t (walk-error cst "unbound variable ~s" raw)))))
          ((and (consp raw) (proper-list-p raw) (symbolp (car raw)))
           (let ((head (car raw)))
             (case (operator-kind head env)
               (:local-macro
                (with-expansion (cst)
                  (walk-place (expand-macro-call cst (env-lookup env :macro head))
                              env)))
               (:macro
                (with-expansion (cst)
                  (walk-place (expand-macro-call cst (gethash head *kernel-macros*))
                              env)))
               (:call (walk cst env))
               (t (walk-error cst "~s is not a place" raw)))))
          (t (walk-error cst "~s is not a place" raw)))))

(defun walk-assignments (cst env args symbols-only)
  (unless (evenp (length args))
    (walk-error cst "odd number of arguments to ~(~a~)" (cst:raw (cst:first cst))))
  (let ((assigns (loop for (place value) on args by #'cddr
                       do (when (and symbols-only
                                     (not (and (symbolp (cst:raw place))
                                               (variable-name-p (cst:raw place)))))
                            (walk-error place "~s is not a variable" (cst:raw place)))
                       collect (let ((target (walk-place place env)))
                                 (make-assign :target target
                                              :value (walk value env)
                                              :source *current-source*)))))
    (if (and assigns (null (rest assigns)))
        (first assigns)
        (make-progn-node :body assigns :source *current-source*))))

(define-walker setq (cst env args)
  (walk-assignments cst env args t))

(define-walker setf (cst env args)
  (walk-assignments cst env args nil))

(define-walker progn (cst env args)
  (make-progn-node :body (walk-all args env) :source *current-source*))

(define-walker if (cst env args)
  (check-arg-count cst args 2 3)
  (make-if-node :test (walk (first args) env)
                :then (walk (second args) env)
                :else (when (third args) (walk (third args) env))
                :source *current-source*))

(define-walker declare (cst env args)
  (walk-error cst "declare is not allowed here"))

(defun walk-let (cst env args sequential)
  (check-arg-count cst args 1 nil)
  (let ((inner (make-env env))
        (bindings '())
        (inits '()))
    (dolist (spec (cst-list (first args) "binding list"))
      (let* ((parts (if (consp (cst:raw spec)) (cst-list spec "binding") (list spec)))
             (name (first parts))
             (init (second parts)))
        (when (or (null parts) (cddr parts))
          (walk-error spec "malformed binding ~s" (cst:raw spec)))
        (let ((init-node (when init (walk init (if sequential inner env))))
              (binding (new-binding name)))
          (env-bind inner :variable (binding-name binding) binding)
          (push binding bindings)
          (push init-node inits))))
    (multiple-value-bind (declarations body) (parse-body (rest args))
      (make-let-node :sequential sequential
                     :bindings (nreverse bindings)
                     :inits (nreverse inits)
                     :declarations declarations
                     :body (walk-all body inner)
                     :source *current-source*))))

(define-walker let (cst env args)
  (walk-let cst env args nil))

(define-walker let* (cst env args)
  (walk-let cst env args t))

(define-walker block (cst env args)
  (check-arg-count cst args 1 nil)
  (check-block-name (first args))
  (let* ((block (make-block-node :name (cst:raw (first args))
                                 :source *current-source*))
         (inner (make-env env)))
    (env-bind inner :block (block-node-name block) block)
    (setf (block-node-body block) (walk-all (rest args) inner))
    block))

(defun check-block-name (cst)
  (unless (symbolp (cst:raw cst))
    (walk-error cst "block name must be a symbol, got ~s" (cst:raw cst))))

(define-walker return-from (cst env args)
  (check-arg-count cst args 1 2)
  (check-block-name (first args))
  (let* ((name (cst:raw (first args)))
         (block (env-lookup env :block name)))
    (unless block
      (walk-error cst "no block named ~s is visible here" name))
    (make-return-node :block block
                      :value (when (second args) (walk (second args) env))
                      :source *current-source*)))

;;; Loops establish a NIL block around themselves, like DO.
(defun walk-loop (env binding-cst init-cst test-cst step-cst body-csts)
  (let* ((block (make-block-node :name nil :source *current-source*))
         (block-env (make-env env))
         (inner (make-env block-env))
         (binding (when binding-cst (new-binding binding-cst)))
         (init (when init-cst (walk init-cst block-env))))
    (env-bind block-env :block nil block)
    (when binding
      (env-bind inner :variable (binding-name binding) binding))
    (multiple-value-bind (declarations body) (parse-body body-csts)
      (setf (block-node-body block)
            (list (make-loop-node
                   :binding binding
                   :init init
                   :test (when test-cst (walk test-cst inner))
                   :step (when step-cst (walk step-cst inner))
                   :declarations declarations
                   :body (walk-all body inner)
                   :source *current-source*)))
      block)))

(define-walker for (cst env args)
  (check-arg-count cst args 1 nil)
  (let ((spec (cst-list (first args) "for clause")))
    (unless (<= 2 (length spec) 4)
      (walk-error (first args) "for clause must be (variable init [test [step]])"))
    (destructuring-bind (variable init &optional test step) spec
      (walk-loop env variable init test step (rest args)))))

(define-walker while (cst env args)
  (check-arg-count cst args 1 nil)
  (walk-loop env nil nil (first args) nil (rest args)))

(defun compile-local-macro (cst lambda-list body)
  (handler-case
      (let ((whole (gensym "WHOLE")))
        (handler-bind ((warning #'muffle-warning))
          (compile nil `(lambda (,whole)
                          (destructuring-bind ,lambda-list (cdr ,whole)
                            ,@body)))))
    (error (condition)
      (walk-error cst "invalid macro definition: ~a" condition))))

(define-walker macrolet (cst env args)
  (check-arg-count cst args 1 nil)
  (let ((inner (make-env env)))
    (dolist (definition (cst-list (first args) "macrolet definitions"))
      (let ((parts (cst-list definition "macro definition")))
        (unless (and (>= (length parts) 2) (symbolp (cst:raw (first parts))))
          (walk-error definition "malformed macro definition ~s" (cst:raw definition)))
        (env-bind inner :macro (cst:raw (first parts))
                  (compile-local-macro definition
                                       (cst:raw (second parts))
                                       (mapcar #'cst:raw (cddr parts))))))
    (body-node (walk-all (rest args) inner))))

(define-walker symbol-macrolet (cst env args)
  (check-arg-count cst args 1 nil)
  (let ((inner (make-env env)))
    (dolist (definition (cst-list (first args) "symbol-macrolet definitions"))
      (let ((parts (cst-list definition "symbol macro definition")))
        (unless (= (length parts) 2)
          (walk-error definition "malformed symbol macro definition ~s"
                      (cst:raw definition)))
        (check-variable-name (first parts))
        (env-bind inner :variable (cst:raw (first parts))
                  (cons :symbol-macro (second parts)))))
    (body-node (walk-all (rest args) inner))))

(defun walk-kernel (form &key variables)
  "Walk FORM, a CST from READ-KERNEL or a plain form, into an AST.
VARIABLES names the variables already in scope. Return the AST and their BINDINGs."
  (let* ((*current-source* nil)
         (*expansion-depth* 0)
         (cst (if (typep form 'cst:cst) form (cst:cst-from-expression form)))
         (env (make-env))
         (bindings (loop for name in variables
                         collect (let ((binding (make-binding :name name)))
                                   (env-bind env :variable name binding)
                                   binding))))
    (values (walk cst env) bindings)))
