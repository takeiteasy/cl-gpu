(in-package #:cl-gpu)

(defvar *max-expansion-depth* 256
  "Deepest nesting of macro expansions before the walker gives up.")

(defvar *kernel-macros* (make-hash-table :test 'eq))

(defmacro define-kernel-macro (name lambda-list &body body)
  "Define a macro usable inside kernels. BODY returns the expansion as a form."
  (let ((whole (gensym "WHOLE")))
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (setf (gethash ',name *kernel-macros*)
             (lambda (,whole)
               (destructuring-bind ,lambda-list (cdr ,whole)
                 ,@body))))))

;;; Each namespace is an alist from name to entry, searched innermost first.
;;; Variable entries are a BINDING or a (:symbol-macro . expansion-cst). Operator entries
;;; are a macro expander or a LOCAL-FUNCTION; macrolet, flet and labels share the namespace.
;;; A BARRIER scope hides the blocks outside it, so a local function cannot return from them.
(defstruct (env (:constructor make-env (&optional parent)))
  parent variables operators blocks barrier)

(defun env-lookup (env namespace name)
  (loop for scope = env then (env-parent scope)
        while scope
        do (let ((entry (assoc name (ecase namespace
                                      (:variable (env-variables scope))
                                      (:operator (env-operators scope))
                                      (:block (env-blocks scope))))))
             (when entry (return (cdr entry)))
             (when (and (eq namespace :block) (env-barrier scope))
               (return nil)))))

(defun env-bind (env namespace name entry)
  (ecase namespace
    (:variable (push (cons name entry) (env-variables env)))
    (:operator (push (cons name entry) (env-operators env)))
    (:block (push (cons name entry) (env-blocks env)))))
