(in-package #:cl-gpu)

;;; Variables are objects, so shadowed names are distinct. ID numbers them within a kernel.
;;; TEMP marks compiler-made variables that are assigned once and then only read.
(defstruct (ir-variable (:constructor make-ir-variable (name type id temp)))
  name type id temp)

(defmethod print-object ((variable ir-variable) stream)
  (print-unreadable-object (variable stream :type t)
    (format stream "~(~a~)_~d" (ir-variable-name variable) (ir-variable-id variable))))

;;; Expressions. TYPE is NIL for a call with no result.
(defstruct (ir-expr (:constructor nil)) type)

(defstruct (ir-literal (:include ir-expr)) value)
(defstruct (ir-var (:include ir-expr)) variable)

;;; OPERATOR is one of :add :sub :mul :div :rem :neg, :eq :ne :lt :le :gt :ge,
;;; :and :or :not, :bit-and :bit-or :bit-xor :bit-not :shl :shr.
(defstruct (ir-op (:include ir-expr)) operator operands)

(defstruct (ir-ternary (:include ir-expr)) test then else)
(defstruct (ir-index (:include ir-expr)) base index)
(defstruct (ir-component (:include ir-expr)) base index)

;;; COMPONENTS are the indices read from BASE, a vector. A read only; never a place.
(defstruct (ir-swizzle (:include ir-expr)) base components)

;;; A construct with one argument broadcasts a scalar to every component.
(defstruct (ir-construct (:include ir-expr)) args)

;;; KIND is :convert (value cast) or :bitcast (reinterpret bits).
(defstruct (ir-convert (:include ir-expr)) kind operand)

;;; OPTIONS is a plist: :axis (0-2) for thread positions, :scope for barriers.
(defstruct (ir-intrinsic (:include ir-expr)) intrinsic args options)

;;; Statements. Bodies are lists of statements.
(defstruct (ir-stmt (:constructor nil)))

(defstruct (ir-declare (:include ir-stmt)) variable init)
(defstruct (ir-assign (:include ir-stmt)) target value)
(defstruct (ir-if (:include ir-stmt)) test then else)

;;; STEP is an assignment or NIL.
(defstruct (ir-for (:include ir-stmt)) variable init test step body)

(defstruct (ir-while (:include ir-stmt)) test body)
(defstruct (ir-break (:include ir-stmt)))
(defstruct (ir-return (:include ir-stmt)))
(defstruct (ir-expr-stmt (:include ir-stmt)) expr)

;;; PARAMETERS are IR-VARIABLEs; SCALAR-TYPES lists the scalar types the kernel uses.
(defstruct ir-kernel name parameters body scalar-types)

(defun ir-children (node)
  (flet ((present (&rest nodes) (remove nil nodes)))
    (etypecase node
      ((or ir-literal ir-var ir-break ir-return) nil)
      (ir-op (ir-op-operands node))
      (ir-ternary (list (ir-ternary-test node) (ir-ternary-then node) (ir-ternary-else node)))
      (ir-index (list (ir-index-base node) (ir-index-index node)))
      (ir-component (list (ir-component-base node) (ir-component-index node)))
      (ir-swizzle (list (ir-swizzle-base node)))
      (ir-construct (ir-construct-args node))
      (ir-convert (list (ir-convert-operand node)))
      (ir-intrinsic (ir-intrinsic-args node))
      (ir-declare (present (ir-declare-init node)))
      (ir-assign (list (ir-assign-target node) (ir-assign-value node)))
      (ir-if (append (list (ir-if-test node)) (ir-if-then node) (ir-if-else node)))
      (ir-for (append (present (ir-for-init node) (ir-for-test node) (ir-for-step node))
                      (ir-for-body node)))
      (ir-while (cons (ir-while-test node) (ir-while-body node)))
      (ir-expr-stmt (list (ir-expr-stmt-expr node)))
      (ir-kernel (ir-kernel-body node)))))

(defun map-ir (function node)
  "Call FUNCTION on NODE and its descendants, parents first."
  (funcall function node)
  (dolist (child (ir-children node))
    (map-ir function child)))
