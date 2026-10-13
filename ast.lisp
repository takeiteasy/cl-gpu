(in-package #:cl-gpu)

(defstruct (node (:constructor nil))
  (source nil))

(defmethod print-object ((node node) stream)
  (print-unreadable-object (node stream :type t :identity t)))

;;; One binding object per introduced variable, so shadowing is resolved by identity.
(defstruct (binding (:include node))
  (name nil :type symbol))

(defmethod print-object ((binding binding) stream)
  (print-unreadable-object (binding stream :type t :identity t)
    (prin1 (binding-name binding) stream)))

(defstruct (literal (:include node)) value)
(defstruct (var-ref (:include node)) binding)

;;; INITS parallels BINDINGS; an uninitialised variable has a NIL init.
;;; DECLARATIONS holds the raw `(declare ...)` forms.
(defstruct (let-node (:include node))
  sequential bindings inits declarations body)

;;; TARGET is a VAR-REF or a CALL (an array or field place).
(defstruct (assign (:include node)) target value)

;;; OPERATOR is + or -. TARGET is read and written, its subforms evaluated once.
(defstruct (modify (:include node)) target operator delta)

(defstruct (if-node (:include node)) test then else)
(defstruct (progn-node (:include node)) body)

;;; BINDING and INIT are NIL for a `while` loop. TEST and STEP may be NIL.
(defstruct (loop-node (:include node))
  binding init test step declarations body)

(defstruct (block-node (:include node)) name body)
(defstruct (return-node (:include node)) block value)
(defstruct (call (:include node)) operator args)

(defun node-children (node)
  (flet ((present (&rest nodes) (remove nil nodes)))
    (etypecase node
      ((or binding literal var-ref) nil)
      (let-node (append (remove nil (let-node-inits node)) (let-node-body node)))
      (assign (present (assign-target node) (assign-value node)))
      (modify (list (modify-target node) (modify-delta node)))
      (if-node (present (if-node-test node) (if-node-then node) (if-node-else node)))
      (progn-node (progn-node-body node))
      (loop-node (append (present (loop-node-init node) (loop-node-test node)
                                  (loop-node-step node))
                         (loop-node-body node)))
      (block-node (block-node-body node))
      (return-node (present (return-node-value node)))
      (call (call-args node)))))

(defun map-nodes (function node)
  "Call FUNCTION on NODE and its descendants, parents first."
  (funcall function node)
  (dolist (child (node-children node))
    (map-nodes function child)))
