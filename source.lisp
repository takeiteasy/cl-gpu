(in-package #:cl-gpu)

;;; `define-kernel` receives plain conses from the host reader. To locate its errors, the
;;; form is found again by re-reading the file being compiled or loaded.

(defun editor-snippet-p ()
  "True when the form being compiled is an editor snippet rather than file text."
  ;; TODO: map snippet positions back to the editor buffer (#27)
  #+sbcl (and (getf sb-c::*source-plist* :emacs-buffer) t)
  #-sbcl nil)

(defun same-form-p (a b)
  "True when A and B are the same form, comparing numbers, characters and strings exactly."
  (typecase a
    (cons (and (consp b)
               (same-form-p (car a) (car b))
               (same-form-p (cdr a) (cdr b))))
    (string (and (stringp b) (string= a b)))
    (vector (and (vectorp b)
                 (not (stringp b))
                 (= (length a) (length b))
                 (every #'same-form-p a b)))
    (t (eql a b))))

(defun find-form-cst-in (cst form)
  "The innermost sub-CST of CST that reads as FORM, or NIL."
  (when (cst:consp cst)
    (let ((head (cst:first cst)))
      (or (loop for rest = cst then (cst:rest rest)
                while (cst:consp rest)
                do (let ((match (find-form-cst-in (cst:first rest) form)))
                     (when match (return match))))
          (and (cst:atom head)
               (eql (cst:raw head) (car form))
               (same-form-p (cst:raw cst) form)
               cst)))))

(defun find-form-in-file (form path)
  (let* ((text (uiop:read-file-string path))
         (eclector.reader:*client* (make-kernel-client text (namestring path)))
         (*read-eval* nil)
         (*package* *package*)
         (stream (make-string-input-stream text)))
    (handler-bind ((error (lambda (condition)
                            (declare (ignore condition))
                            (let ((restart (find-restart 'eclector.base:recover)))
                              (when restart (invoke-restart restart))))))
      (loop for cst = (eclector.concrete-syntax-tree:read stream nil stream)
            until (eq cst stream)
            do (let ((match (find-form-cst-in cst form)))
                 (when match (return match)))
               (let ((raw (cst:raw cst)))
                 (when (and (consp raw) (eq (car raw) 'in-package) (consp (cdr raw)))
                   (let ((package (ignore-errors (find-package (cadr raw)))))
                     (if package
                         (setf *package* package)
                         (return)))))))))

(defun find-form-cst (form)
  "The CST of FORM, read from the file being loaded or compiled, or NIL if it cannot be found."
  (unless (editor-snippet-p)
    (dolist (path (remove-duplicates (remove nil (list *load-truename* *compile-file-truename*))
                                     :test #'equal))
      ;; Locating is best effort: a failure must not break compilation.
      (let ((cst (ignore-errors (find-form-in-file form path))))
        (when cst (return cst))))))
