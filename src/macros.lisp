(in-package #:cl-gpu)

(define-kernel-macro when (test &body body)
  `(if ,test (progn ,@body)))

(define-kernel-macro unless (test &body body)
  `(if (not ,test) (progn ,@body)))

(define-kernel-macro cond (&rest clauses)
  (when clauses
    (destructuring-bind ((test &rest body) &rest more) clauses
      (cond ((null body) (if more `(or ,test (cond ,@more)) test))
            ((eq test t) `(progn ,@body))
            (more `(if ,test (progn ,@body) (cond ,@more)))
            (t `(if ,test (progn ,@body)))))))

(define-kernel-macro case (key &rest clauses)
  (let ((variable (gensym "KEY")))
    `(let ((,variable ,key))
       (cond ,@(loop for (keys . body) in clauses
                     collect (cons (if (member keys '(t otherwise))
                                       t
                                       (let ((tests (loop for value in (if (listp keys) keys (list keys))
                                                          collect `(= ,variable ,value))))
                                         (if (rest tests) `(or ,@tests) (first tests))))
                                   body))))))

(define-kernel-macro return (&optional (value nil value-p))
  `(return-from nil ,@(when value-p (list value))))

(defun expand-modify (operator place delta)
  "Expand `(setf PLACE (OPERATOR PLACE DELTA))`, evaluating the place's subforms once."
  (if (consp place)
      (let* ((temporaries (loop for nil in (rest place) collect (gensym "ARG")))
             (target `(,(first place) ,@temporaries))
             (step (if (atom delta) delta (gensym "DELTA"))))
        `(let (,@(mapcar #'list temporaries (rest place))
               ,@(unless (eq step delta) `((,step ,delta))))
           (setf ,target (,operator ,target ,step))))
      `(setf ,place (,operator ,place ,delta))))

(define-kernel-macro incf (place &optional (delta 1))
  (expand-modify '+ place delta))

(define-kernel-macro decf (place &optional (delta 1))
  (expand-modify '- place delta))

(define-kernel-macro dotimes ((variable count &rest result) &body body)
  (when result
    (error "result forms are not supported"))
  (let* ((literal (integerp count))
         (limit (if literal count (gensym "LIMIT")))
         (loop-form `(for (,variable 0 (< ,variable ,limit) (incf ,variable))
                       ,@body)))
    (if literal
        loop-form
        `(let ((,limit ,count)) ,loop-form))))
