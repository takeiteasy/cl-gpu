(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(test walks-plain-forms-without-locations
  (let ((node (gpu:walk-kernel '(let ((a 1)) (+ a 2)))))
    (is (typep node 'gpu:let-node))
    (is (every (lambda (child) (null (gpu:node-source child)))
               (nodes-of-type t node)))))

(test walks-text-with-locations
  (let ((node (walk-text (format nil "(let ((a 1))~%  (foo a))"))))
    (is (equal '(2 3)
               (let ((location (gpu:node-source
                                (first (gpu:let-node-body node)))))
                 (list (gpu:source-location-line location)
                       (gpu:source-location-column location)))))))

(test let-init-sees-outer-binding
  (let* ((outer (walk-text "(let ((a 1)) (let ((a 2) (b a)) b))"))
         (inner (first (gpu:let-node-body outer)))
         (outer-a (first (gpu:let-node-bindings outer)))
         (inner-a (first (gpu:let-node-bindings inner)))
         (inner-b (second (gpu:let-node-bindings inner)))
         (init (second (gpu:let-node-inits inner))))
    (is (eq outer-a (gpu:var-ref-binding init)))
    (is (not (eq inner-a (gpu:var-ref-binding init))))
    (is (eq inner-b (gpu:var-ref-binding (first (gpu:let-node-body inner)))))))

(test let*-binds-sequentially
  (let* ((outer (walk-text "(let ((a 1)) (let* ((a 2) (b a)) b))"))
         (inner (first (gpu:let-node-body outer)))
         (init (second (gpu:let-node-inits inner))))
    (is (gpu:let-node-sequential inner))
    (is (eq (first (gpu:let-node-bindings inner)) (gpu:var-ref-binding init)))))

(test dotimes-variable-shadows-outer
  (let* ((node (walk-text "(let ((i 5)) (dotimes (i 3) i) i)"))
         (loop-node (first (nodes-of-type 'gpu:loop-node node)))
         (outer-i (first (gpu:let-node-bindings node)))
         (inside (first (gpu:loop-node-body loop-node)))
         (after (second (gpu:let-node-body node))))
    (is (eq (gpu:loop-node-binding loop-node) (gpu:var-ref-binding inside)))
    (is (not (eq outer-i (gpu:loop-node-binding loop-node))))
    (is (eq outer-i (gpu:var-ref-binding after)))))

(test let-shadows-symbol-macro
  (let ((plain (walk-text "(symbol-macrolet ((x (aref p 0))) x)" :variables '("p")))
        (shadowed (walk-text "(symbol-macrolet ((x (aref p 0))) (let ((x 1)) x))"
                             :variables '("p"))))
    (is (typep plain 'gpu:call))
    (is (eq 'aref (gpu:call-operator plain)))
    (is (typep (first (gpu:let-node-body shadowed)) 'gpu:var-ref))))

(test macrolet-shadows-kernel-macro
  (let ((node (walk-text "(macrolet ((when (c) `(foo ,c))) (when 1))")))
    (is (typep node 'gpu:call))
    (is (eq (ksym "foo") (gpu:call-operator node)))))

(test macrolet-shadows-nothing-outside
  (let ((node (walk-text "(progn (macrolet ((when (c) `(foo ,c))) (when 1)) (when 1 2))")))
    (is (typep (second (gpu:progn-node-body node)) 'gpu:if-node))))

(test variables-are-in-scope
  (multiple-value-bind (node bindings) (walk-text "(+ n 1)" :variables '("n"))
    (is (eq (first bindings)
            (gpu:var-ref-binding (first (gpu:call-args node)))))))

(test multi-pair-setf-assigns-in-order
  (let ((node (walk-text "(let ((a 0) (b 0)) (setf a 1 b a))")))
    (let* ((progn (first (gpu:let-node-body node)))
           (assigns (gpu:progn-node-body progn)))
      (is (= 2 (length assigns)))
      (is (every (lambda (assign) (typep assign 'gpu:assign)) assigns))
      (is (eq (first (gpu:let-node-bindings node))
              (gpu:var-ref-binding (gpu:assign-target (first assigns)))))
      (is (eq (second (gpu:let-node-bindings node))
              (gpu:var-ref-binding (gpu:assign-target (second assigns)))))
      (is (eq (first (gpu:let-node-bindings node))
              (gpu:var-ref-binding (gpu:assign-value (second assigns))))))))

(test setf-through-symbol-macro-reaches-place
  (let* ((node (walk-text "(symbol-macrolet ((x (aref p 0))) (setf x 1))" :variables '("p")))
         (target (gpu:assign-target node)))
    (is (typep target 'gpu:call))
    (is (eq 'aref (gpu:call-operator target)))))

(test setq-rejects-non-variables
  (is (equal '(1 7) (error-position-of gpu:kernel-syntax-error
                      (walk-text "(setq (aref p 0) 1)" :variables '("p"))))))

(test incf-evaluates-place-subforms-once
  (let* ((node (walk-text "(incf (aref p (f i)) 2)" :variables '("p" "i")))
         (calls (nodes-of-type 'gpu:call node)))
    (is (= 1 (count (ksym "f") calls :key #'gpu:call-operator)))
    (is (typep node 'gpu:let-node))
    (is (eq 'aref (gpu:call-operator
                   (gpu:assign-target (first (gpu:let-node-body node))))))))

(test decf-on-a-variable
  (let* ((node (walk-text "(let ((a 3)) (decf a))"))
         (assign (first (gpu:let-node-body node))))
    (is (typep assign 'gpu:assign))
    (is (eq '- (gpu:call-operator (gpu:assign-value assign))))))

(test return-targets-the-loop-block
  (let* ((node (walk-text "(dotimes (i 4) (return i))"))
         (return-node (first (nodes-of-type 'gpu:return-node node))))
    (is (typep node 'gpu:block-node))
    (is (eq node (gpu:return-node-block return-node)))))

(test return-targets-while-block
  (let* ((node (walk-text "(while t (return))"))
         (return-node (first (nodes-of-type 'gpu:return-node node))))
    (is (eq node (gpu:return-node-block return-node)))
    (is (null (gpu:return-node-value return-node)))))

(test return-from-finds-named-block
  (let* ((node (walk-text "(block done (while t (return-from done 1)))"))
         (return-node (first (nodes-of-type 'gpu:return-node node))))
    (is (eq node (gpu:return-node-block return-node)))))

(test return-from-unknown-block-is-an-error
  (is (equal '(2 3) (error-position-of gpu:kernel-syntax-error
                      (walk-text (format nil "(block a~%  (return-from b 1))"))))))

(test for-and-while-make-loop-nodes
  (let* ((for-node (first (nodes-of-type 'gpu:loop-node
                                         (walk-text "(for (i 0 (< i 4) (incf i)) (foo i))"))))
         (while-node (first (nodes-of-type 'gpu:loop-node
                                           (walk-text "(while (< 1 2) (foo))")))))
    (is (gpu:loop-node-binding for-node))
    (is (typep (gpu:loop-node-init for-node) 'gpu:literal))
    (is (typep (gpu:loop-node-test for-node) 'gpu:call))
    (is (typep (gpu:loop-node-step for-node) 'gpu:assign))
    (is (null (gpu:loop-node-binding while-node)))
    (is (null (gpu:loop-node-step while-node)))))

(test dotimes-evaluates-count-once
  (let* ((node (walk-text "(dotimes (i (f n)) (foo i))" :variables '("n")))
         (calls (nodes-of-type 'gpu:call node)))
    (is (typep node 'gpu:let-node))
    (is (= 1 (count (ksym "f") calls :key #'gpu:call-operator)))))

(test declarations-are-kept-raw
  (let ((node (walk-text "(let ((a 1)) (declare (type f32 a)) a)")))
    (is (equal (list (cst:raw (read-one "(declare (type f32 a))")))
               (gpu:let-node-declarations node)))
    (is (= 1 (length (gpu:let-node-body node))))))

(test misplaced-declare-is-an-error
  (signals gpu:kernel-syntax-error (walk-text "(progn (declare (type f32 a)))")))

(test cond-case-when-unless-expand-to-ifs
  (is (typep (walk-text "(cond (a 1) (b 2) (t 3))" :variables '("a" "b")) 'gpu:if-node))
  (is (typep (walk-text "(when a 1)" :variables '("a")) 'gpu:if-node))
  (is (typep (walk-text "(unless a 1)" :variables '("a")) 'gpu:if-node))
  (let ((node (walk-text "(case n (1 :one) ((2 3) :few) (t :many))" :variables '("n"))))
    (is (typep node 'gpu:let-node))
    (is (typep (first (gpu:let-node-body node)) 'gpu:if-node))))

(test and-or-stay-calls
  (is (eq 'and (gpu:call-operator (walk-text "(and a b c)" :variables '("a" "b" "c")))))
  (is (eq 'or (gpu:call-operator (walk-text "(or a b)" :variables '("a" "b"))))))

(test user-kernel-macros-expand
  (gpu:define-kernel-macro cl-gpu/tests/kernel::test-square (x) `(* ,x ,x))
  (let ((node (walk-text "(test-square 3)")))
    (is (eq '* (gpu:call-operator node)))
    (is (= 2 (length (gpu:call-args node))))))

(test unsupported-forms-report-their-location
  (is (equal '(2 3) (error-position-of gpu:kernel-syntax-error
                      (walk-text (format nil "(progn~%  (loop for i below 3 do (foo)))")))))
  (is (equal '(2 3) (error-position-of gpu:kernel-syntax-error
                      (walk-text (format nil "(progn~%  (flet ((f ())) (f)))")))))
  (is (equal '(1 1) (error-position-of gpu:kernel-syntax-error
                      (walk-text "(tagbody)")))))

(test unbound-variable-reports-its-location
  (is (equal '(2 7) (error-position-of gpu:kernel-syntax-error
                      (walk-text (format nil "(+ 1~%      zz)"))))))

(test macro-expansion-errors-point-at-the-macro-call
  (gpu:define-kernel-macro cl-gpu/tests/kernel::test-bad-expansion () '(loop))
  (is (equal '(2 3) (error-position-of gpu:kernel-syntax-error
                      (walk-text (format nil "(progn~%  (test-bad-expansion))"))))))

(test malformed-macro-call-is-an-error
  (is (equal '(1 1) (error-position-of gpu:kernel-syntax-error
                      (walk-text "(dotimes)")))))

(test runaway-macro-hits-the-depth-cap
  (gpu:define-kernel-macro cl-gpu/tests/kernel::test-forever ()
    '(cl-gpu/tests/kernel::test-forever))
  (signals gpu:kernel-syntax-error (walk-text "(test-forever)")))

(test errors-without-source-use-nearest-location
  (let ((cst (read-one (format nil "(progn~%  (foo))"))))
    (is (equal '(1 1)
               (error-position-of gpu:kernel-syntax-error
                 (gpu:walk-kernel cst :variables '())
                 (gpu:walk-kernel (cst:cst-from-expression '(loop)
                                                           :source (cst:source cst))))))))
