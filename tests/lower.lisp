(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(defparameter *saxpy-arguments*
  "((out (ptr f32 :global)) (x (ptr f32 :global)) (y (ptr f32 :global)) (a f32) (n u32))")

(defun f32 () (type-of-spec "f32"))
(defun i32 () (type-of-spec "i32"))
(defun u32 () (type-of-spec "u32"))
(defun bool () (type-of-spec "bool"))

(defun type-error-position (text &key (arguments "()"))
  (error-position-of gpu:kernel-type-error (lower-text text :arguments arguments)))

;;; Kernels

(test lowers-a-saxpy-kernel
  (let* ((kernel (lower-text "(let ((i (global-id 0)))
                                (when (< i n)
                                  (setf (aref out i) (+ (* a (aref x i)) (aref y i)))))"
                             :arguments *saxpy-arguments*))
         (parameters (gpu:ir-kernel-parameters kernel)))
    (is (equal '("OUT" "X" "Y" "A" "N")
               (mapcar (lambda (parameter) (symbol-name (gpu:ir-variable-name parameter)))
                       parameters)))
    (is (eq (type-of-spec "(ptr f32 :global)") (gpu:ir-variable-type (first parameters))))
    (is (equal '("BOOL" "F32" "U32")
               (mapcar (lambda (type) (string-upcase (symbol-name (gpu:scalar-type-name type))))
                       (gpu:ir-kernel-scalar-types kernel))))
    (let ((body (gpu:ir-kernel-body kernel)))
      (is (typep (first body) 'gpu:ir-declare))
      (is (typep (second body) 'gpu:ir-if))
      (let ((assign (first (gpu:ir-if-then (second body)))))
        (is (typep (gpu:ir-assign-target assign) 'gpu:ir-index))
        (is (eq :add (gpu:ir-op-operator (gpu:ir-assign-value assign))))))))

(test kernel-parameters-are-typed
  (signals gpu:kernel-type-error
    (lower-text "nil" :arguments "((p (atomic u32)))"))
  (signals gpu:kernel-type-error
    (lower-text "nil" :arguments "((p float))")))

(test kernel-variables-have-unique-ids
  (let* ((kernel (lower-text "(let ((a 1)) (let ((a 2)) a))"))
         (variables (mapcar #'gpu:ir-declare-variable (ir-nodes-of-type 'gpu:ir-declare kernel))))
    (is (= 2 (length variables)))
    (is (not (eql (gpu:ir-variable-id (first variables)) (gpu:ir-variable-id (second variables)))))))

(test kernel-return-leaves-the-kernel
  (let ((kernel (lower-text "(progn (when (>= (global-id 0) n) (return)) (setf (aref out 0) 1.0))"
                            :arguments "((out (ptr f32 :global)) (n u32))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-return kernel))))
    (is (null (ir-nodes-of-type 'gpu:ir-break kernel)))
    (is (null (ir-nodes-of-type 'gpu:ir-while kernel)))))

(test kernel-return-from-inside-loops-needs-no-flags
  (let ((kernel (lower-text "(dotimes (i 4) (dotimes (j 4) (when (= (* i j) 6) (return-from nil))))")))
    (is (= 2 (length (ir-nodes-of-type 'gpu:ir-for kernel))))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-break kernel))))))

;;; Literals and typing

(test literals-adopt-the-type-they-meet
  (let ((kernel (lower-text "(let ((r (+ x 1))) r)" :arguments "((x f32))")))
    (let ((literal (first (ir-nodes-of-type 'gpu:ir-literal kernel))))
      (is (eq (f32) (gpu:ir-expr-type literal)))
      (is (eql 1.0 (gpu:ir-literal-value literal))))))

(test literals-default-by-their-lisp-type
  (flet ((literal-type (text)
           (gpu:ir-expr-type (first (ir-nodes-of-type 'gpu:ir-literal (lower-text text))))))
    (is (eq (i32) (literal-type "(let ((a 1)) a)")))
    (is (eq (type-of-spec "i64") (literal-type "(let ((a 3000000000)) a)")))
    (is (eq (f32) (literal-type "(let ((a 1.5)) a)")))
    (is (eq (type-of-spec "f64") (literal-type "(let ((a 1.5d0)) a)")))
    (is (eq (bool) (literal-type "(let ((a t)) a)")))))

(test float-literals-are-typed-by-their-value
  (let ((*read-default-float-format* 'double-float))
    (is (eq (type-of-spec "f64")
            (gpu:ir-expr-type (first (ir-nodes-of-type 'gpu:ir-literal
                                                       (lower-text "(let ((a 1.5)) a)"))))))
    (is (eq (f32)
            (gpu:ir-expr-type (first (ir-nodes-of-type 'gpu:ir-literal
                                                       (lower-text "(let ((a 1.5f0)) a)"))))))))

(test constant-expressions-adopt-the-type-of-their-operand
  (let ((kernel (lower-text "(let ((r (+ (* 2 3) x))) r)" :arguments "((x f32))")))
    (is (every (lambda (literal) (eq (f32) (gpu:ir-expr-type literal)))
               (ir-nodes-of-type 'gpu:ir-literal kernel)))))

(test mixing-types-is-an-error
  (is (equal '(1 15) (type-error-position "(let ((r (+ i 1.5))) r)" :arguments "((i i32))")))
  (is (equal '(1 15) (type-error-position "(let ((r (+ x i))) r)" :arguments "((x f32) (i i32))"))))

(test literals-must-fit-their-type
  (signals gpu:kernel-type-error (lower-text "(let ((r (+ b 300))) r)" :arguments "((b u8))"))
  (signals gpu:kernel-type-error (lower-text "(let ((r (+ b -1))) r)" :arguments "((b u8))"))
  (is (typep (lower-text "(let ((r (+ b 255))) r)" :arguments "((b u8))") 'gpu:ir-kernel)))

(test declarations-set-variable-types
  (let ((kernel (lower-text "(let ((a 1)) (declare (type f32 a)) a)")))
    (is (eq (f32) (gpu:ir-variable-type
                   (gpu:ir-declare-variable (first (ir-nodes-of-type 'gpu:ir-declare kernel))))))
    (is (eql 1.0 (gpu:ir-literal-value (first (ir-nodes-of-type 'gpu:ir-literal kernel))))))
  (let ((kernel (lower-text "(let ((a)) (declare (type u32 a)) (setf a 3))")))
    (is (null (gpu:ir-declare-init (first (ir-nodes-of-type 'gpu:ir-declare kernel))))))
  (signals gpu:kernel-type-error (lower-text "(let ((a)) a)"))
  (signals gpu:kernel-type-error (lower-text "(let ((a 1.5)) (declare (type i32 a)) a)")))

(test assignments-must-match-the-place
  (signals gpu:kernel-type-error (lower-text "(setf x 1.5)" :arguments "((x i32))"))
  (signals gpu:kernel-type-error (lower-text "(setf x i)" :arguments "((x f32) (i i32))"))
  (signals gpu:kernel-type-error (lower-text "(setf (foo x) 1)" :arguments "((x f32))"))
  (is (typep (lower-text "(setf x 2)" :arguments "((x f32))") 'gpu:ir-kernel)))

(test unknown-functions-are-errors
  (is (equal '(2 3) (type-error-position (format nil "(progn~%  (frobnicate 1))")))))

(test errors-point-at-the-offending-form
  (is (equal '(2 4) (type-error-position (format nil "(if~%   n 1 2)") :arguments "((n i32))"))))

;;; Operators

(test arithmetic-folds-left
  (let* ((kernel (lower-text "(let ((r (- a b c))) r)" :arguments "((a i32) (b i32) (c i32))"))
         (op (gpu:ir-declare-init (first (ir-nodes-of-type 'gpu:ir-declare kernel)))))
    (is (eq :sub (gpu:ir-op-operator op)))
    (is (eq :sub (gpu:ir-op-operator (first (gpu:ir-op-operands op)))))))

(test unary-minus-and-bitwise
  (is (eq :neg (gpu:ir-op-operator
                (gpu:ir-declare-init
                 (first (ir-nodes-of-type 'gpu:ir-declare
                                          (lower-text "(let ((r (- a))) r)" :arguments "((a f32))")))))))
  (signals gpu:kernel-type-error (lower-text "(let ((r (- a))) r)" :arguments "((a u32))"))
  (signals gpu:kernel-type-error (lower-text "(let ((r (logand a b))) r)" :arguments "((a f32) (b f32))"))
  (is (typep (lower-text "(let ((r (shl (logxor a 1) 2))) r)" :arguments "((a u32))") 'gpu:ir-kernel))
  (signals gpu:kernel-type-error (lower-text "(let ((r (mod a 2))) r)" :arguments "((a i32))"))
  (is (typep (lower-text "(let ((r (mod a 2))) r)" :arguments "((a u32))") 'gpu:ir-kernel)))

(test comparisons-give-bool
  (let* ((kernel (lower-text "(let ((r (< a b))) r)" :arguments "((a f32) (b f32))"))
         (op (gpu:ir-declare-init (first (ir-nodes-of-type 'gpu:ir-declare kernel)))))
    (is (eq (bool) (gpu:ir-expr-type op)))
    (is (eq :lt (gpu:ir-op-operator op))))
  (signals gpu:kernel-type-error (lower-text "(let ((r (< a b))) r)" :arguments "((a bool) (b bool))")))

(test chained-comparisons-bind-the-middle-once
  (let* ((kernel (lower-text "(let ((r (< a (abs b) c))) r)"
                             :arguments "((a i32) (b i32) (c i32))"))
         (ops (ir-nodes-of-type 'gpu:ir-op kernel)))
    (is (eq :and (gpu:ir-op-operator (gpu:ir-declare-init (car (last (ir-nodes-of-type 'gpu:ir-declare kernel)))))))
    (is (= 2 (count :lt ops :key #'gpu:ir-op-operator)))))

(test tests-must-be-bool
  (signals gpu:kernel-type-error (lower-text "(if n 1 2)" :arguments "((n i32))"))
  (signals gpu:kernel-type-error (lower-text "(while n (barrier))" :arguments "((n i32))"))
  (signals gpu:kernel-type-error (lower-text "(and n t)" :arguments "((n i32))"))
  (signals gpu:kernel-type-error (lower-text "(not n)" :arguments "((n i32))")))

(test body-less-cond-clauses-need-bool-tests
  (is (typep (lower-text "(let ((r (cond ((< a 1)) (t nil)))) r)" :arguments "((a i32))")
             'gpu:ir-kernel))
  (signals gpu:kernel-type-error (lower-text "(let ((r (cond (a) (t nil)))) r)" :arguments "((a i32))")))

(test and-or-short-circuit-with-statements
  (let ((kernel (lower-text "(let ((r (and (> n 0) (let ((m (* n 2))) (> m 3))))) r)"
                            :arguments "((n i32))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-if kernel)))))
  (let ((kernel (lower-text "(let ((r (or (> n 0) (< n -5)))) r)" :arguments "((n i32))")))
    (is (null (ir-nodes-of-type 'gpu:ir-if kernel)))
    (is (member :or (mapcar #'gpu:ir-op-operator (ir-nodes-of-type 'gpu:ir-op kernel))))))

(test aref-on-pointers-and-vectors
  (let ((kernel (lower-text "(let ((v (aref p 0)) (c 0.0)) (setf c (aref v 2)) (setf (aref p 1) v))"
                            :arguments "((p (ptr (vec f32 4) :global)))")))
    (is (typep (first (ir-nodes-of-type 'gpu:ir-index kernel)) 'gpu:ir-index))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-component kernel))))
    (is (eq (f32) (gpu:ir-expr-type (first (ir-nodes-of-type 'gpu:ir-component kernel))))))
  (signals gpu:kernel-type-error (lower-text "(aref v 4)" :arguments "((v (vec f32 4)))"))
  (signals gpu:kernel-type-error (lower-text "(aref v i)" :arguments "((v (vec f32 4)) (i u32))"))
  (signals gpu:kernel-type-error (lower-text "(aref p 1.0)" :arguments "((p (ptr f32 :global)))"))
  (signals gpu:kernel-type-error (lower-text "(aref x 0)" :arguments "((x f32))")))

(test vector-construction
  (let ((kernel (lower-text "(let ((v (vec f32 x 2 3 4))) v)" :arguments "((x f32))")))
    (is (eq (type-of-spec "(vec f32 4)")
            (gpu:ir-expr-type (first (ir-nodes-of-type 'gpu:ir-construct kernel))))))
  (signals gpu:kernel-type-error (lower-text "(vec f32 1)"))
  (signals gpu:kernel-type-error (lower-text "(vec f32 n 1)" :arguments "((n i32))"))
  (signals gpu:kernel-type-error (lower-text "(vec bool t t)")))

;;; Statements in value position

(test simple-if-values-become-ternaries
  (let ((kernel (lower-text "(let ((r (if c 1.0 2.0))) r)" :arguments "((c bool))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-ternary kernel))))
    (is (null (ir-nodes-of-type 'gpu:ir-if kernel)))))

(test if-branches-agree-on-a-type
  (let ((kernel (lower-text "(let ((r (if c 1 x))) r)" :arguments "((c bool) (x f32))")))
    (is (eq (f32) (gpu:ir-expr-type (first (ir-nodes-of-type 'gpu:ir-ternary kernel))))))
  (signals gpu:kernel-type-error (lower-text "(let ((r (if c i x))) r)"
                                             :arguments "((c bool) (x f32) (i i32))"))
  (signals gpu:kernel-type-error (lower-text "(let ((r (if c 1))) r)" :arguments "((c bool))")))

(test statements-in-if-branches-use-a-result-variable
  (let ((kernel (lower-text "(let ((r (if c (let ((m (* n 2))) (+ m 1)) 0))) r)"
                            :arguments "((c bool) (n i32))")))
    (is (null (ir-nodes-of-type 'gpu:ir-ternary kernel)))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-if kernel))))
    (let ((branch (gpu:ir-if-then (first (ir-nodes-of-type 'gpu:ir-if kernel)))))
      (is (typep (first branch) 'gpu:ir-declare))
      (is (typep (car (last branch)) 'gpu:ir-assign)))))

(test let-in-value-position-hoists-its-variables
  (let ((kernel (lower-text "(let ((r (+ 1 (let ((y n)) (* y y))))) r)" :arguments "((n i32))")))
    (let ((declares (ir-nodes-of-type 'gpu:ir-declare kernel)))
      (is (equal '("Y" "R") (mapcar (lambda (declare)
                                      (symbol-name (gpu:ir-variable-name (gpu:ir-declare-variable declare))))
                                    declares))))))

(test operands-keep-evaluation-order
  (let* ((kernel (lower-text "(let ((r (+ x (progn (setf x 5) x)))) r)" :arguments "((x i32))"))
         (declares (ir-nodes-of-type 'gpu:ir-declare kernel))
         (first-declare (first declares)))
    (is (gpu:ir-variable-temp (gpu:ir-declare-variable first-declare)))
    (is (typep (gpu:ir-declare-init first-declare) 'gpu:ir-var))
    (is (typep (second (gpu:ir-kernel-body kernel)) 'gpu:ir-assign))))

(test assignment-values
  (let ((kernel (lower-text "(let ((r (setf (aref p 0) 3))) r)" :arguments "((p (ptr i32 :global)))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-assign kernel))))
    (is (typep (first (gpu:ir-kernel-body kernel)) 'gpu:ir-declare))))

(test case-and-incf-in-value-position
  (let ((kernel (lower-text "(let ((r (case n (1 10) ((2 3) 20) (t 30)))) (incf r))"
                            :arguments "((n i32))")))
    (is (typep kernel 'gpu:ir-kernel))
    (is (= 2 (length (ir-nodes-of-type 'gpu:ir-ternary kernel))))))

(test incf-evaluates-place-subforms-once-in-ir
  (let ((kernel (lower-text "(incf (aref p (global-id 0)) 2)" :arguments "((p (ptr u32 :global)))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-intrinsic kernel))))))

(test incf-through-symbol-macro-evaluates-place-subforms-once
  (let ((kernel (lower-text "(symbol-macrolet ((x (aref p (global-id 0)))) (incf x 2))"
                            :arguments "((p (ptr u32 :global)))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-intrinsic kernel))))))

(test incf-on-a-vector-component-writes-the-vector
  (let* ((kernel (lower-text "(let ((v (vec f32 1.0 2.0 3.0))) (incf (aref v 0) 1.5))"))
         (assign (first (ir-nodes-of-type 'gpu:ir-assign kernel))))
    (is (typep (gpu:ir-assign-target assign) 'gpu:ir-component))
    (is (typep (gpu:ir-component-base (gpu:ir-assign-target assign)) 'gpu:ir-var))))

(test incf-in-value-position-returns-the-new-value
  (let ((kernel (lower-text "(setf (aref p 0) (incf (aref p 1)))" :arguments "((p (ptr i32 :global)))")))
    (is (= 2 (length (ir-nodes-of-type 'gpu:ir-assign kernel))))))

(test incf-needs-a-numeric-place
  (signals gpu:kernel-type-error (lower-text "(let ((b t)) (incf b))")))

(test flet-call-in-value-and-statement-position
  (let ((kernel (lower-text "(flet ((sq (x) (* x x)))
                               (setf (aref p 0) (sq (aref p 1)))
                               (sq 3))"
                            :arguments "((p (ptr i32 :global)))")))
    (is (typep kernel 'gpu:ir-kernel))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-assign kernel))))))

(test flet-calls-infer-their-own-types
  (let ((kernel (lower-text "(flet ((twice (x) (* x 2)))
                               (setf (aref a 0) (twice (aref a 1)))
                               (setf (aref b 0) (twice (aref b 1))))"
                            :arguments "((a (ptr i32 :global)) (b (ptr f32 :global)))")))
    (is (equal (list (i32) (f32))
               (mapcar #'gpu:ir-expr-type
                       (mapcar #'gpu:ir-assign-value (ir-nodes-of-type 'gpu:ir-assign kernel)))))))

(test local-function-may-use-thread-positions-and-captures
  (let ((kernel (lower-text "(let ((base 10))
                               (declare (type u32 base))
                               (labels ((cell () (+ base (global-id 0)))
                                        (store (v) (declare (type u32 v)) (setf (aref p (cell)) v)))
                                 (store 1)))"
                            :arguments "((p (ptr u32 :global)))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-intrinsic kernel))))))

(test local-function-return-leaves-only-the-function
  (let ((kernel (lower-text "(flet ((f (x) (when (> x 2) (return-from f 1)) 2))
                               (setf (aref p 0) (f (aref p 1))))
                             (setf (aref p 2) 3)"
                            :arguments "((p (ptr i32 :global)))")))
    (is (null (ir-nodes-of-type 'gpu:ir-return kernel)))))

(test unknown-function-outside-flet-is-a-type-error
  (signals gpu:kernel-type-error (lower-text "(flet ((f (x) (g x)) (g (x) x)) (f 1))")))

(test value-statements-have-no-value
  (signals gpu:kernel-type-error (lower-text "(let ((r (while t (barrier)))) r)"))
  (signals gpu:kernel-type-error (lower-text "(let ((r (progn))) r)"))
  (signals gpu:kernel-type-error (lower-text "(let ((r (dotimes (i 3) (return i)))) r)")))

;;; Loops and blocks

(test for-loops-lower-to-for
  (let* ((kernel (lower-text "(for (i 0 (< i n) (incf i)) (setf (aref p i) 0))"
                             :arguments "((p (ptr i32 :global)) (n i32))"))
         (loop-node (first (ir-nodes-of-type 'gpu:ir-for kernel))))
    (is (typep (gpu:ir-for-step loop-node) 'gpu:ir-assign))
    (is (eq (i32) (gpu:ir-variable-type (gpu:ir-for-variable loop-node))))))

(test while-loops-lower-to-while
  (let ((kernel (lower-text "(while (< n 10) (incf n))" :arguments "((n i32))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-while kernel))))
    (is (null (ir-nodes-of-type 'gpu:ir-for kernel)))))

(test dotimes-variable-takes-the-count-type
  (let ((kernel (lower-text "(dotimes (i n) (barrier))" :arguments "((n u32))")))
    (is (eq (u32) (gpu:ir-variable-type
                   (gpu:ir-for-variable (first (ir-nodes-of-type 'gpu:ir-for kernel)))))))
  (let ((kernel (lower-text "(dotimes (i 8) (barrier))")))
    (is (eq (i32) (gpu:ir-variable-type
                   (gpu:ir-for-variable (first (ir-nodes-of-type 'gpu:ir-for kernel)))))))
  (let ((kernel (lower-text "(dotimes (i n) (declare (type u64 i)) (barrier))" :arguments "((n u64))")))
    (is (eq (type-of-spec "u64")
            (gpu:ir-variable-type (gpu:ir-for-variable (first (ir-nodes-of-type 'gpu:ir-for kernel))))))))

(test loop-init-must-match-the-declaration
  (signals gpu:kernel-type-error
    (lower-text "(dotimes (i n) (declare (type i32 i)) (barrier))" :arguments "((n u32))")))

(test return-in-a-loop-is-a-break
  (let ((kernel (lower-text "(for (i 0 (< i 4) (incf i)) (when (= i 2) (return)))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-break kernel))))
    (is (null (ir-nodes-of-type 'gpu:ir-while kernel)))))

(test return-from-an-outer-loop-sets-a-flag
  (let* ((kernel (lower-text "(block done
                                (dotimes (i 3)
                                  (dotimes (j 3)
                                    (when (= (+ i j) 3) (return-from done)))
                                  (barrier))
                                (barrier))"))
         (body (gpu:ir-kernel-body kernel)))
    (is (typep (first body) 'gpu:ir-declare))
    (is (null (gpu:ir-literal-value (gpu:ir-declare-init (first body)))))
    (is (= 4 (length (ir-nodes-of-type 'gpu:ir-break kernel))))
    (is (= 3 (length (ir-nodes-of-type 'gpu:ir-if kernel))))))

(test block-values-use-a-result-variable
  (let ((kernel (lower-text "(let ((r (block b (when c (return-from b 1.0)) 2.0))) r)"
                            :arguments "((c bool))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-while kernel))))
    (is (= 2 (length (ir-nodes-of-type 'gpu:ir-break kernel))))))

(test blocks-without-value-need-no-result
  (let ((kernel (lower-text "(block b (when c (return-from b)) (barrier))" :arguments "((c bool))")))
    (is (= 1 (length (ir-nodes-of-type 'gpu:ir-while kernel))))))

(test return-without-value-in-a-value-block-is-an-error
  (signals gpu:kernel-type-error
    (lower-text "(let ((r (block b (when c (return-from b)) 2.0))) r)" :arguments "((c bool))"))
  (signals gpu:kernel-type-error
    (lower-text "(let ((r (block b (when c (return-from b 1)) 2.0))) r)" :arguments "((c bool))")))

(test loop-test-with-statements-falls-back-to-while-true
  (let ((kernel (lower-text "(while (let ((m (* n 2))) (< m 10)) (incf n))" :arguments "((n i32))")))
    (let ((loop-node (first (ir-nodes-of-type 'gpu:ir-while kernel))))
      (is (eq t (gpu:ir-literal-value (gpu:ir-while-test loop-node))))
      (is (typep (second (gpu:ir-while-body loop-node)) 'gpu:ir-if)))))
