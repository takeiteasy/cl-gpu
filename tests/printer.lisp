(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(defun kernel-text (text &key (arguments "()") (name 'kernel))
  "The Metal source for the kernel TEXT."
  (let ((*package* (find-package '#:cl-gpu/tests/kernel)))
    (gpu:kernel-source (gpu:lower-kernel (read-one text)
                                         :name name
                                         :arguments (read-from-string arguments))
                       :metal)))

(defun contains-p (text fragment)
  (and (search fragment text) t))

(defun lines-of (text)
  (mapcar (lambda (line) (string-trim " " line))
          (uiop:split-string text :separator '(#\Newline))))

(defun has-line-p (text line)
  (and (member line (lines-of text) :test #'string=) t))

(defparameter *scalars* "((a i32) (b i32) (c i32) (x f32) (out (ptr i32 :global)))")

(defun body-line (expression)
  "The line that stores EXPRESSION into out[0], with the scalar parameters above in scope."
  (let ((text (kernel-text (format nil "(setf (aref out 0) ~a)" expression)
                           :arguments *scalars*)))
    (find-if (lambda (line) (eql 0 (search "out[0] = " line))) (lines-of text))))

;;; Expressions

(test arithmetic-uses-minimal-parentheses
  (is (string= "out[0] = a + b * c;" (body-line "(+ a (* b c))")))
  (is (string= "out[0] = (a + b) * c;" (body-line "(* (+ a b) c)")))
  (is (string= "out[0] = a - (b - c);" (body-line "(- a (- b c))")))
  (is (string= "out[0] = a - b - c;" (body-line "(- a b c)"))))

(test bitwise-and-shift-mixes-are-parenthesised
  (is (string= "out[0] = (a & b) | c;" (body-line "(logior (logand a b) c)")))
  (is (string= "out[0] = (a << b) + c;" (body-line "(+ (shl a b) c)")))
  (is (string= "out[0] = a & b & c;" (body-line "(logand a b c)"))))

(test comparison-chains-need-no-parentheses
  (is (has-line-p (kernel-text "(when (and (< a b) (< b c)) (setf (aref out 0) 1))"
                               :arguments *scalars*)
                  "if (a < b && b < c) {")))

(test unary-operators-do-not-merge
  (is (string= "out[0] = -(-a);" (body-line "(- (- a))")))
  (is (string= "out[0] = ~a;" (body-line "(lognot a)"))))

(test conditional-values-print-as-ternaries
  (is (string= "out[0] = a < b ? a : b;" (body-line "(if (< a b) a b)"))))

;;; Literals

(defun float-line (type literal)
  (let ((text (kernel-text (format nil "(setf (aref out 0) ~a)" literal)
                           :arguments (format nil "((out (ptr ~a :global)))" type))))
    (find-if (lambda (line) (eql 0 (search "out[0] = " line))) (lines-of text))))

(test float-literals-carry-their-type
  (is (string= "out[0] = 1.5f;" (float-line "f32" "1.5f0")))
  (is (string= "out[0] = 1.0e10f;" (float-line "f32" "1f10")))
  (is (string= "out[0] = 2.5h;" (float-line "f16" "2.5f0")))
  (is (string= "out[0] = static_cast<bfloat>(0.5f);" (float-line "bf16" "0.5f0")))
  (is (string= "out[0] = -1.5f;" (float-line "f32" "-1.5f0"))))

(test integer-literals-carry-their-type
  (is (string= "out[0] = 7u;" (float-line "u32" "7")))
  (is (string= "out[0] = 7l;" (float-line "i64" "7")))
  (is (string= "out[0] = 7ul;" (float-line "u64" "7")))
  (is (string= "out[0] = static_cast<uchar>(200);" (float-line "u8" "200")))
  (is (string= "out[0] = -5;" (float-line "i32" "-5")))
  (is (string= "out[0] = (-2147483647 - 1);" (float-line "i32" "-2147483648"))))

(test boolean-literals-print-as-keywords
  (is (has-line-p (kernel-text "(setf (aref out 0) t)" :arguments "((out (ptr bool :global)))")
                  "out[0] = true;")))

;;; Narrow integers

(test narrow-integer-arithmetic-is-cast-back
  (let ((text (kernel-text "(setf (aref out 0) (+ (aref out 1) (aref out 2)))"
                           :arguments "((out (ptr u8 :global)))")))
    (is (has-line-p text "out[0] = static_cast<uchar>(out[1] + out[2]);")))
  (let ((text (kernel-text "(setf (aref out 0) (logand (aref out 1) (aref out 2)))"
                           :arguments "((out (ptr u8 :global)))")))
    (is (has-line-p text "out[0] = out[1] & out[2];"))))

;;; Names

(test locals-are-numbered-and-parameters-are-plain
  (let ((text (kernel-text "(let ((v 1)) (let ((v 2)) (setf (aref out 0) (+ v 0))))"
                           :arguments "((out (ptr i32 :global)))")))
    (is (contains-p text "int v_"))
    (is (contains-p text "device int* out [[buffer(0)]]"))
    (is (= 2 (count-if (lambda (line) (eql 0 (search "int v_" line))) (lines-of text))))))

(test names-are-sanitised
  (is (string= "kernel_name" (gpu::sanitise-name 'kernel-name)))
  (is (string= "v_special_" (gpu::sanitise-name '*special*)))
  (is (string= "v2d" (gpu::sanitise-name '2d)))
  (is (string= "a_b" (gpu::sanitise-name 'a--b))))

(test parameters-avoid-reserved-names
  (let ((text (kernel-text "(setf (aref min 0) (aref max 0))"
                           :arguments "((min (ptr i32 :global)) (max (ptr i32 :global)))")))
    (is (contains-p text "device int* min_ [[buffer(0)]]"))
    (is (contains-p text "min_[0] = max_[0];")))
  (let ((text (kernel-text "(setf (aref float 0) (aref thread_3 0))"
                           :arguments "((float (ptr i32 :global)) (thread_3 (ptr i32 :global)))")))
    (is (contains-p text "float_ "))
    (is (contains-p text "thread_3_ "))))

(test clashing-parameter-names-are-an-error
  (signals gpu:kernel-target-error
    (kernel-text "nil" :arguments "((a-b i32) (a_b i32))")))

(test kernel-name-is-sanitised
  (is (contains-p (kernel-text "nil" :name '*my-kernel*) "kernel void v_my_kernel_(")))

;;; Statements

(test else-if-chains-are-flattened
  (let ((text (kernel-text "(cond ((< a b) (setf (aref out 0) 1))
                                  ((< b c) (setf (aref out 0) 2))
                                  (t (setf (aref out 0) 3)))"
                           :arguments *scalars*)))
    (is (has-line-p text "if (a < b) {"))
    (is (has-line-p text "} else if (b < c) {"))
    (is (has-line-p text "} else {"))))

(test loops-print-as-c-loops
  (let ((text (kernel-text "(dotimes (i a) (setf (aref out i) i))" :arguments *scalars*)))
    (is (contains-p text "for (int i_"))
    (is (contains-p text "= i_"))
    (is (not (contains-p text "while"))))
  (let ((text (kernel-text "(while (< a b) (setf a (+ a 1)))" :arguments *scalars*)))
    (is (contains-p text "while (a < b) {"))))

(test return-and-break-print
  (let ((text (kernel-text "(progn (when (> a 0) (return)) (dotimes (i 4) (when (= i 2) (return))))"
                           :arguments *scalars*)))
    (is (has-line-p text "return;"))
    (is (has-line-p text "break;"))))

(test unsupported-types-are-a-target-error
  (signals gpu:kernel-target-error
    (kernel-text "(setf (aref out 0) 1.0d0)" :arguments "((out (ptr f64 :global)))")))

(test unknown-targets-are-a-target-error
  (signals gpu:kernel-target-error
    (gpu:kernel-source (lower-text "nil") :cuda)))
