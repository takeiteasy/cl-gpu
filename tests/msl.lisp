(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

;;; Signatures

(test parameters-bind-by-position
  (let ((text (kernel-text "nil"
                           :arguments "((a (ptr f32 :global)) (n u32) (k (ptr i32 :constant))
                                        (tile (ptr f32 :shared)) (v (vec f32 4)))")))
    (is (contains-p text "device float* a [[buffer(0)]]"))
    (is (contains-p text "constant uint& n [[buffer(1)]]"))
    (is (contains-p text "constant int* k [[buffer(2)]]"))
    (is (contains-p text "threadgroup float* tile [[threadgroup(0)]]"))
    (is (contains-p text "constant float4& v [[buffer(3)]]"))))

(test private-pointer-parameters-are-a-target-error
  (signals gpu:kernel-target-error
    (kernel-text "nil" :arguments "((p (ptr f32 :private)))")))

(test thread-positions-become-parameters
  (let ((text (kernel-text "(setf (aref out (global-id 1)) (local-id 2))"
                           :arguments "((out (ptr u32 :global)))")))
    (is (contains-p text "uint3 cl_gpu_thread_position_in_grid [[thread_position_in_grid]]"))
    (is (contains-p text "uint3 cl_gpu_thread_position_in_threadgroup [[thread_position_in_threadgroup]]"))
    (is (contains-p text "out[cl_gpu_thread_position_in_grid.y] = cl_gpu_thread_position_in_threadgroup.z;")))
  (is (not (contains-p (kernel-text "nil") "uint3"))))

(test assigned-scalar-parameters-are-copied
  (let ((text (kernel-text "(setf n (+ n 1))" :arguments "((n u32) (m u32))")))
    (is (contains-p text "constant uint& cl_gpu_n [[buffer(0)]]"))
    (is (contains-p text "constant uint& m [[buffer(1)]]"))
    (is (has-line-p text "uint n = cl_gpu_n;"))
    (is (has-line-p text "n = n + 1u;"))))

;;; Intrinsics

(test barriers-name-their-scope
  (is (contains-p (kernel-text "(barrier)") "threadgroup_barrier(mem_flags::mem_threadgroup);"))
  (is (contains-p (kernel-text "(barrier :global)") "threadgroup_barrier(mem_flags::mem_device);"))
  (is (contains-p (kernel-text "(barrier :all)")
                  "threadgroup_barrier(mem_flags::mem_device_and_threadgroup);")))

(test atomics-take-a-memory-order
  (let ((text (kernel-text "(progn (atomic-add c 1) (setf (aref out 0) (atomic-load c)))"
                           :arguments "((c (ptr (atomic u32) :global)) (out (ptr u32 :global)))")))
    (is (contains-p text "device atomic_uint* c [[buffer(0)]]"))
    (is (contains-p text "atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);"))
    (is (contains-p text "atomic_load_explicit(c, memory_order_relaxed)"))
    (is (not (contains-p text "cl_gpu_atomic_cas")))))

(test compare-and-swap-uses-a-helper
  (let ((text (kernel-text "(setf (aref out 0) (atomic-cas c 1 2))"
                           :arguments "((c (ptr (atomic i32) :shared)) (out (ptr i32 :global)))")))
    (is (contains-p text "inline int cl_gpu_atomic_cas(threadgroup atomic_int* p"))
    (is (contains-p text "= cl_gpu_atomic_cas(c, 1, 2);"))))

(test math-calls-print-by-name
  (let ((text (kernel-text "(setf (aref out 0) (fma (sqrt x) (min x 2.0) (abs x)))"
                           :arguments "((x f32) (out (ptr f32 :global)))")))
    (is (contains-p text "fma(sqrt(x), min(x, 2.0f), abs(x))"))))

(test conversions-and-vectors
  (let ((text (kernel-text "(let ((v (vec f32 x 2.0 3.0 4.0)))
                              (setf (aref out 0) (convert i32 (aref v 1)))
                              (setf (aref out 1) (bitcast i32 x)))"
                           :arguments "((x f32) (out (ptr i32 :global)))")))
    (is (contains-p text "float4 v_"))
    (is (contains-p text "float4(x, 2.0f, 3.0f, 4.0f)"))
    (is (contains-p text "static_cast<int>(v_"))
    (is (contains-p text "as_type<int>(x)"))))

;;; Compiling the output

(defun metal-available-p ()
  (zerop (nth-value 2 (uiop:run-program '("xcrun" "-sdk" "macosx" "-f" "metal")
                                        :ignore-error-status t :output nil :error-output nil))))

(defun compile-metal (source)
  "Compile SOURCE with the Metal compiler. Return NIL, or the compiler's complaints."
  (uiop:with-temporary-file (:pathname input :type "metal" :keep nil)
    (with-open-file (out input :direction :output :if-exists :supersede)
      (write-string source out))
    (let ((output (make-pathname :type "air" :defaults input)))
      (unwind-protect
           (multiple-value-bind (stdout stderr status)
               (uiop:run-program (list "xcrun" "-sdk" "macosx" "metal" "-c"
                                       (namestring input) "-o" (namestring output))
                                 :ignore-error-status t :output :string :error-output :string)
             (declare (ignore stdout))
             (unless (zerop status) stderr))
        (uiop:delete-file-if-exists output)))))

(defparameter *corpus*
  '(("saxpy"
     "(let ((i (global-id 0)))
        (when (< i n)
          (setf (aref out i) (+ (* a (aref x i)) (aref y i)))))"
     "((out (ptr f32 :global)) (x (ptr f32 :global)) (y (ptr f32 :global)) (a f32) (n u32))")
    ("block-sum"
     "(let ((i (local-id 0)))
        (setf (aref tile i) (aref in (global-id 0)))
        (barrier)
        (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
          (when (< i s)
            (incf (aref tile i) (aref tile (+ i s))))
          (barrier))
        (when (= i 0)
          (atomic-add total (convert u32 (aref tile 0)))))"
     "((in (ptr f32 :global)) (total (ptr (atomic u32) :global)) (tile (ptr f32 :shared)))")
    ("compare-and-swap"
     "(let ((old (atomic-cas lock 0 1)))
        (when (= old 0)
          (atomic-store lock 0)))"
     "((lock (ptr (atomic u32) :global)))")
    ("shared-compare-and-swap"
     "(setf (aref out 0) (atomic-cas lock 0 1))"
     "((lock (ptr (atomic i32) :shared)) (out (ptr i32 :global)))")
    ("vectors"
     "(let ((v (vec f32 x 2.0 3.0 4.0)))
        (setf (aref out 0) (convert i32 (aref v 1)))
        (setf (aref out 1) (bitcast i32 x))
        (setf (aref w 0) (convert (vec i32 4) v)))"
     "((x f32) (out (ptr i32 :global)) (w (ptr (vec i32 4) :global)))")
    ("local-functions"
     "(symbol-macrolet ((cell (aref out (global-id 0))))
        (labels ((clamp (x) (if (> x hi) hi x))
                 (bump (x) (incf cell (clamp x))))
          (bump (aref in 0))
          (when (> (aref in 1) 0)
            (return))
          (decf cell 1)))"
     "((out (ptr i32 :global)) (in (ptr i32 :global)) (hi i32))")
    ("nested-loops"
     "(block search
        (dotimes (i 4)
          (dotimes (j 4)
            (when (= (* i j) k)
              (setf (aref out 0) (+ i j))
              (return-from search nil)))))"
     "((out (ptr i32 :global)) (k i32))")
    ("assigned-parameters"
     "(progn (setf n (+ n 1)) (setf (aref out 0) n) (setf (aref v 0) 1.0))"
     "((out (ptr u32 :global)) (n u32) (v (vec f32 2)))")
    ("half"
     "(setf (aref out 0) (* (aref in 0) 2.0))"
     "((out (ptr f16 :global)) (in (ptr f16 :global)))")
    ("narrow-integers"
     "(setf (aref out 0) (+ (lognot (aref out 1)) (shl (aref out 2) 1)))"
     "((out (ptr u8 :global)))")
    ("literals"
     "(setf (aref out 0) (+ (aref out 1) -2147483648))
      (setf (aref big 0) (- (aref big 1) 9223372036854775807))"
     "((out (ptr i32 :global)) (big (ptr i64 :global)))")
    ("pointers"
     "(let ((q p))
        (incf q 2)
        (decf q)
        (setf (aref (ptr+ q (incf i)) 0) 1.0)
        (setf p (ptr+ p 1))
        (atomic-add (ptr+ c i) 1)
        (when (and (< q p) (= p p)) (setf (aref q 0) 2.0))
        (setf (aref d 0) (ptr- q p)))"
     "((p (ptr f32 :global)) (c (ptr (atomic u32) :global)) (d (ptr i64 :global)) (i u32))")
    ("vectors"
     "(let ((a (aref vs 0)))
        (setf (aref vs 1) (+ (* a 2.0) (/ 1 (swizzle a :wzyx))))
        (setf (aref vs 2) (- x a))
        (setf (aref ws 0) (mod (aref ws 1) 3))
        (setf (aref o 0) (+ (swizzle a :b) (aref (swizzle a :xy) 1)))
        (setf (aref us 0) (vec f32 (swizzle a :r) (swizzle a :g) (swizzle a :b))))"
     "((vs (ptr (vec f32 4) :global)) (ws (ptr (vec i32 2) :global)) (o (ptr f32 :global))
       (us (ptr (vec f32 3) :global)) (x f32))")
    ("remainders"
     "(progn
        (setf (aref o 0) (mod (aref o 1) (aref o 2)))
        (setf (aref o 3) (rem (aref o 1) -3))
        (setf (aref f 0) (mod (aref f 1) 3.0))
        (setf (aref f 2) (rem (aref f 1) 3.0))
        (setf (aref h 0) (mod (aref h 1) (aref h 2)))
        (setf (aref h 3) (rem (aref h 1) (aref h 2)))
        (setf (aref v 0) (mod (aref v 1) (aref v 2)))
        (setf (aref w 0) (mod (aref w 1) (aref w 2)))
        (setf (aref w 3) (rem (aref w 1) (aref w 2))))"
     "((o (ptr i32 :global)) (f (ptr f32 :global)) (h (ptr f16 :global))
       (v (ptr (vec i32 4) :global)) (w (ptr (vec f32 4) :global)))")
    ("math"
     "(setf (aref out 0) (clamp (pow (exp x) (rsqrt (log (floor x)))) 0.0 1.0))"
     "((x f32) (out (ptr f32 :global)))")))

(test generated-source-compiles
  (if (metal-available-p)
      (loop for (name text arguments) in *corpus*
            do (let ((errors (compile-metal (kernel-text text :arguments arguments :name name))))
                 (is (null errors) "~a did not compile:~%~a" name errors)))
      (skip "the Metal compiler is not available")))

(test bfloat-remainders-are-a-target-error
  (signals gpu:kernel-target-error
    (kernel-text "(setf (aref h 0) (rem (aref h 1) (aref h 2)))" :arguments "((h (ptr bf16 :global)))"))
  (signals gpu:kernel-target-error
    (kernel-text "(setf (aref h 0) (mod (aref h 1) (aref h 2)))" :arguments "((h (ptr bf16 :global)))")))

(test floored-mod-defines-its-helper-only-when-used
  (let ((signed (kernel-text "(setf (aref o 0) (mod (aref o 1) 3))" :arguments "((o (ptr i32 :global)))"))
        (floats (kernel-text "(setf (aref o 0) (mod (aref o 1) 3.0))" :arguments "((o (ptr f32 :global)))"))
        (plain (kernel-text "(setf (aref o 0) (rem (aref o 1) 3))" :arguments "((o (ptr i32 :global)))")))
    (is (contains-p signed "cl_gpu_mod(o[1], 3)"))
    (is (contains-p signed "T r = a % b;"))
    (is (contains-p floats "cl_gpu_fmod(o[1], 3.0f)"))
    (is (not (contains-p plain "cl_gpu_")))
    (is (contains-p plain "o[1] % 3"))))

(test broadcasts-and-swizzles-print-as-constructors-and-selectors
  (let ((text (kernel-text "(let ((v (aref vs 0)))
                              (setf (aref vs 1) (* v 2.0))
                              (setf (aref o 0) (aref (swizzle v :zx) 0)))"
                           :arguments "((vs (ptr (vec f32 4) :global)) (o (ptr f32 :global)))")))
    (is (contains-p text "v_3 * float4(2.0f)"))
    (is (contains-p text "v_3.zx.x"))))

(test pointer-comparisons-print-as-c
  (let ((text (kernel-text "(when (and (/= p q) (< p (ptr+ q 1))) (setf (aref p 0) 1.0))"
                           :arguments "((p (ptr f32 :global)) (q (ptr f32 :global)))")))
    (is (contains-p text "p != q && p < q + 1"))))

(test pointer-arithmetic-prints-c-pointer-arithmetic
  (let ((text (kernel-text "(progn (setf (aref (ptr+ p 1) 0) 1.0) (setf (aref d 0) (ptr- p (ptr+ p 2))))"
                           :arguments "((p (ptr f32 :global)) (d (ptr i64 :global)))")))
    (is (contains-p text "(p + 1)[0] = 1.0f;"))
    (is (contains-p text "d[0] = p - (p + 2);"))))
