(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(defun spelling (name target &optional type)
  (multiple-value-list (gpu:intrinsic-spelling (gpu:find-intrinsic (ksym name)) target
                                               (and type (type-of-spec type)))))

(test intrinsics-have-per-target-spellings
  (is (equal '(:param "thread_position_in_grid" nil) (spelling "global-id" :metal)))
  (is (equal '(:expr "blockIdx.{axis} * blockDim.{axis} + threadIdx.{axis}" nil)
             (spelling "global-id" :cuda)))
  (is (equal '(:call "get_global_id" nil) (spelling "global-id" :opencl)))
  (is (equal '(:call "__syncthreads" nil) (spelling "barrier" :cuda))))

(test spellings-override-by-element-type
  (is (equal "sqrtf" (second (spelling "sqrt" :cuda "f32"))))
  (is (equal "sqrt" (second (spelling "sqrt" :cuda "f64"))))
  (is (equal "sqrtf" (second (spelling "sqrt" :cuda "(vec f32 4)"))))
  (is (equal "fabs" (second (spelling "abs" :opencl "f32"))))
  (is (equal "abs" (second (spelling "abs" :opencl "i32")))))

(test spellings-carry-options
  (is (equal '("memory_order_relaxed")
             (getf (third (spelling "atomic-add" :metal)) :extra)))
  (is (equal "mem_flags::mem_device" (getf (third (spelling "barrier" :metal)) :global))))

(test atomics-and-barrier-are-impure
  (is (gpu:intrinsic-impure-p (gpu:find-intrinsic (ksym "atomic-add"))))
  (is (gpu:intrinsic-impure-p (gpu:find-intrinsic (ksym "barrier"))))
  (is (not (gpu:intrinsic-impure-p (gpu:find-intrinsic (ksym "global-id")))))
  (is (not (gpu:intrinsic-impure-p (gpu:find-intrinsic 'sqrt)))))

(test thread-ids-are-u32
  (let* ((kernel (lower-text "(let ((i (global-id 0)) (j (local-id 1)) (k (group-id 2))
                                    (g (group-size 0)) (n (grid-size 0)))
                                (+ i j k g n))"))
         (calls (ir-nodes-of-type 'gpu:ir-intrinsic kernel)))
    (is (= 5 (length calls)))
    (is (every (lambda (call) (eq (type-of-spec "u32") (gpu:ir-expr-type call))) calls))
    (is (equal '(0 1 2 0 0) (mapcar (lambda (call) (getf (gpu:ir-intrinsic-options call) :axis))
                                    calls)))))

(test axis-must-be-a-literal-from-0-to-2
  (signals gpu:kernel-type-error (lower-text "(global-id 3)"))
  (signals gpu:kernel-type-error (lower-text "(global-id n)" :arguments "((n u32))"))
  (signals gpu:kernel-type-error (lower-text "(global-id)")))

(test barrier-scope
  (let ((kernel (lower-text "(progn (barrier) (barrier :global))")))
    (is (equal '(:shared :global)
               (mapcar (lambda (call) (getf (gpu:ir-intrinsic-options call) :scope))
                       (ir-nodes-of-type 'gpu:ir-intrinsic kernel))))
    (is (= 2 (length (ir-nodes-of-type 'gpu:ir-expr-stmt kernel)))))
  (signals gpu:kernel-type-error (lower-text "(barrier :everywhere)"))
  (signals gpu:kernel-type-error (lower-text "(+ 1 (barrier))")))

(test atomic-operations
  (let* ((kernel (lower-text "(progn (atomic-add counter 1)
                                      (let ((old (atomic-max counter 7)))
                                        (atomic-cas counter old 0)
                                        (atomic-store counter 2)
                                        (atomic-load counter)))"
                             :arguments "((counter (ptr (atomic u32) :global)))"))
         (calls (ir-nodes-of-type 'gpu:ir-intrinsic kernel)))
    (is (= 5 (length calls)))
    (is (every (lambda (call)
                 (or (eq (type-of-spec "u32") (gpu:ir-expr-type call))
                     (null (gpu:ir-expr-type call))))
               calls))
    (is (null (gpu:ir-expr-type (find (ksym "atomic-store") calls
                                      :key (lambda (call) (gpu:intrinsic-name (gpu:ir-intrinsic-intrinsic call)))))))))

(test atomic-results-are-bound-in-value-position
  (let ((kernel (lower-text "(let ((a (atomic-add c 1)) (b (atomic-add c 1))) (+ a b))"
                            :arguments "((c (ptr (atomic i32) :global)))")))
    (is (= 2 (count-if (lambda (declare)
                         (typep (gpu:ir-declare-init declare) 'gpu:ir-intrinsic))
                       (ir-nodes-of-type 'gpu:ir-declare kernel))))))

(test atomics-need-atomic-pointers
  (signals gpu:kernel-type-error
    (lower-text "(atomic-add p 1)" :arguments "((p (ptr u32 :global)))"))
  (signals gpu:kernel-type-error
    (lower-text "(atomic-add p 1)" :arguments "((p (ptr (atomic u32) :private)))"))
  (signals gpu:kernel-type-error
    (lower-text "(atomic-add p 1.0)" :arguments "((p (ptr (atomic u32) :global)))"))
  (signals gpu:kernel-type-error
    (lower-text "(aref p 0)" :arguments "((p (ptr (atomic u32) :global)))")))

(test convert-and-bitcast
  (let ((kernel (lower-text "(let* ((a (convert f32 n)) (b (bitcast u32 a)) (c (convert f32 1))
                                    (r (+ a (convert f32 b) c)))
                               r)"
                            :arguments "((n i32))")))
    (is (equal '(:convert :bitcast :convert)
               (mapcar #'gpu:ir-convert-kind (ir-nodes-of-type 'gpu:ir-convert kernel)))))
  (signals gpu:kernel-type-error
    (lower-text "(bitcast f64 n)" :arguments "((n i32))"))
  (signals gpu:kernel-type-error
    (lower-text "(convert (vec f32 4) n)" :arguments "((n i32))"))
  (signals gpu:kernel-type-error
    (lower-text "(bitcast bool n)" :arguments "((n u8))")))

(test constant-conversion-adopts-the-target-type
  (let ((kernel (lower-text "(convert f32 1)")))
    (is (null (ir-nodes-of-type 'gpu:ir-convert kernel)))))

(test math-intrinsics-type-rules
  (let ((kernel (lower-text "(let ((r (+ (sqrt x) (min x 2) (fma x x 1) (clamp x 0 1) (abs x)))) r)"
                            :arguments "((x f32))")))
    (is (= 5 (length (ir-nodes-of-type 'gpu:ir-intrinsic kernel)))))
  (signals gpu:kernel-type-error (lower-text "(sqrt n)" :arguments "((n i32))"))
  (signals gpu:kernel-type-error (lower-text "(min x n)" :arguments "((x f32) (n i32))"))
  (signals gpu:kernel-type-error (lower-text "(pow x)" :arguments "((x f32))")))
