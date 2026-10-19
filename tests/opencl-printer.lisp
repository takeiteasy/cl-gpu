(in-package #:cl-gpu/tests)
(in-suite :cl-gpu)

(defun opencl-text (text &optional (arguments "()"))
  (gpu:kernel-source (lower-text text :arguments arguments) :opencl))

(test opencl-signatures-and-builtins
  (let ((text (opencl-text "(setf (aref out (global-id 0)) n)" "((out (ptr u32 :global)) (n u32))")))
    (is (contains-p text "__global uint* out"))
    (is (contains-p text "uint n"))
    (is (contains-p text "get_global_id(0)"))))

(test opencl-extensions-and-unsupported-types
  (is (contains-p (opencl-text "nil" "((x f64))") "cl_khr_fp64"))
  (is (contains-p (opencl-text "nil" "((x f16))") "cl_khr_fp16"))
  (signals gpu:kernel-target-error (opencl-text "nil" "((x bf16))"))
  (signals gpu:kernel-target-error (opencl-text "nil" "((x bool))"))
  (signals gpu:kernel-target-error (opencl-text "nil" "((x (ptr i32 :private)))")))

(test opencl-vectors-arrays-and-atomics
  (is (contains-p (opencl-text "(setf (aref out 0) (vec f32 1.0 2.0 3.0))" "((out (ptr (vec f32 3) :global)))") "(float3)("))
  (is (contains-p (opencl-text "(atomic-add p 1)" "((p (ptr (atomic u32) :global)))") "atomic_add(p, 1u)"))
  (is (contains-p (opencl-text "(let ((a (local-array f32 4 :zero))) (setf (aref a 0) 2.0))") "= {0}")))

(test opencl-mod-helpers
  (let ((text (opencl-text "(setf (aref out 0) (mod x y))" "((out (ptr i32 :global)) (x i32) (y i32))")))
    (is (contains-p text "inline int cl_gpu_mod_int"))
    (is (not (contains-p text "template")))))
