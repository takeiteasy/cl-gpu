(in-package #:cl-gpu/tests)
(in-suite :cl-gpu)

(defun parts-text (body arguments &key initialized)
  (let ((*package* (find-package '#:cl-gpu/tests/kernel)))
    (gpu:kernel-parts (gpu:lower-kernel (read-one body) :name 'embedded
                                      :arguments (read-from-string arguments))
                      :metal :initialized-outputs initialized)))

(test embedded-parts-order-and-body
  (let ((parts (parts-text "(setf (aref out (global-id 0)) (* (aref x (global-id 0)) gain))"
                           "((out (ptr f32 :global)) (x (ptr f32 :global)) (gain f32))")))
    (is (equal '(:output :input :value) (mapcar #'gpu:kernel-parameter-mode (gpu:kernel-parts-parameters parts))))
    (is (equal '("x" "gain") (mapcar #'gpu:kernel-parameter-name (gpu:kernel-parts-inputs parts))))
    (is (equal '("out") (mapcar #'gpu:kernel-parameter-name (gpu:kernel-parts-outputs parts))))
    (is (string= "    out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * gain;
" (gpu:kernel-parts-body parts)))
    (is (equal '("thread_position_in_grid") (gpu:kernel-parts-attributes parts))))
  (signals gpu:kernel-target-error (gpu:kernel-parts (gpu:lower-kernel nil) :cuda)))

(test embedded-alias-access-and-initialization
  (let ((body "(let ((a (ptr+ out 1))) (setf (aref a 0) (+ (aref out 0) 1.0)))")
        (args "((out (ptr f32 :global)))"))
    (signals gpu:kernel-target-error (parts-text body args))
    (is (eq :inout (gpu:kernel-parameter-mode (first (gpu:kernel-parts-outputs (parts-text body args :initialized t))))))))

(test embedded-rejects-unbindable-parameters
  (dolist (space '(:private :shared))
    (signals gpu:kernel-target-error (parts-text "nil" (format nil "((x (ptr f32 ~s)))" space)))))

(test embedded-shared-and-value-copies
  (let ((parts (parts-text "(let ((tile (shared-array f32 4 :zero))) (setf n (+ n 1)) (setf (aref out 0) (+ (aref tile 0) (convert f32 n))))"
                           "((out (ptr f32 :global)) (n u32))")))
    (is (search "thread_index_in_threadgroup" (gpu:kernel-parts-body parts)))
    (is (not (search "cl_gpu_thread_index" (gpu:kernel-parts-body parts))))
    (is (search "uint n = cl_gpu_n;" (gpu:kernel-parts-body parts)))
    (is (string= "cl_gpu_n" (gpu:kernel-parameter-name (second (gpu:kernel-parts-parameters parts)))))))

(test embedded-parts-compile-in-host-signature
  (if (metal-available-p)
      (let* ((parts (parts-text "(setf (aref out (global-id 0)) (+ (aref x (global-id 0)) n))"
                                "((out (ptr f32 :global)) (x (ptr f32 :global)) (n f32))"))
             (source (format nil "~a kernel void host(const device float* x [[buffer(0)]], constant float& n [[buffer(1)]], device float* out [[buffer(2)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {~%~a}"
                             (gpu:kernel-parts-header parts) (gpu:kernel-parts-body parts))))
        (is (null (compile-metal source))))
      (skip "Metal compiler unavailable")))

(test embedded-atomic-accesses-and-helpers
  (let ((args "((out (ptr (atomic u32) :global)))"))
    (signals gpu:kernel-target-error (parts-text "(atomic-add out 1)" args))
    (let ((parts (parts-text "(atomic-cas out 0 1)" args :initialized t)))
      (is (eq :inout (gpu:kernel-parameter-mode (first (gpu:kernel-parts-outputs parts)))))
      (is (search "inline uint cl_gpu_atomic_cas" (gpu:kernel-parts-header parts)))))
  (let ((parts (parts-text "(atomic-store out 1)" "((out (ptr (atomic u32) :global)))")))
    (is (eq :output (gpu:kernel-parameter-mode (first (gpu:kernel-parts-outputs parts)))))))

(test embedded-alias-fixed-point
  (let* ((parts (parts-text "(let ((a x) (b out)) (setf a b) (setf (aref a 0) (aref b 1)))"
                            "((x (ptr f32 :global)) (out (ptr f32 :global)))" :initialized t))
         (parameters (gpu:kernel-parts-parameters parts)))
    (is (equal '(:output :inout) (mapcar #'gpu:kernel-parameter-mode parameters)))))
