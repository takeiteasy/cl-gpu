(asdf:load-system "cl-gpu/opencl")

(defpackage #:cl-gpu/examples/opencl
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu) (#:ocl #:cl-gpu/opencl))
  (:export #:run))
(in-package #:cl-gpu/examples/opencl)

(gpu:define-kernel saxpy ((out (gpu:ptr gpu:f32 :global))
                         (x (gpu:ptr gpu:f32 :global)) (a gpu:f32))
  (let ((i (gpu:global-id 0)))
    (setf (aref out i) (+ (* a (aref x i)) (aref out i)))))

(defun run ()
  (let ((gpu:*device* (ocl:default-device)))
    (gpu:with-gpu-array (x (gpu:to-device #(1.0 2.0 3.0 4.0)))
      (gpu:with-gpu-array (out (gpu:make-gpu-array 4))
        (saxpy out x 2.0 :grid 4 :group 2)
        (gpu:from-device out)))))
