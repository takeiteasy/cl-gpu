(asdf:load-system "cl-gpu/reductions")
(asdf:load-system "cl-gpu/metal")

(defpackage #:cl-gpu/reductions-example
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu) (#:red #:cl-gpu/reductions))
  (:export #:run-example))

(in-package #:cl-gpu/reductions-example)

(defun run-example ()
  (gpu:with-gpu-array (data (gpu:to-device #(1.0 2.0 3.0 4.0 5.0 6.0)))
    (let ((input (red:make-strided-view data '(2 3))))
      (gpu:with-gpu-array (totals (red:strided-view-data (red:sum input :axis 1)))
        (gpu:with-gpu-array (probabilities (red:strided-view-data (red:softmax input)))
          (gpu:with-gpu-array (normalized (red:strided-view-data (red:rmsnorm input :weights 2)))
            (list :sum (gpu:from-device totals)
                  :softmax (gpu:from-device probabilities)
                  :rmsnorm (gpu:from-device normalized))))))))
