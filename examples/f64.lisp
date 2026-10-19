(defpackage #:cl-gpu/examples/f64
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu) (#:ocl #:cl-gpu/opencl)
                    (#:ew #:cl-gpu/elementwise) (#:red #:cl-gpu/reductions))
  (:export #:run-example))
(in-package #:cl-gpu/examples/f64)

(defun run-example (&optional (device (or gpu:*device* (ocl:default-device))))
  (unless (gpu:device-native-type-p device 'gpu:f64)
    (error 'gpu:gpu-unsupported-capability :device device :operation 'run-example :element-type 'gpu:f64
           :message "This example requires native f64 computation"))
  (let ((gpu:*device* device))
    (gpu:with-gpu-array (data (gpu:to-device #(16777217d0 1d0) :element-type 'gpu:f64))
      (let* ((input (gpu:make-strided-view data '(2))) (added (ew:add input 1d0)))
        (unwind-protect
             (let ((total (red:sum input)))
               (unwind-protect
                    (list :add (gpu:from-device (gpu:strided-view-data added))
                          :sum (gpu:from-device (gpu:strided-view-data total)))
                 (gpu:free (gpu:strided-view-data total))))
          (gpu:free (gpu:strided-view-data added)))))))
