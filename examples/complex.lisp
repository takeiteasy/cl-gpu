(defpackage #:cl-gpu/examples/complex
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu) (#:ew #:cl-gpu/elementwise))
  (:export #:run-example))
(in-package #:cl-gpu/examples/complex)

(defun run-example (&optional (device (gpu:current-device)))
  "Multiply a reversed complex window and return its magnitudes."
  (gpu:with-gpu-array (data (gpu:to-device #(#c(1f0 2f0) #c(3f0 -4f0)) :device device))
    (let* ((view (gpu:make-strided-view data '(2) :strides '(-1) :offset 1))
           (product (ew:multiply view #c(0f0 1f0))))
      (unwind-protect
           (let ((magnitude (ew:abs product)))
             (unwind-protect
                  (list :product (gpu:from-device (gpu:strided-view-data product))
                        :magnitude (gpu:from-device (gpu:strided-view-data magnitude)))
               (gpu:free (gpu:strided-view-data magnitude))))
        (gpu:free (gpu:strided-view-data product))))))
