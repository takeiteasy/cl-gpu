(require :asdf)
(asdf:load-system "cl-gpu/metal")
(asdf:load-system "cl-gpu/elementwise")

(cl-gpu:with-gpu-array (data (cl-gpu:to-device #(1f0 2f0 3f0 10f0 20f0)))
  (let* ((rows (cl-gpu:make-strided-view data '(2 1) :offset 3))
         (columns (cl-gpu:make-strided-view data '(1 3)))
         (sum (cl-gpu/elementwise:add rows columns)))
    (unwind-protect
         (let ((integers (cl-gpu/elementwise:convert sum 'cl-gpu:i8)))
           (unwind-protect
                (format t "Broadcast sums: ~s~%Integer conversion: ~s~%"
                        (cl-gpu:from-device (cl-gpu:strided-view-data sum))
                        (cl-gpu:from-device (cl-gpu:strided-view-data integers)))
             (cl-gpu:free (cl-gpu:strided-view-data integers))))
      (cl-gpu:free (cl-gpu:strided-view-data sum)))))
