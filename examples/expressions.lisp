(require :asdf)
(asdf:load-system "cl-gpu/metal")
(asdf:load-system "cl-gpu/expressions")

(let ((multiply-add (cl-gpu/expressions:make-expression-plan '(a b c) '(:add (:multiply a b) c)))
      (softmax (cl-gpu/expressions:make-expression-plan
                '(x) '(:divide (:exp (:subtract x (:maximum x)))
                               (:sum (:exp (:subtract x (:maximum x))))))))
  (cl-gpu:with-gpu-array (data (cl-gpu:to-device #(1f0 2f0 3f0 4f0 5f0 6f0)))
    (let* ((input (cl-gpu:make-strided-view data '(2 3)))
           (result (cl-gpu/expressions:execute-expression multiply-add (list input 2 1))))
      (unwind-protect
           (progn
             (format t "Multiply-add: ~s~%" (cl-gpu:from-device (cl-gpu:strided-view-data result)))
             (cl-gpu/expressions:execute-expression! result softmax (list input) :axis -1)
             (format t "Row softmax: ~s~%" (cl-gpu:from-device (cl-gpu:strided-view-data result))))
        (cl-gpu:free (cl-gpu:strided-view-data result))))))
