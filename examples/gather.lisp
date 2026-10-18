(require :asdf)
(asdf:load-system "cl-gpu/metal")
(asdf:load-system "cl-gpu/gather")

(cl-gpu:with-gpu-array (data (cl-gpu:to-device #(1f0 2f0 3f0 10f0 20f0 30f0 100f0 200f0 300f0)))
  (cl-gpu:with-gpu-array (ids (cl-gpu:to-device #(2 0 -1 1) :element-type 'cl-gpu:i32))
    (let* ((embeddings (cl-gpu:make-strided-view data '(3 3)))
           (token-ids (cl-gpu:make-strided-view ids '(2 2)))
           (result (cl-gpu/gather:gather embeddings token-ids)))
      (unwind-protect
           (format t "Embedding shape: ~s~%Values: ~s~%"
                   (cl-gpu:strided-view-shape result)
                   (cl-gpu:from-device (cl-gpu:strided-view-data result)))
        (cl-gpu:free (cl-gpu:strided-view-data result))))))
