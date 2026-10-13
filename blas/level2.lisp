(in-package #:cl-gpu/blas)

;;; One thread per element of y; a row of A is walked by one thread.
;;; TODO: one group per row with a shared reduction would help few rows of many columns (#45)
(define-kernel %sgemv ((a (ptr f32 :global)) (x (ptr f32 :global)) (y (ptr f32 :global))
                       (alpha f32) (beta f32) (m u32) (n u32)
                       (a-off i64) (a-rs i64) (a-cs i64)
                       (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
  (let ((i (global-id 0)))
    (when (< i m)
      (let ((row (+ a-off (* (convert i64 i) a-rs)))
            (at (+ y-start (* (convert i64 i) y-inc)))
            (sum 0.0))
        (declare (type f32 sum))
        (for (j 0 (< j n) (incf j))
          (incf sum (* (aref a (+ row (* (convert i64 j) a-cs)))
                       (aref x (+ x-start (* (convert i64 j) x-inc))))))
        (setf (aref y at)
              (if (= beta 0.0)
                  (* alpha sum)
                  (+ (* alpha sum) (* beta (aref y at)))))))))

(defmethod device-gemv (device m n alpha a x beta y)
  (destructuring-bind (array offset row-stride column-stride) a
    (%sgemv array (first x) (first y) alpha beta m n offset row-stride column-stride
            (second x) (third x) (second y) (third y)
            :grid m :device device)))

(defun scale-span (device span beta)
  "Set the elements of SPAN to BETA times themselves."
  (unless (or (= beta 1f0) (zerop (span-count span)))
    (%sscal (span-array span) beta (span-count span) (span-start span) (span-increment span)
            :grid (span-count span) :device device)))

(defun sgemv (transpose alpha view x incx beta y incy &key (x-offset 0) (y-offset 0))
  "Set Y to ALPHA*op(VIEW)*X + BETA*Y, where op is the TRANSPOSE flag. Return Y."
  (let* ((a (view-operand 'sgemv view transpose))
         (m (operand-rows a))
         (n (operand-cols a))
         (xs (check-span 'sgemv n x incx x-offset))
         (ys (check-span 'sgemv m y incy y-offset))
         (alpha (real-scalar 'sgemv "ALPHA" alpha))
         (beta (real-scalar 'sgemv "BETA" beta))
         (device (same-device-arrays 'sgemv (operand-array a) x y)))
    (check-apart 'sgemv (span-region ys) (operand-region a) (span-region xs))
    (cond ((zerop m))
          ((or (zerop alpha) (zerop n))
           (scale-span device ys beta))
          (t
           (device-gemv device m n alpha (operand-list a)
                        (list x (span-start xs) incx) beta (list y (span-start ys) incy))))
    y))
