(in-package #:cl-gpu/blas)

;;; Elementwise kernels: one thread per element, indexed from the span start.

(define-kernel %saxpy ((x (ptr f32 :global)) (y (ptr f32 :global)) (alpha f32) (n u32)
                       (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
  (let ((i (global-id 0)))
    (when (< i n)
      (let ((k (convert i64 i)))
        (incf (aref y (+ y-start (* k y-inc))) (* alpha (aref x (+ x-start (* k x-inc)))))))))

(define-kernel %sscal ((x (ptr f32 :global)) (alpha f32) (n u32) (x-start i64) (x-inc i64))
  (let ((i (global-id 0)))
    (when (< i n)
      (let ((k (convert i64 i)))
        (setf (aref x (+ x-start (* k x-inc)))
              (if (= alpha 0.0) 0.0 (* alpha (aref x (+ x-start (* k x-inc))))))))))

(define-kernel %scopy ((x (ptr f32 :global)) (y (ptr f32 :global)) (n u32)
                       (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
  (let ((i (global-id 0)))
    (when (< i n)
      (let ((k (convert i64 i)))
        (setf (aref y (+ y-start (* k y-inc))) (aref x (+ x-start (* k x-inc))))))))

(define-kernel %sswap ((x (ptr f32 :global)) (y (ptr f32 :global)) (n u32)
                       (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
  (let ((i (global-id 0)))
    (when (< i n)
      (let* ((k (convert i64 i))
             (held (aref x (+ x-start (* k x-inc)))))
        (setf (aref x (+ x-start (* k x-inc))) (aref y (+ y-start (* k y-inc)))
              (aref y (+ y-start (* k y-inc))) held)))))

;;; Reductions: each group of 256 threads writes one partial result and the host combines them.
;;; Threads stride over the input by the grid size and take part in every barrier.

(defconstant +group-size+ 256)
(defconstant +max-groups+ 256)

(define-kernel %sdot ((x (ptr f32 :global)) (y (ptr f32 :global)) (partials (ptr f32 :global))
                      (n u32) (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
  (let ((tile (shared-array f32 256))
        (lid (local-id 0))
        (sum 0.0))
    (declare (type f32 sum))
    (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
      (let ((k (convert i64 i)))
        (incf sum (* (aref x (+ x-start (* k x-inc))) (aref y (+ y-start (* k y-inc)))))))
    (setf (aref tile lid) sum)
    (barrier)
    (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
      (when (< lid s)
        (incf (aref tile lid) (aref tile (+ lid s))))
      (barrier))
    (when (= lid 0)
      (setf (aref partials (group-id 0)) (aref tile 0)))))

(define-kernel %sasum ((x (ptr f32 :global)) (partials (ptr f32 :global))
                       (n u32) (x-start i64) (x-inc i64))
  (let ((tile (shared-array f32 256))
        (lid (local-id 0))
        (sum 0.0))
    (declare (type f32 sum))
    (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
      (incf sum (abs (aref x (+ x-start (* (convert i64 i) x-inc))))))
    (setf (aref tile lid) sum)
    (barrier)
    (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
      (when (< lid s)
        (incf (aref tile lid) (aref tile (+ lid s))))
      (barrier))
    (when (= lid 0)
      (setf (aref partials (group-id 0)) (aref tile 0)))))

;;; (scale, ssq) pairs as in LAPACK: the norm is scale*sqrt(ssq). Scale 0 means no elements yet.
(define-kernel %snrm2 ((x (ptr f32 :global)) (scale-out (ptr f32 :global))
                       (ssq-out (ptr f32 :global)) (n u32) (x-start i64) (x-inc i64))
  (let ((scales (shared-array f32 256))
        (squares (shared-array f32 256))
        (lid (local-id 0))
        (scale 0.0)
        (ssq 1.0))
    (declare (type f32 scale ssq))
    (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
      (let ((a (abs (aref x (+ x-start (* (convert i64 i) x-inc))))))
        (when (> a 0.0)
          (if (< scale a)
              (let ((r (/ scale a)))
                (setf ssq (+ 1.0 (* ssq r r))
                      scale a))
              (let ((r (/ a scale)))
                (incf ssq (* r r)))))))
    (setf (aref scales lid) scale
          (aref squares lid) ssq)
    (barrier)
    (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
      (when (< lid s)
        (let ((s1 (aref scales lid))
              (q1 (aref squares lid))
              (s2 (aref scales (+ lid s)))
              (q2 (aref squares (+ lid s))))
          (when (> s2 0.0)
            (cond ((= s1 0.0)
                   (setf (aref scales lid) s2
                         (aref squares lid) q2))
                  ((< s1 s2)
                   (let ((r (/ s1 s2)))
                     (setf (aref scales lid) s2
                           (aref squares lid) (+ q2 (* q1 r r)))))
                  (t
                   (let ((r (/ s2 s1)))
                     (setf (aref squares lid) (+ q1 (* q2 r r)))))))))
      (barrier))
    (when (= lid 0)
      (setf (aref scale-out (group-id 0)) (aref scales 0)
            (aref ssq-out (group-id 0)) (aref squares 0)))))

;;; The largest magnitude wins and ties go to the lower logical index. A thread with no element
;;; holds magnitude -1.
(define-kernel %isamax ((x (ptr f32 :global)) (best-out (ptr f32 :global))
                        (index-out (ptr u32 :global)) (n u32) (x-start i64) (x-inc i64))
  (let ((bests (shared-array f32 256))
        (indices (shared-array u32 256))
        (lid (local-id 0))
        (best -1.0)
        (found 4294967295))
    (declare (type f32 best) (type u32 found))
    (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
      (let ((a (abs (aref x (+ x-start (* (convert i64 i) x-inc))))))
        (when (> a best)
          (setf best a
                found i))))
    (setf (aref bests lid) best
          (aref indices lid) found)
    (barrier)
    (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
      (when (< lid s)
        (let ((b (aref bests (+ lid s)))
              (j (aref indices (+ lid s))))
          (when (or (> b (aref bests lid))
                    (and (= b (aref bests lid)) (< j (aref indices lid))))
            (setf (aref bests lid) b
                  (aref indices lid) j))))
      (barrier))
    (when (= lid 0)
      (setf (aref best-out (group-id 0)) (aref bests 0)
            (aref index-out (group-id 0)) (aref indices 0)))))

;;; Routines

(defun saxpy (n alpha x incx y incy &key (x-offset 0) (y-offset 0))
  "Set Y to ALPHA*X+Y over N elements. Return Y."
  (let* ((xs (check-span 'saxpy n x incx x-offset))
         (ys (check-span 'saxpy n y incy y-offset))
         (alpha (real-scalar 'saxpy "ALPHA" alpha))
         (device (same-device 'saxpy xs ys)))
    (check-disjoint 'saxpy ys xs)
    (unless (or (zerop n) (zerop alpha))
      (%saxpy x y alpha n (span-start xs) incx (span-start ys) incy :grid n :device device))
    y))

(defun sscal (n alpha x incx &key (x-offset 0))
  "Set X to ALPHA*X over N elements. Return X."
  (let* ((xs (check-span 'sscal n x incx x-offset))
         (alpha (real-scalar 'sscal "ALPHA" alpha)))
    (unless (zerop n)
      (%sscal x alpha n (span-start xs) incx :grid n :device (gpu-array-device x)))
    x))

(defun scopy (n x incx y incy &key (x-offset 0) (y-offset 0))
  "Copy N elements of X to Y. Return Y."
  (let* ((xs (check-span 'scopy n x incx x-offset))
         (ys (check-span 'scopy n y incy y-offset))
         (device (same-device 'scopy xs ys)))
    (check-disjoint 'scopy ys xs)
    (unless (zerop n)
      (%scopy x y n (span-start xs) incx (span-start ys) incy :grid n :device device))
    y))

(defun sswap (n x incx y incy &key (x-offset 0) (y-offset 0))
  "Exchange N elements of X and Y. Return X and Y."
  (let* ((xs (check-span 'sswap n x incx x-offset))
         (ys (check-span 'sswap n y incy y-offset))
         (device (same-device 'sswap xs ys)))
    (check-disjoint 'sswap xs ys)
    (unless (zerop n)
      (%sswap x y n (span-start xs) incx (span-start ys) incy :grid n :device device))
    (values x y)))

(defun group-count (n)
  (min +max-groups+ (ceiling n +group-size+)))

;; TODO: one partials array per call, a reusable per-device buffer would avoid the allocation (#43)
(defmacro with-partials ((variable device count &optional (type 'f32)) &body body)
  `(with-gpu-array (,variable (make-gpu-array ,count :element-type ',type :device ,device))
     ,@body))

(defun sdot (n x incx y incy &key (x-offset 0) (y-offset 0))
  "The dot product of N elements of X and Y."
  (let* ((xs (check-span 'sdot n x incx x-offset))
         (ys (check-span 'sdot n y incy y-offset))
         (device (same-device 'sdot xs ys)))
    (if (zerop n)
        0f0
        (let ((groups (group-count n)))
          (with-partials (partials device groups)
            (%sdot x y partials n (span-start xs) incx (span-start ys) incy
                   :grid (* groups +group-size+) :group +group-size+ :device device)
            (float (reduce #'+ (from-device partials) :key (lambda (v) (float v 1d0))) 1f0))))))

(defun sasum (n x incx &key (x-offset 0))
  "The sum of the absolute values of N elements of X."
  (let ((xs (check-span 'sasum n x incx x-offset))
        (device (gpu-array-device x)))
    (if (zerop n)
        0f0
        (let ((groups (group-count n)))
          (with-partials (partials device groups)
            (%sasum x partials n (span-start xs) incx
                    :grid (* groups +group-size+) :group +group-size+ :device device)
            (float (reduce #'+ (from-device partials) :key (lambda (v) (float v 1d0))) 1f0))))))

(defun snrm2 (n x incx &key (x-offset 0))
  "The Euclidean norm of N elements of X."
  (let ((xs (check-span 'snrm2 n x incx x-offset))
        (device (gpu-array-device x)))
    (if (zerop n)
        0f0
        (let ((groups (group-count n)))
          (with-partials (scales device groups)
            (with-partials (squares device groups)
              (%snrm2 x scales squares n (span-start xs) incx
                      :grid (* groups +group-size+) :group +group-size+ :device device)
              (let ((scale 0d0) (ssq 1d0))
                (map nil (lambda (s q)
                           (let ((s (float s 1d0)) (q (float q 1d0)))
                             (cond ((zerop s))
                                   ((zerop scale) (setf scale s ssq q))
                                   ((< scale s) (setf ssq (+ q (* ssq (expt (/ scale s) 2))) scale s))
                                   (t (incf ssq (* q (expt (/ s scale) 2)))))))
                     (from-device scales) (from-device squares))
                (float (* scale (sqrt ssq)) 1f0))))))))

(defun isamax (n x incx &key (x-offset 0))
  "The zero-based index, in visiting order, of the first of N elements of X with the largest magnitude."
  (let ((xs (check-span 'isamax n x incx x-offset))
        (device (gpu-array-device x)))
    (if (zerop n)
        0
        (let ((groups (group-count n)))
          (with-partials (bests device groups)
            (with-partials (indices device groups u32)
              (%isamax x bests indices n (span-start xs) incx
                       :grid (* groups +group-size+) :group +group-size+ :device device)
              (let ((best -1f0) (found 0))
                (map nil (lambda (b i)
                           (when (or (> b best) (and (= b best) (< i found)))
                             (setf best b found i)))
                     (from-device bests) (from-device indices))
                (min found (1- n)))))))))
