(in-package #:cl-gpu/blas)

;;; Kernels exist per element type. f16 and bf16 data is widened to f32 on load, all arithmetic and
;;; shared tiles are f32, and results are rounded once on store.

(defconstant +group-size+ 256)
(defconstant +max-groups+ 256)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *element-types* '(f32 f16 bf16))

  (defun kernel-symbol (operation element)
    (intern (format nil "%~a-~a" operation element) '#:cl-gpu/blas)))

(defvar *level1-kernels* (make-hash-table :test 'equal)
  "(operation element-type-name) -> the symbol of the kernel function.")

(defmacro define-level1-kernels (element)
  "Define the Level 1 kernels for arrays of ELEMENT."
  (flet ((w (array index)
           (if (eq element 'f32) `(aref ,array ,index) `(convert f32 (aref ,array ,index))))
         (narrow (form) (if (eq element 'f32) form `(convert ,element ,form)))
         (kernel (operation) (kernel-symbol operation element)))
    `(progn
       ,@(loop for operation in '("AXPY" "SCAL" "COPY" "SWAP" "DOT" "ASUM" "NRM2" "IAMAX")
               collect `(setf (gethash (list ,operation ',element) *level1-kernels*)
                              ',(kernel operation)))

       ;; Elementwise: one thread per element, indexed from the span start.
       (define-kernel ,(kernel "AXPY") ((x (ptr ,element :global)) (y (ptr ,element :global))
                                        (alpha f32) (n u32)
                                        (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
         (let ((i (global-id 0)))
           (when (< i n)
             (let* ((k (convert i64 i))
                    (at (+ y-start (* k y-inc))))
               (setf (aref y at)
                     ,(narrow `(+ ,(w 'y 'at)
                                  (* alpha ,(w 'x '(+ x-start (* k x-inc)))))))))))

       (define-kernel ,(kernel "SCAL") ((x (ptr ,element :global)) (alpha f32) (n u32)
                                        (x-start i64) (x-inc i64))
         (let ((i (global-id 0)))
           (when (< i n)
             (let ((at (+ x-start (* (convert i64 i) x-inc))))
               (setf (aref x at)
                     ,(narrow `(if (= alpha 0.0) 0.0 (* alpha ,(w 'x 'at)))))))))

       (define-kernel ,(kernel "COPY") ((x (ptr ,element :global)) (y (ptr ,element :global))
                                        (n u32) (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
         (let ((i (global-id 0)))
           (when (< i n)
             (let ((k (convert i64 i)))
               (setf (aref y (+ y-start (* k y-inc))) (aref x (+ x-start (* k x-inc))))))))

       (define-kernel ,(kernel "SWAP") ((x (ptr ,element :global)) (y (ptr ,element :global))
                                        (n u32) (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
         (let ((i (global-id 0)))
           (when (< i n)
             (let* ((k (convert i64 i))
                    (held (aref x (+ x-start (* k x-inc)))))
               (setf (aref x (+ x-start (* k x-inc))) (aref y (+ y-start (* k y-inc)))
                     (aref y (+ y-start (* k y-inc))) held)))))

       ;; Reductions: each group of 256 threads writes one partial result and the host combines
       ;; them. Threads stride over the input by the grid size and take part in every barrier.
       (define-kernel ,(kernel "DOT") ((x (ptr ,element :global)) (y (ptr ,element :global))
                                       (partials (ptr f32 :global)) (n u32)
                                       (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
         (let ((tile (shared-array f32 256))
               (lid (local-id 0))
               (sum 0.0))
           (declare (type f32 sum))
           (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
             (let ((k (convert i64 i)))
               (incf sum (* ,(w 'x '(+ x-start (* k x-inc)))
                            ,(w 'y '(+ y-start (* k y-inc)))))))
           (setf (aref tile lid) sum)
           (barrier)
           (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
             (when (< lid s)
               (incf (aref tile lid) (aref tile (+ lid s))))
             (barrier))
           (when (= lid 0)
             (setf (aref partials (group-id 0)) (aref tile 0)))))

       (define-kernel ,(kernel "ASUM") ((x (ptr ,element :global)) (partials (ptr f32 :global))
                                        (n u32) (x-start i64) (x-inc i64))
         (let ((tile (shared-array f32 256))
               (lid (local-id 0))
               (sum 0.0))
           (declare (type f32 sum))
           (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
             (incf sum (abs ,(w 'x '(+ x-start (* (convert i64 i) x-inc))))))
           (setf (aref tile lid) sum)
           (barrier)
           (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
             (when (< lid s)
               (incf (aref tile lid) (aref tile (+ lid s))))
             (barrier))
           (when (= lid 0)
             (setf (aref partials (group-id 0)) (aref tile 0)))))

       ;; (scale, ssq) pairs as in LAPACK: the norm is scale*sqrt(ssq). Scale 0 means no elements yet.
       (define-kernel ,(kernel "NRM2") ((x (ptr ,element :global)) (scale-out (ptr f32 :global))
                                        (ssq-out (ptr f32 :global)) (n u32)
                                        (x-start i64) (x-inc i64))
         (let ((scales (shared-array f32 256))
               (squares (shared-array f32 256))
               (lid (local-id 0))
               (scale 0.0)
               (ssq 1.0))
           (declare (type f32 scale ssq))
           (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
             (let ((a (abs ,(w 'x '(+ x-start (* (convert i64 i) x-inc))))))
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

       ;; The largest magnitude wins and ties go to the lower logical index. A thread with no
       ;; element holds magnitude -1.
       (define-kernel ,(kernel "IAMAX") ((x (ptr ,element :global)) (best-out (ptr f32 :global))
                                         (index-out (ptr u32 :global)) (n u32)
                                         (x-start i64) (x-inc i64))
         (let ((bests (shared-array f32 256))
               (indices (shared-array u32 256))
               (lid (local-id 0))
               (best -1.0)
               (found 4294967295))
           (declare (type f32 best) (type u32 found))
           (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
             (let ((a (abs ,(w 'x '(+ x-start (* (convert i64 i) x-inc))))))
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
                   (aref index-out (group-id 0)) (aref indices 0))))))))

(define-level1-kernels f32)
(define-level1-kernels f16)
(define-level1-kernels bf16)

;;; Products with float-float accumulation: each thread keeps a (hi, lo) pair whose sum is the
;;; running total to about 48 bits, using fma for the exact product error and two-sum for the adds.
(define-kernel %dsdot ((x (ptr f32 :global)) (y (ptr f32 :global))
                       (hi-out (ptr f32 :global)) (lo-out (ptr f32 :global)) (n u32)
                       (x-start i64) (x-inc i64) (y-start i64) (y-inc i64))
  (declare (precise-math))
  (let ((his (shared-array f32 256))
        (los (shared-array f32 256))
        (lid (local-id 0))
        (hi 0.0)
        (lo 0.0))
    (declare (type f32 hi lo))
    (for (i (global-id 0) (< i n) (incf i (grid-size 0)))
      (let* ((k (convert i64 i))
             (a (aref x (+ x-start (* k x-inc))))
             (b (aref y (+ y-start (* k y-inc))))
             (p (* a b))
             (e (fma a b (- p)))
             (sum (+ hi p))
             (bb (- sum hi)))
        (setf lo (+ lo e (+ (- hi (- sum bb)) (- p bb)))
              hi sum)))
    (setf (aref his lid) hi
          (aref los lid) lo)
    (barrier)
    (for (s (shr (group-size 0) 1) (> s 0) (setf s (shr s 1)))
      (when (< lid s)
        (let* ((h1 (aref his lid))
               (h2 (aref his (+ lid s)))
               (sum (+ h1 h2))
               (bb (- sum h1))
               (l (+ (aref los lid) (aref los (+ lid s)) (+ (- h1 (- sum bb)) (- h2 bb))))
               (h (+ sum l)))
          (setf (aref his lid) h
                (aref los lid) (- l (- h sum)))))
      (barrier))
    (when (= lid 0)
      (setf (aref hi-out (group-id 0)) (aref his 0)
            (aref lo-out (group-id 0)) (aref los 0)))))

;;; Routines

(defun array-element (name array)
  "The name of the element type of ARRAY, one of *ELEMENT-TYPES*."
  (unless (typep array 'gpu-array)
    (gpu-fail "~(~a~): expected a gpu-array, got ~s" name array))
  (or (find (gpu-array-element-type array) *element-types* :key #'parse-type)
      ;; TODO: f64 needs a backend that supports it (#55)
      (gpu-fail "~(~a~): expected an f32, f16 or bf16 array, got ~a" name
                (type-description (gpu-array-element-type array)))))

(defun level1-kernel (operation element)
  (gethash (list operation element) *level1-kernels*))

(defmacro define-level1-routine ((generic specific) (&rest parameters) keys array documentation
                                 &body body)
  "Define GENERIC, which takes the element type from ARRAY, and SPECIFIC, which needs f32. BODY
sees NAME, the routine called, and ELEMENT, the element type name."
  `(progn
     (defun ,generic (,@parameters ,@(when keys `(&key ,@keys)))
       ,documentation
       (let ((name ',generic) (element (array-element ',generic ,array)))
         ,@body))
     (defun ,specific (,@parameters ,@(when keys `(&key ,@keys)))
       ,(format nil "~a The arrays are f32." documentation)
       (let ((name ',specific) (element 'f32))
         ,@body))))

(define-level1-routine (axpy saxpy) (n alpha x incx y incy)
    ((x-offset 0) (y-offset 0)) x
  "Set Y to ALPHA*X+Y over N elements. Return Y."
  (let* ((xs (check-span name n x incx x-offset (list element)))
         (ys (check-span name n y incy y-offset (list element)))
         (alpha (real-scalar name "ALPHA" alpha))
         (device (same-device name xs ys)))
    (check-disjoint name ys xs)
    (unless (or (zerop n) (zerop alpha))
      (funcall (level1-kernel "AXPY" element)
               (span-array xs) (span-array ys) alpha n (span-start xs) incx (span-start ys) incy
               :grid n :device device))
    y))

(define-level1-routine (scal sscal) (n alpha x incx) ((x-offset 0)) x
  "Set X to ALPHA*X over N elements. Return X."
  (let* ((xs (check-span name n x incx x-offset (list element)))
         (alpha (real-scalar name "ALPHA" alpha)))
    (unless (zerop n)
      (funcall (level1-kernel "SCAL" element)
               (span-array xs) alpha n (span-start xs) incx :grid n :device (gpu-array-device x)))
    x))

(define-level1-routine (copy scopy) (n x incx y incy) ((x-offset 0) (y-offset 0)) x
  "Copy N elements of X to Y. Return Y."
  (let* ((xs (check-span name n x incx x-offset (list element)))
         (ys (check-span name n y incy y-offset (list element)))
         (device (same-device name xs ys)))
    (check-disjoint name ys xs)
    (unless (zerop n)
      (funcall (level1-kernel "COPY" element)
               (span-array xs) (span-array ys) n (span-start xs) incx (span-start ys) incy
               :grid n :device device))
    y))

(define-level1-routine (swap sswap) (n x incx y incy) ((x-offset 0) (y-offset 0)) x
  "Exchange N elements of X and Y. Return X and Y."
  (let* ((xs (check-span name n x incx x-offset (list element)))
         (ys (check-span name n y incy y-offset (list element)))
         (device (same-device name xs ys)))
    (check-disjoint name xs ys)
    (unless (zerop n)
      (funcall (level1-kernel "SWAP" element)
               (span-array xs) (span-array ys) n (span-start xs) incx (span-start ys) incy
               :grid n :device device))
    (values x y)))

(defun group-count (n)
  (min +max-groups+ (ceiling n +group-size+)))

;; TODO: one partials array per call, a reusable per-device buffer would avoid the allocation (#43)
(defmacro with-partials ((variable device count &optional (type 'f32)) &body body)
  `(with-gpu-array (,variable (make-gpu-array ,count :element-type ',type :device ,device))
     ,@body))

(defun sum-partials (partials)
  "The sum of the PARTIALS array, accumulated in double precision."
  (reduce #'+ (from-device partials) :key (lambda (v) (float v 1d0))))

(define-level1-routine (dot sdot) (n x incx y incy) ((x-offset 0) (y-offset 0)) x
  "The dot product of N elements of X and Y, accumulated in f32."
  (let* ((xs (check-span name n x incx x-offset (list element)))
         (ys (check-span name n y incy y-offset (list element)))
         (device (same-device name xs ys)))
    (if (zerop n)
        0f0
        (let ((groups (group-count n)))
          (with-partials (partials device groups)
            (funcall (level1-kernel "DOT" element)
                     (span-array xs) (span-array ys) partials n (span-start xs) incx (span-start ys) incy
                     :grid (* groups +group-size+) :group +group-size+ :device device)
            (float (sum-partials partials) 1f0))))))

(define-level1-routine (asum sasum) (n x incx) ((x-offset 0)) x
  "The sum of the absolute values of N elements of X, accumulated in f32."
  (let ((xs (check-span name n x incx x-offset (list element)))
        (device (gpu-array-device x)))
    (if (zerop n)
        0f0
        (let ((groups (group-count n)))
          (with-partials (partials device groups)
            (funcall (level1-kernel "ASUM" element)
                     (span-array xs) partials n (span-start xs) incx
                     :grid (* groups +group-size+) :group +group-size+ :device device)
            (float (sum-partials partials) 1f0))))))

(define-level1-routine (nrm2 snrm2) (n x incx) ((x-offset 0)) x
  "The Euclidean norm of N elements of X, accumulated in f32."
  (let ((xs (check-span name n x incx x-offset (list element)))
        (device (gpu-array-device x)))
    (if (zerop n)
        0f0
        (let ((groups (group-count n)))
          (with-partials (scales device groups)
            (with-partials (squares device groups)
              (funcall (level1-kernel "NRM2" element)
                       (span-array xs) scales squares n (span-start xs) incx
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

(define-level1-routine (iamax isamax) (n x incx) ((x-offset 0)) x
  "The zero-based index, in visiting order, of the first of N elements of X with the largest magnitude."
  (let ((xs (check-span name n x incx x-offset (list element)))
        (device (gpu-array-device x)))
    (if (zerop n)
        0
        (let ((groups (group-count n)))
          (with-partials (bests device groups)
            (with-partials (indices device groups u32)
              (funcall (level1-kernel "IAMAX" element)
                       (span-array xs) bests indices n (span-start xs) incx
                       :grid (* groups +group-size+) :group +group-size+ :device device)
              (let ((best -1f0) (found 0))
                (map nil (lambda (b i)
                           (when (or (> b best) (and (= b best) (< i found)))
                             (setf best b found i)))
                     (from-device bests) (from-device indices))
                (min found (1- n)))))))))

;;; Mixed precision: f32 inputs, a double-float result.

(defun dsdot (n x incx y incy &key (x-offset 0) (y-offset 0))
  "The dot product of N elements of f32 X and Y, accumulated to about 48 bits. A double-float."
  (let* ((xs (check-span 'dsdot n x incx x-offset))
         (ys (check-span 'dsdot n y incy y-offset))
         (device (same-device 'dsdot xs ys)))
    (if (zerop n)
        0d0
        (let ((groups (group-count n)))
          (with-partials (his device groups)
            (with-partials (los device groups)
              (%dsdot (span-array xs) (span-array ys) his los n (span-start xs) incx (span-start ys) incy
                      :grid (* groups +group-size+) :group +group-size+ :device device)
              (+ (sum-partials his) (sum-partials los))))))))

(defun sdsdot (n alpha x incx y incy &key (x-offset 0) (y-offset 0))
  "ALPHA plus the dsdot of N elements of f32 X and Y, rounded to single-float."
  (let ((alpha (real-scalar 'sdsdot "ALPHA" alpha)))
    (float (+ (float alpha 1d0) (dsdot n x incx y incy :x-offset x-offset :y-offset y-offset)) 1f0)))
