(in-package #:cl-gpu/blas)

(defconstant +tile+ 16)

;;; One thread per element of C, over 16 by 16 tiles of A and B staged in shared memory. The grid
;;; covers whole tiles, so every thread of a group reaches every barrier and loads zero outside
;;; the matrices.
;;; TODO: several outputs per thread (register blocking) (#44)
(define-kernel %sgemm ((a (ptr f32 :global)) (b (ptr f32 :global)) (c (ptr f32 :global))
                       (alpha f32) (beta f32) (m u32) (n u32) (k u32)
                       (a-off i64) (a-rs i64) (a-cs i64)
                       (b-off i64) (b-rs i64) (b-cs i64)
                       (c-off i64) (c-rs i64) (c-cs i64))
  (let ((a-tile (shared-array f32 256))
        (b-tile (shared-array f32 256))
        (tx (local-id 0))
        (ty (local-id 1))
        (col (global-id 0))
        (row (global-id 1))
        (sum 0.0))
    (declare (type f32 sum))
    (for (base 0 (< base k) (incf base 16))
      (let ((ak (+ base tx))
            (bk (+ base ty)))
        (setf (aref a-tile (+ (* ty 16) tx))
              (if (and (< row m) (< ak k))
                  (aref a (+ a-off (* (convert i64 row) a-rs) (* (convert i64 ak) a-cs)))
                  0.0)
              (aref b-tile (+ (* ty 16) tx))
              (if (and (< bk k) (< col n))
                  (aref b (+ b-off (* (convert i64 bk) b-rs) (* (convert i64 col) b-cs)))
                  0.0)))
      (barrier)
      (for (q 0 (< q 16) (incf q))
        (declare (type u32 q))
        (incf sum (* (aref a-tile (+ (* ty 16) q)) (aref b-tile (+ (* q 16) tx)))))
      (barrier))
    (when (and (< row m) (< col n))
      (let ((at (+ c-off (* (convert i64 row) c-rs) (* (convert i64 col) c-cs))))
        (setf (aref c at)
              (if (= beta 0.0)
                  (* alpha sum)
                  (+ (* alpha sum) (* beta (aref c at)))))))))

(define-kernel %sgescal ((c (ptr f32 :global)) (beta f32) (m u32) (n u32)
                         (c-off i64) (c-rs i64) (c-cs i64))
  (let ((col (global-id 0))
        (row (global-id 1)))
    (when (and (< row m) (< col n))
      (let ((at (+ c-off (* (convert i64 row) c-rs) (* (convert i64 col) c-cs))))
        (setf (aref c at)
              (if (= beta 0.0) 0.0 (* beta (aref c at))))))))

(defun round-up-to-tiles (count)
  (* +tile+ (ceiling count +tile+)))

(defmethod device-gemm (device m n k alpha a b beta c)
  (destructuring-bind (a-array a-off a-rs a-cs) a
    (destructuring-bind (b-array b-off b-rs b-cs) b
      (destructuring-bind (c-array c-off c-rs c-cs) c
        (%sgemm a-array b-array c-array alpha beta m n k a-off a-rs a-cs b-off b-rs b-cs
                c-off c-rs c-cs
                :grid (list (round-up-to-tiles n) (round-up-to-tiles m))
                :group (list +tile+ +tile+) :device device)))))

(defun scale-operand (device operand beta)
  "Set the elements of OPERAND to BETA times themselves."
  (unless (= beta 1f0)
    (let ((m (operand-rows operand))
          (n (operand-cols operand)))
      (%sgescal (operand-array operand) beta m n (operand-offset operand)
                (operand-row-stride operand) (operand-column-stride operand)
                :grid (list n m) :device device))))

(defun sgemm (transa transb alpha a b beta c)
  "Set C to ALPHA*op(A)*op(B) + BETA*C, where op is the TRANSA or TRANSB flag. Return C."
  (let* ((oa (view-operand 'sgemm a transa))
         (ob (view-operand 'sgemm b transb))
         (oc (view-operand 'sgemm c :no-transpose))
         (m (operand-rows oc))
         (n (operand-cols oc))
         (k (operand-cols oa))
         (alpha (real-scalar 'sgemm "ALPHA" alpha))
         (beta (real-scalar 'sgemm "BETA" beta))
         (device (same-device-arrays 'sgemm (operand-array oa) (operand-array ob)
                                     (operand-array oc))))
    (unless (and (= m (operand-rows oa)) (= k (operand-rows ob)) (= n (operand-cols ob)))
      (gpu-fail "sgemm: ~dx~d times ~dx~d does not fill ~dx~d"
                (operand-rows oa) k (operand-rows ob) (operand-cols ob) m n))
    (check-unique-output 'sgemm oc)
    (check-apart 'sgemm (operand-region oc) (operand-region oa) (operand-region ob))
    (cond ((or (zerop m) (zerop n)))
          ((or (zerop alpha) (zerop k))
           (scale-operand device oc beta))
          (t
           (device-gemm device m n k alpha (operand-list oa) (operand-list ob) beta
                        (operand-list oc))))
    c))
