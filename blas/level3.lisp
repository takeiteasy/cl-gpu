(in-package #:cl-gpu/blas)

(defconstant +tile+ 16)

;;; One thread per element of C, over 16 by 16 tiles of A and B staged in shared memory. The grid
;;; covers whole tiles, so every thread of a group reaches every barrier and loads zero outside
;;; the matrices. Grid z selects the product of a batch.
;;; TODO: several outputs per thread (register blocking) (#44)
(define-kernel %sgemm ((a (ptr f32 :global)) (b (ptr f32 :global)) (c (ptr f32 :global))
                       (alpha f32) (beta f32) (m u32) (n u32) (k u32)
                       (a-off i64) (a-rs i64) (a-cs i64)
                       (b-off i64) (b-rs i64) (b-cs i64)
                       (c-off i64) (c-rs i64) (c-cs i64)
                       (a-bs i64) (b-bs i64) (c-bs i64))
  (let* ((a-tile (shared-array f32 256))
         (b-tile (shared-array f32 256))
         (tx (local-id 0))
         (ty (local-id 1))
         (col (global-id 0))
         (row (global-id 1))
         (batch (convert i64 (global-id 2)))
         (a-base (+ a-off (* batch a-bs)))
         (b-base (+ b-off (* batch b-bs)))
         (c-base (+ c-off (* batch c-bs)))
         (sum 0.0))
    (declare (type f32 sum))
    (for (base 0 (< base k) (incf base 16))
      (let ((ak (+ base tx))
            (bk (+ base ty)))
        (setf (aref a-tile (+ (* ty 16) tx))
              (if (and (< row m) (< ak k))
                  (aref a (+ a-base (* (convert i64 row) a-rs) (* (convert i64 ak) a-cs)))
                  0.0)
              (aref b-tile (+ (* ty 16) tx))
              (if (and (< bk k) (< col n))
                  (aref b (+ b-base (* (convert i64 bk) b-rs) (* (convert i64 col) b-cs)))
                  0.0)))
      (barrier)
      (for (q 0 (< q 16) (incf q))
        (declare (type u32 q))
        (incf sum (* (aref a-tile (+ (* ty 16) q)) (aref b-tile (+ (* q 16) tx)))))
      (barrier))
    (when (and (< row m) (< col n))
      (let ((at (+ c-base (* (convert i64 row) c-rs) (* (convert i64 col) c-cs))))
        (setf (aref c at)
              (if (= beta 0.0)
                  (* alpha sum)
                  (+ (* alpha sum) (* beta (aref c at)))))))))

(define-kernel %sgescal ((c (ptr f32 :global)) (beta f32) (m u32) (n u32)
                         (c-off i64) (c-rs i64) (c-cs i64) (c-bs i64))
  (let ((col (global-id 0))
        (row (global-id 1))
        (batch (convert i64 (global-id 2))))
    (when (and (< row m) (< col n))
      (let ((at (+ c-off (* batch c-bs) (* (convert i64 row) c-rs) (* (convert i64 col) c-cs))))
        (setf (aref c at)
              (if (= beta 0.0) 0.0 (* beta (aref c at))))))))

(defun round-up-to-tiles (count)
  (* +tile+ (ceiling count +tile+)))

(defmethod device-gemm (device m n k alpha a b beta c count)
  (destructuring-bind (a-array a-off a-rs a-cs a-bs) a
    (destructuring-bind (b-array b-off b-rs b-cs b-bs) b
      (destructuring-bind (c-array c-off c-rs c-cs c-bs) c
        (%sgemm a-array b-array c-array alpha beta m n k a-off a-rs a-cs b-off b-rs b-cs
                c-off c-rs c-cs a-bs b-bs c-bs
                :grid (list (round-up-to-tiles n) (round-up-to-tiles m) count)
                :group (list +tile+ +tile+ 1) :device device)))))

(defun scale-operand (device operand beta)
  "Set the elements of every product of OPERAND to BETA times themselves."
  (unless (= beta 1f0)
    (let ((m (operand-rows operand))
          (n (operand-cols operand)))
      (%sgescal (operand-array operand) beta m n (operand-offset operand)
                (operand-row-stride operand) (operand-column-stride operand)
                (operand-batch-stride operand)
                :grid (list n m (operand-batch-count operand)) :device device))))

(defconstant +max-batch+ 1073741823)

(defun gemm-products (name transa transb alpha a b beta c count a-stride b-stride c-stride)
  "Set COUNT products C[i] to ALPHA*op(A[i])*op(B[i]) + BETA*C[i], the views STRIDE elements apart."
  (let* ((oa (view-operand name a transa count a-stride))
         (ob (view-operand name b transb count b-stride))
         (oc (view-operand name c :no-transpose count c-stride))
         (m (operand-rows oc))
         (n (operand-cols oc))
         (k (operand-cols oa))
         (alpha (real-scalar name "ALPHA" alpha))
         (beta (real-scalar name "BETA" beta))
         (device (same-device-arrays name (operand-array oa) (operand-array ob)
                                     (operand-array oc))))
    (unless (and (= m (operand-rows oa)) (= k (operand-rows ob)) (= n (operand-cols ob)))
      (gpu-fail "~(~a~): ~dx~d times ~dx~d does not fill ~dx~d" name
                (operand-rows oa) k (operand-rows ob) (operand-cols ob) m n))
    (dolist (operand (list oa ob oc))
      (check-operand-bounds name operand))
    (check-unique-output name oc)
    (check-apart name (operand-region oc) (operand-region oa) (operand-region ob))
    (cond ((or (zerop m) (zerop n) (zerop count)))
          ((or (zerop alpha) (zerop k))
           (scale-operand device oc beta))
          (t
           (device-gemm device m n k alpha (operand-batch-list oa) (operand-batch-list ob) beta
                        (operand-batch-list oc) count)))
    c))

(defun sgemm (transa transb alpha a b beta c)
  "Set C to ALPHA*op(A)*op(B) + BETA*C, where op is the TRANSA or TRANSB flag. Return C."
  (gemm-products 'sgemm transa transb alpha a b beta c 1 0 0 0))

(defun sgemm-batch-strided (transa transb alpha a b beta c &key batch-count a-stride b-stride c-stride)
  "Set the product C[i] to ALPHA*op(A[i])*op(B[i]) + BETA*C[i] for each of BATCH-COUNT products.
Product i starts at each view's offset plus i times its stride: A-STRIDE, B-STRIDE or C-STRIDE
elements, signed. A zero input stride repeats one matrix. Return C."
  (unless (typep batch-count `(integer 0 ,+max-batch+))
    (gpu-fail "sgemm-batch-strided: :batch-count must be an integer from 0 to ~d, got ~s"
              +max-batch+ batch-count))
  (loop for (label stride) in `((":a-stride" ,a-stride) (":b-stride" ,b-stride)
                                (":c-stride" ,c-stride))
        unless (integerp stride)
          do (gpu-fail "sgemm-batch-strided: ~a must be an integer, got ~s" label stride))
  (gemm-products 'sgemm-batch-strided transa transb alpha a b beta c batch-count
                 a-stride b-stride c-stride))
