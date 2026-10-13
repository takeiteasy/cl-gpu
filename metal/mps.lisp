(in-package #:cl-gpu/metal)

;;; gemm and gemv through MetalPerformanceShaders. Layouts it cannot address take the DSL kernels.

(defvar *use-mps* t
  "When true, GEMM and GEMV on a Metal device use MetalPerformanceShaders where the layouts allow.")

(defun mps-supported-p (device)
  (when (eq (device-mps-support device) :unknown)
    (setf (device-mps-support device) (/= 0 (%mps-supported (live-handle device)))))
  (device-mps-support device))

;;; A spec is (array offset rows columns row-stride column-stride) in elements.

(defun spec (list rows columns)
  (destructuring-bind (array offset row-stride column-stride) list
    (list array offset rows columns row-stride column-stride)))

(defun transpose-spec (spec)
  (destructuring-bind (array offset rows columns row-stride column-stride) spec
    (list array offset columns rows column-stride row-stride)))

(defun mps-matrix (spec)
  "(transposed stored-rows stored-columns row-elements) for the row-major matrix MPS reads, or NIL.
A column-major spec is the transpose of a row-major matrix. Strides must be positive."
  (destructuring-bind (array offset rows columns row-stride column-stride) spec
    (let ((matrix (cond ((and (or (= columns 1) (= column-stride 1))
                              (or (= rows 1) (<= columns row-stride)))
                         (list nil rows columns (if (= rows 1) columns row-stride)))
                        ((and (or (= rows 1) (= row-stride 1))
                              (or (= columns 1) (<= rows column-stride)))
                         (list t columns rows (if (= columns 1) rows column-stride))))))
      (when (and matrix
                 (<= (+ offset (* (second matrix) (fourth matrix))) (gpu:gpu-array-length array)))
        matrix))))

(defun row-major-p (spec)
  (let ((matrix (mps-matrix spec)))
    (and matrix (not (first matrix)))))

(defun fill-matrix-shape (shapes index spec matrix)
  (destructuring-bind (stored-rows stored-columns row-elements) (rest matrix)
    (loop for value in (list (* 4 (second spec)) stored-rows stored-columns (* 4 row-elements))
          for slot from (* 4 index)
          do (setf (cffi:mem-aref shapes :unsigned-long slot) value))))

(defun buffer-handle (spec)
  (live-handle (gpu::live-buffer (first spec))))

(defun gemm-plan (m n k a b c)
  "(left right result m n) as specs MPS can address for C = A*B + C, or NIL.
MPS writes a row-major result; a column-major C is computed as its transpose."
  (let ((a (spec a m k))
        (b (spec b k n))
        (c (spec c m n)))
    (flet ((plan (left right result m n)
             (when (and (mps-matrix left) (mps-matrix right) (row-major-p result))
               (list left right result m n))))
      (or (plan a b c m n)
          (plan (transpose-spec b) (transpose-spec a) (transpose-spec c) n m)))))

(defmethod gpu:device-gemm ((device metal-device) m n k alpha a b beta c)
  (let ((plan (and *use-mps* (mps-supported-p device) (gemm-plan m n k a b c))))
    (if plan
        (destructuring-bind (left right result m n) plan
          (cffi:with-foreign-objects ((buffers :pointer 3) (shapes :unsigned-long 12))
            (loop for spec in (list left right result)
                  for index from 0
                  do (setf (cffi:mem-aref buffers :pointer index) (buffer-handle spec))
                     (fill-matrix-shape shapes index spec (mps-matrix spec)))
            (multiple-value-bind (status message)
                (with-error-slot (slot)
                  (%mps-gemm (device-queue device)
                             (if (first (mps-matrix left)) 1 0) (if (first (mps-matrix right)) 1 0)
                             m n k alpha beta buffers shapes slot))
              (unless (zerop status)
                (fail "~a" (or message "gemm failed"))))))
        (call-next-method))))

(defmethod gpu:device-gemv ((device metal-device) m n alpha a x beta y)
  (let* ((a-spec (spec a m n))
         (matrix (and *use-mps* (mps-supported-p device) (= 1 (third x)) (= 1 (third y))
                      (mps-matrix a-spec))))
    (if matrix
        (let ((transposed (first matrix)))
          (cffi:with-foreign-objects ((buffers :pointer 3) (shapes :unsigned-long 8))
            (setf (cffi:mem-aref buffers :pointer 0) (buffer-handle a-spec)
                  (cffi:mem-aref buffers :pointer 1) (live-handle (gpu::live-buffer (first x)))
                  (cffi:mem-aref buffers :pointer 2) (live-handle (gpu::live-buffer (first y))))
            (fill-matrix-shape shapes 0 a-spec matrix)
            (loop for value in (list (* 4 (second x)) n (* 4 (second y)) m)
                  for slot from 4
                  do (setf (cffi:mem-aref shapes :unsigned-long slot) value))
            (multiple-value-bind (status message)
                (with-error-slot (slot)
                  (%mps-gemv (device-queue device) (if transposed 1 0) m n
                             alpha beta buffers shapes slot))
              (unless (zerop status)
                (fail "~a" (or message "gemv failed"))))))
        (call-next-method))))
