(in-package #:cl-gpu/blas)

;;; Element (row column) is at offset + row*row-stride + column*column-stride.
(defstruct (matrix-view (:constructor %make-matrix-view))
  (data nil :read-only t)
  (rows 0 :read-only t)
  (cols 0 :read-only t)
  (layout :row-major :read-only t)
  (leading-dimension 1 :read-only t)
  (row-stride 1 :read-only t)
  (column-stride 1 :read-only t)
  (offset 0 :read-only t))

(defun stride-layout (rows cols row-stride column-stride)
  "The layout and leading dimension that regular strides describe, else :strided and 1."
  (cond ((and (or (= cols 1) (= column-stride 1))
              (or (= rows 1) (<= (max 1 cols) row-stride)))
         (values :row-major (if (= rows 1) (max 1 cols) row-stride)))
        ((and (or (= rows 1) (= row-stride 1))
              (or (= cols 1) (<= (max 1 rows) column-stride)))
         (values :column-major (if (= cols 1) (max 1 rows) column-stride)))
        (t (values :strided 1))))

(defun extent-bounds (offset extents strides)
  "The lowest and highest index offset + sum(i*stride) takes for i below each extent."
  (let ((low offset) (high offset))
    (loop for extent in extents
          for stride in strides
          for delta = (* (max 0 (1- extent)) stride)
          do (incf low (min 0 delta))
             (incf high (max 0 delta)))
    (values low high)))

(defun make-matrix-view (data rows cols &key (layout :row-major) leading-dimension (offset 0)
                                          (row-stride nil row-stride-p)
                                          (column-stride nil column-stride-p))
  "A view of ROWS by COLS elements of the f32 array DATA. It shares DATA's storage.
LAYOUT and LEADING-DIMENSION give regular storage; ROW-STRIDE and COLUMN-STRIDE, both or neither,
give signed strides instead."
  (check-f32-array 'make-matrix-view data)
  (unless (and (typep rows '(integer 0)) (typep cols '(integer 0)))
    (gpu-fail "make-matrix-view: the dimensions must be non-negative integers, got ~s by ~s" rows cols))
  (unless (member layout '(:row-major :column-major))
    (gpu-fail "make-matrix-view: the layout is :row-major or :column-major, got ~s" layout))
  (unless (typep offset '(integer 0))
    (gpu-fail "make-matrix-view: the offset must be a non-negative integer, got ~s" offset))
  (unless (eq row-stride-p column-stride-p)
    (gpu-fail "make-matrix-view: give both strides or neither"))
  (when (and row-stride-p leading-dimension)
    (gpu-fail "make-matrix-view: strides exclude a leading dimension"))
  (let* ((minor (max 1 (if (eq layout :row-major) cols rows)))
         (ld (or leading-dimension minor))
         (rs (if row-stride-p row-stride (if (eq layout :row-major) ld 1)))
         (cs (if row-stride-p column-stride (if (eq layout :row-major) 1 ld))))
    (unless (and (integerp rs) (integerp cs) (integerp ld) (or row-stride-p (<= minor ld)))
      (gpu-fail "make-matrix-view: invalid strides or leading dimension"))
    (when row-stride-p
      (multiple-value-setq (layout ld) (stride-layout rows cols rs cs)))
    (let ((length (gpu-array-length data)))
      (if (or (zerop rows) (zerop cols))
          (unless (<= offset length)
            (gpu-fail "make-matrix-view: the offset ~d exceeds an array of ~d elements" offset length))
          (multiple-value-bind (low high) (extent-bounds offset (list rows cols) (list rs cs))
            (unless (and (<= 0 low) (< high length))
              (gpu-fail "make-matrix-view: the view exceeds an array of ~d elements" length)))))
    (%make-matrix-view :data data :rows rows :cols cols :layout layout :leading-dimension ld
                       :row-stride rs :column-stride cs :offset offset)))

(defun matrix-subview (view row column rows cols)
  "The ROWS by COLS part of VIEW from (ROW COLUMN). It shares VIEW's storage and strides."
  (unless (and (matrix-view-p view)
               (typep row '(integer 0)) (typep column '(integer 0))
               (typep rows '(integer 0)) (typep cols '(integer 0))
               (<= (+ row rows) (matrix-view-rows view))
               (<= (+ column cols) (matrix-view-cols view)))
    (gpu-fail "matrix-subview: invalid subview"))
  (let ((length (gpu-array-length (matrix-view-data view))))
    (%make-matrix-view
     :data (matrix-view-data view) :rows rows :cols cols
     :layout (matrix-view-layout view)
     :leading-dimension (matrix-view-leading-dimension view)
     :row-stride (matrix-view-row-stride view)
     :column-stride (matrix-view-column-stride view)
     :offset (max 0 (min length (+ (matrix-view-offset view)
                                   (* row (matrix-view-row-stride view))
                                   (* column (matrix-view-column-stride view))))))))
