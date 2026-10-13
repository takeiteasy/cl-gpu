(in-package #:cl-gpu/blas)

(defstruct (span (:constructor make-span (array low increment count)))
  array low increment count)

(defun check-f32-array (name array)
  (unless (typep array 'gpu-array)
    (gpu-fail "~(~a~): expected a gpu-array, got ~s" name array))
  (unless (eq (gpu-array-element-type array) (parse-type 'f32))
    (gpu-fail "~(~a~): expected an f32 array, got ~a" name
              (type-description (gpu-array-element-type array))))
  (cl-gpu::live-buffer array)
  array)

(defun check-span (name n array increment offset)
  "The SPAN of N elements of ARRAY at INCREMENT from OFFSET."
  (unless (typep n '(integer 0))
    (gpu-fail "~(~a~): N must be a non-negative integer, got ~s" name n))
  (unless (and (integerp increment) (/= increment 0))
    (gpu-fail "~(~a~): the increment must be a non-zero integer, got ~s" name increment))
  (unless (typep offset '(integer 0))
    (gpu-fail "~(~a~): the offset must be a non-negative integer, got ~s" name offset))
  (check-f32-array name array)
  (let ((last (if (zerop n) offset (+ offset (* (1- n) (abs increment))))))
    (unless (if (zerop n)
                (<= offset (gpu-array-length array))
                (< last (gpu-array-length array)))
      (gpu-fail "~(~a~): the span exceeds an array of ~d elements" name (gpu-array-length array))))
  (make-span array offset increment n))

(defun span-start (span)
  "The first index SPAN visits."
  (if (minusp (span-increment span))
      (+ (span-low span) (* (1- (span-count span)) (- (span-increment span))))
      (span-low span)))

(defun span-high (span)
  (+ (span-low span) (* (max 0 (1- (span-count span))) (abs (span-increment span)))))

(defun spans-overlap-p (a b)
  "True when A and B may visit one element. Exact for spans of one stride, conservative otherwise."
  (and (eq (span-array a) (span-array b))
       (plusp (span-count a))
       (plusp (span-count b))
       (<= (span-low a) (span-high b))
       (<= (span-low b) (span-high a))
       (zerop (mod (- (span-low b) (span-low a))
                   (gcd (abs (span-increment a)) (abs (span-increment b)))))))

(defun check-disjoint (name a b)
  "Signal unless A and B are one span or visit different elements."
  (when (and (spans-overlap-p a b)
             (not (and (= (span-start a) (span-start b))
                       (= (span-increment a) (span-increment b))
                       (= (span-count a) (span-count b)))))
    (gpu-fail "~(~a~): the spans overlap in one array" name)))

(defun same-device-arrays (name &rest arrays)
  (let ((device (gpu-array-device (first arrays))))
    (unless (every (lambda (array) (eq (gpu-array-device array) device)) arrays)
      (gpu-fail "~(~a~): the arrays are on different devices" name))
    device))

(defun same-device (name &rest spans)
  (apply #'same-device-arrays name (mapcar #'span-array spans)))

;;; Matrix operands: a view after its transpose flag, as (rows, columns, strides) over an array.

(defstruct (operand (:constructor make-operand (array offset row-stride column-stride rows cols)))
  array offset row-stride column-stride rows cols)

(defun operand-list (operand)
  "OPERAND as the (array offset row-stride column-stride) of the device protocol."
  (list (operand-array operand) (operand-offset operand)
        (operand-row-stride operand) (operand-column-stride operand)))

(defun transposed-p (name flag)
  (case flag
    (:no-transpose nil)
    ((:transpose :conjugate-transpose) t)
    (t (gpu-fail "~(~a~): the transpose flag is :no-transpose, :transpose or :conjugate-transpose, got ~s"
                 name flag))))

(defun view-operand (name view flag)
  "The operand for VIEW under the transpose FLAG."
  (let ((transposed (transposed-p name flag)))
    (unless (matrix-view-p view)
      (gpu-fail "~(~a~): expected a matrix view, got ~s" name view))
    (check-f32-array name (matrix-view-data view))
    (let ((data (matrix-view-data view))
          (offset (matrix-view-offset view))
          (rs (matrix-view-row-stride view))
          (cs (matrix-view-column-stride view))
          (rows (matrix-view-rows view))
          (cols (matrix-view-cols view)))
      (if transposed
          (make-operand data offset cs rs cols rows)
          (make-operand data offset rs cs rows cols)))))

(defun real-scalar (name label value)
  (unless (realp value)
    (gpu-fail "~(~a~): ~a must be a real number, got ~s" name label value))
  (float value 1f0))

;;; Regions: (array lowest highest) of the elements an operand or span visits, NIL when empty.

(defun operand-region (operand)
  (when (and (plusp (operand-rows operand)) (plusp (operand-cols operand)))
    (multiple-value-bind (low high)
        (extent-bounds (operand-offset operand)
                       (list (operand-rows operand) (operand-cols operand))
                       (list (operand-row-stride operand) (operand-column-stride operand)))
      (list (operand-array operand) low high))))

(defun span-region (span)
  (when (plusp (span-count span))
    (list (span-array span) (span-low span) (span-high span))))

(defun regions-overlap-p (a b)
  (and a b
       (eq (first a) (first b))
       (<= (second a) (third b))
       (<= (second b) (third a))))

(defun check-apart (name output &rest inputs)
  "Signal when the region OUTPUT shares a range of an array with any of INPUTS."
  (dolist (input inputs)
    (when (regions-overlap-p output input)
      (gpu-fail "~(~a~): the output overlaps an input in one array" name))))

(defun check-unique-output (name operand)
  "Signal unless every element of OPERAND is addressed once, which writing it in parallel needs.
Conservative: it refuses strides that interleave without sharing an element."
  (let ((axes (sort (remove-if (lambda (axis) (<= (car axis) 1))
                               (list (cons (operand-rows operand) (abs (operand-row-stride operand)))
                                     (cons (operand-cols operand) (abs (operand-column-stride operand)))))
                    #'< :key #'cdr)))
    (unless (or (zerop (operand-rows operand))
                (zerop (operand-cols operand))
                (null axes)
                (and (plusp (cdr (first axes)))
                     (or (null (rest axes))
                         (>= (cdr (second axes)) (* (car (first axes)) (cdr (first axes)))))))
      (gpu-fail "~(~a~): the output addresses an element more than once" name))))
