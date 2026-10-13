(in-package #:cl-gpu/blas)

(defstruct (span (:constructor make-span (array low increment count)))
  array low increment count)

(defun check-span (name n array increment offset)
  "The SPAN of N elements of ARRAY at INCREMENT from OFFSET."
  (unless (typep n '(integer 0))
    (gpu-fail "~(~a~): N must be a non-negative integer, got ~s" name n))
  (unless (and (integerp increment) (/= increment 0))
    (gpu-fail "~(~a~): the increment must be a non-zero integer, got ~s" name increment))
  (unless (typep offset '(integer 0))
    (gpu-fail "~(~a~): the offset must be a non-negative integer, got ~s" name offset))
  (unless (typep array 'gpu-array)
    (gpu-fail "~(~a~): expected a gpu-array, got ~s" name array))
  (unless (eq (gpu-array-element-type array) (parse-type 'f32))
    (gpu-fail "~(~a~): expected an f32 array, got ~a" name
              (type-description (gpu-array-element-type array))))
  (cl-gpu::live-buffer array)
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

(defun same-device (name &rest spans)
  (let ((device (gpu-array-device (span-array (first spans)))))
    (unless (every (lambda (span) (eq (gpu-array-device (span-array span)) device)) spans)
      (gpu-fail "~(~a~): the arrays are on different devices" name))
    device))
