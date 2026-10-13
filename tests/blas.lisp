(defpackage #:cl-gpu/blas/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:gpu #:cl-gpu) (#:blas #:cl-gpu/blas) (#:simd #:trivial-simd/blas))
  (:export #:run-tests))

(in-package #:cl-gpu/blas/tests)

(def-suite :cl-gpu/blas)
(in-suite :cl-gpu/blas)

(defun run-tests ()
  (run! :cl-gpu/blas))

;;; Oracle: trivial-simd/blas on Lisp vectors against the same data on the GPU

(defvar *seed* 19)

(defun next-unit ()
  "A deterministic number in [-1, 1)."
  (setf *seed* (mod (+ (* *seed* 1103515245) 12345) 2147483648))
  (- (* 2 (/ *seed* 2147483648f0)) 1))

(defun random-vector (length &optional (scale 1f0))
  (let ((vector (make-array length :element-type 'single-float)))
    (dotimes (index length vector)
      (setf (aref vector index) (* scale (next-unit))))))

(defun span-length (n inc offset)
  (+ offset 1 (* (max 0 (1- n)) (abs inc)) 3))

(defun close-p (a b)
  (<= (abs (- a b)) (* 1f-4 (max 1 (abs a) (abs b)))))

(defun all-close-p (a b)
  (every #'close-p a b))

(defparameter *spans*
  '((0 1 0) (1 1 0) (5 1 0) (255 1 0) (256 1 0) (257 1 3) (1000 3 2) (300 -1 0) (257 -2 5)
    (70000 1 0) (70000 -1 1))
  "(n increment offset) of x; y uses increment 2 from offset 1.")

(defmacro do-spans ((n inc offset) &body body)
  `(loop for (,n ,inc ,offset) in *spans*
         do (let ((message (format nil "n ~d inc ~d offset ~d" ,n ,inc ,offset)))
              (declare (ignorable message))
              ,@body)))

(defmacro with-pair ((x gx y gy n inc offset) &body body)
  "Bind X and Y to random vectors and GX and GY to device copies."
  `(let ((,x (random-vector (span-length ,n ,inc ,offset)))
         (,y (random-vector (span-length ,n 2 1))))
     (gpu:with-gpu-array (,gx (gpu:to-device ,x))
       (gpu:with-gpu-array (,gy (gpu:to-device ,y))
         ,@body))))

(test saxpy-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (simd:saxpy n 2.5f0 x inc y 2 :x-offset offset :y-offset 1)
      (is (eq gy (blas:saxpy n 2.5f0 gx inc gy 2 :x-offset offset :y-offset 1)))
      (is (all-close-p y (gpu:from-device gy)) message))))

(test sscal-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (simd:sscal n -1.5f0 x inc :x-offset offset)
      (is (eq gx (blas:sscal n -1.5f0 gx inc :x-offset offset)))
      (is (all-close-p x (gpu:from-device gx)) message))))

(test sscal-by-zero-clears
  (gpu:with-gpu-array (gx (gpu:to-device #(1.0 2.0 3.0 4.0)))
    (blas:sscal 3 0f0 gx 1)
    (is (equalp #(0.0 0.0 0.0 4.0) (gpu:from-device gx)))))

(test scopy-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (simd:scopy n x inc y 2 :x-offset offset :y-offset 1)
      (blas:scopy n gx inc gy 2 :x-offset offset :y-offset 1)
      (is (equalp y (gpu:from-device gy)) message))))

(test sswap-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (simd:sswap n x inc y 2 :x-offset offset :y-offset 1)
      (blas:sswap n gx inc gy 2 :x-offset offset :y-offset 1)
      (is (equalp x (gpu:from-device gx)) message)
      (is (equalp y (gpu:from-device gy)) message))))

(test sdot-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (is (close-p (simd:sdot n x inc y 2 :x-offset offset :y-offset 1)
                   (blas:sdot n gx inc gy 2 :x-offset offset :y-offset 1))
          message))))

(test sasum-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (is (close-p (simd:sasum n x inc :x-offset offset) (blas:sasum n gx inc :x-offset offset))
          message))))

(test snrm2-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (is (close-p (simd:snrm2 n x inc :x-offset offset) (blas:snrm2 n gx inc :x-offset offset))
          message))))

(test snrm2-survives-extreme-magnitudes
  (dolist (scale '(1f20 1f-20))
    (let ((x (random-vector 1000 scale)))
      (gpu:with-gpu-array (gx (gpu:to-device x))
        (let ((norm (blas:snrm2 1000 gx 1)))
          (is (< 0 norm most-positive-single-float))
          (is (close-p (/ (simd:snrm2 1000 x 1) scale) (/ norm scale))))))))

(test isamax-matches-the-oracle
  (do-spans (n inc offset)
    (with-pair (x gx y gy n inc offset)
      (is (= (simd:isamax n x inc :x-offset offset) (blas:isamax n gx inc :x-offset offset))
          message))))

(test isamax-takes-the-first-of-equal-magnitudes
  (let ((x (make-array 1000 :element-type 'single-float :initial-element 1f0)))
    (setf (aref x 700) -3f0 (aref x 300) 3f0 (aref x 900) 3f0)
    (gpu:with-gpu-array (gx (gpu:to-device x))
      (is (= 300 (blas:isamax 1000 gx 1)))
      (is (= (simd:isamax 1000 x -1) (blas:isamax 1000 gx -1))))))

;;; Validation, without a GPU

(defclass mock-device ()
  ((launches :initform 0 :accessor launches)))

(defmethod gpu:device-target ((device mock-device)) :metal)
(defmethod gpu:device-allocate ((device mock-device) bytes) (make-array bytes :initial-element 0))
(defmethod gpu:device-free ((device mock-device) buffer))
(defmethod gpu:device-store ((device mock-device) buffer sequence element-type)
  (replace buffer sequence))
(defmethod gpu:device-load ((device mock-device) buffer sequence element-type)
  (replace sequence buffer))
(defmethod gpu:device-prepare ((device mock-device) source entry-name) entry-name)
(defmethod gpu:device-launch ((device mock-device) prepared arguments grid group wait)
  (incf (launches device))
  :command)

(defmacro with-mock-arrays ((device &rest arrays) &body body)
  "Bind each of ARRAYS to a (name length) array of f32 on a mock DEVICE."
  `(let* ((,device (make-instance 'mock-device))
          ,@(loop for (name length) in arrays
                  collect `(,name (gpu:make-gpu-array ,length :device ,device))))
     ,@body))

(test bad-arguments-signal-gpu-error
  (with-mock-arrays (device (x 10) (y 10))
    (let ((ints (gpu:make-gpu-array 10 :device device :element-type 'gpu:i32))
          (other (gpu:make-gpu-array 10 :device (make-instance 'mock-device))))
      (signals gpu:gpu-error (blas:saxpy -1 1f0 x 1 y 1))
      (signals gpu:gpu-error (blas:saxpy 1.5 1f0 x 1 y 1))
      (signals gpu:gpu-error (blas:saxpy 5 1f0 x 0 y 1))
      (signals gpu:gpu-error (blas:saxpy 5 "1" x 1 y 1))
      (signals gpu:gpu-error (blas:saxpy 5 1f0 x 1 y 1 :x-offset -1))
      (signals gpu:gpu-error (blas:saxpy 5 1f0 #(1 2 3) 1 y 1))
      (signals gpu:gpu-error (blas:saxpy 5 1f0 ints 1 y 1))
      (signals gpu:gpu-error (blas:saxpy 5 1f0 x 1 other 1))
      (signals gpu:gpu-error (blas:sdot 5 x 1 other 1))
      (gpu:free y)
      (signals gpu:gpu-error (blas:sasum 5 y 1))
      (is (zerop (launches device))))))

(test spans-must-fit-their-arrays
  (with-mock-arrays (device (x 10) (y 10))
    (signals gpu:gpu-error (blas:sscal 11 1f0 x 1))
    (signals gpu:gpu-error (blas:sscal 6 1f0 x 2))
    (signals gpu:gpu-error (blas:sscal 5 1f0 x -2 :x-offset 3))
    (signals gpu:gpu-error (blas:scopy 1 x 1 y 1 :y-offset 10))
    (finishes (blas:sscal 5 1f0 x 2))
    (finishes (blas:sscal 10 1f0 x -1))
    (finishes (blas:sscal 0 1f0 x 1 :x-offset 10))))

(test overlapping-spans-of-one-array-are-rejected
  (with-mock-arrays (device (x 20))
    (signals gpu:gpu-error (blas:scopy 5 x 1 x 1 :y-offset 2))
    (signals gpu:gpu-error (blas:sswap 5 x 1 x -1))
    (signals gpu:gpu-error (blas:saxpy 5 1f0 x 1 x 2))
    (finishes (blas:scopy 5 x 1 x 1 :y-offset 5))
    (finishes (blas:sswap 5 x 2 x 2 :y-offset 1))
    (finishes (blas:saxpy 5 1f0 x 1 x 1))
    (finishes (blas:sdot 5 x 1 x 1 :y-offset 2))))

(test empty-calls-do-not-launch
  (with-mock-arrays (device (x 4) (y 4))
    (blas:saxpy 0 1f0 x 1 y 1)
    (blas:saxpy 4 0f0 x 1 y 1)
    (blas:sscal 0 2f0 x 1)
    (blas:scopy 0 x 1 y 1)
    (blas:sswap 0 x 1 y 1)
    (is (eql 0f0 (blas:sdot 0 x 1 y 1)))
    (is (eql 0f0 (blas:sasum 0 x 1)))
    (is (eql 0f0 (blas:snrm2 0 x 1)))
    (is (eql 0 (blas:isamax 0 x 1)))
    (is (zerop (launches device)))))
