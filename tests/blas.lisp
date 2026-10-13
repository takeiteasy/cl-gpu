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

;;; Matrices

(defparameter *kinds* '(:row-major :column-major :padded :subview :reversed))

(defun kind-length (rows cols kind)
  (ecase kind
    ((:row-major :column-major :reversed) (* rows cols))
    (:padded (* rows (+ cols 3)))
    (:subview (* (+ rows 2) (+ cols 3)))))

(defun make-view (library data rows cols kind)
  "A ROWS by COLS view of DATA in the layout KIND, from the LIBRARY :simd or :gpu."
  (multiple-value-bind (make subview)
      (ecase library
        (:simd (values #'simd:make-matrix-view #'simd:matrix-subview))
        (:gpu (values #'blas:make-matrix-view #'blas:matrix-subview)))
    (ecase kind
      (:row-major (funcall make data rows cols))
      (:column-major (funcall make data rows cols :layout :column-major))
      (:padded (funcall make data rows cols :leading-dimension (+ cols 3)))
      (:subview (funcall subview (funcall make data (+ rows 2) (+ cols 3)) 1 2 rows cols))
      (:reversed (funcall make data rows cols :offset (1- (* rows cols))
                                              :row-stride (- cols) :column-stride -1)))))

(defmacro with-both-paths (&body body)
  "Run BODY on the MPS path and on the DSL kernels, with PATH naming the one in use."
  `(dolist (use-mps '(t nil))
     (let ((cl-gpu/metal:*use-mps* use-mps)
           (path (if use-mps "mps" "dsl")))
       (declare (ignorable path))
       ,@body)))

(defun close-k-p (a b k)
  "A and B agree to single precision accumulated over K terms."
  (<= (abs (- a b)) (* 2f-6 (+ k 4) (max 1 (abs a) (abs b)))))

(defun all-close-k-p (a b k)
  (every (lambda (x y) (close-k-p x y k)) a b))

(defparameter *flags* '(:no-transpose :transpose))

(defparameter *scalars* '((1f0 0f0) (2.5f0 0.5f0) (-1f0 1f0) (0f0 2f0) (1f0 1f0) (0.5f0 0f0))
  "(alpha beta) pairs.")

(defun rotate (list count)
  (let ((count (mod count (length list))))
    (append (nthcdr count list) (subseq list 0 count))))

(defun check-gemm (m n k transa transb alpha beta kinds message)
  "Compare SGEMM with the oracle; KINDS are the layouts of A, B and C."
  (destructuring-bind (a-kind b-kind c-kind) kinds
    (let* ((plain-a (eq transa :no-transpose))
           (plain-b (eq transb :no-transpose))
           (ar (if plain-a m k)) (ac (if plain-a k m))
           (br (if plain-b k n)) (bc (if plain-b n k))
           (a (random-vector (kind-length ar ac a-kind)))
           (b (random-vector (kind-length br bc b-kind)))
           (c (random-vector (kind-length m n c-kind)))
           (expected (copy-seq c)))
      (simd:sgemm transa transb alpha
                  (make-view :simd a ar ac a-kind) (make-view :simd b br bc b-kind)
                  beta (make-view :simd expected m n c-kind))
      (gpu:with-gpu-array (ga (gpu:to-device a))
        (gpu:with-gpu-array (gb (gpu:to-device b))
          (gpu:with-gpu-array (gc (gpu:to-device c))
            (let ((view (make-view :gpu gc m n c-kind)))
              (is (eq view (blas:sgemm transa transb alpha (make-view :gpu ga ar ac a-kind)
                                       (make-view :gpu gb br bc b-kind) beta view))))
            (is (all-close-k-p expected (gpu:from-device gc) k) message)))))))

(defparameter *gemm-shapes*
  '((1 1 1) (3 5 7) (16 16 16) (17 17 17) (32 48 20) (64 64 64) (100 70 33) (130 129 131)))

(test sgemm-matches-the-oracle
  (with-both-paths
    (let ((counter 0))
      (loop for (m n k) in *gemm-shapes*
            do (dolist (transa *flags*)
                 (dolist (transb *flags*)
                   (destructuring-bind (alpha beta) (nth (mod counter (length *scalars*)) *scalars*)
                     (check-gemm m n k transa transb alpha beta (subseq (rotate *kinds* counter) 0 3)
                                 (format nil "~a ~dx~dx~d ~a ~a ~a ~a" path m n k transa transb
                                         alpha beta))
                     (incf counter))))))))

(test sgemm-handles-every-layout-pair
  (with-both-paths
    (dolist (a-kind *kinds*)
      (dolist (c-kind *kinds*)
        (check-gemm 19 21 18 :no-transpose :transpose 1.5f0 0.25f0 (list a-kind :padded c-kind)
                    (format nil "~a ~a ~a" path a-kind c-kind))))))

(test sgemm-by-zero-alpha-or-empty-k-scales-c
  (with-both-paths
    (dolist (k '(0 5))
      (let ((c (random-vector 12)))
        (gpu:with-gpu-array (gc (gpu:to-device c))
          (gpu:with-gpu-array (ga (gpu:make-gpu-array 20))
            (gpu:with-gpu-array (gb (gpu:make-gpu-array 20))
              (blas:sgemm :no-transpose :no-transpose (if (zerop k) 1f0 0f0)
                          (blas:make-matrix-view ga 3 k :leading-dimension (max k 1))
                          (blas:make-matrix-view gb k 4) -2f0 (blas:make-matrix-view gc 3 4))
              (is (all-close-p (map 'vector (lambda (x) (* -2 x)) c) (gpu:from-device gc))
                  path))))))))

(test sgemm-by-zero-beta-ignores-c
  (float-features:with-float-traps-masked (:invalid :overflow :divide-by-zero :inexact)
    (with-both-paths
      (dolist (c-kind '(:row-major :column-major))
        (let* ((a (random-vector 6)) (b (random-vector 12))
               (c (make-array 8 :element-type 'single-float
                                :initial-element float-features:single-float-positive-infinity))
               (expected (make-array 8 :element-type 'single-float :initial-element 1f0)))
          (simd:sgemm :no-transpose :no-transpose 1f0 (simd:make-matrix-view a 2 3)
                      (simd:make-matrix-view b 3 4) 0f0
                      (simd:make-matrix-view expected 2 4 :layout c-kind))
          (gpu:with-gpu-array (ga (gpu:to-device a))
            (gpu:with-gpu-array (gb (gpu:to-device b))
              (gpu:with-gpu-array (gc (gpu:to-device c))
                (blas:sgemm :no-transpose :no-transpose 1f0 (blas:make-matrix-view ga 2 3)
                            (blas:make-matrix-view gb 3 4) 0f0
                            (blas:make-matrix-view gc 2 4 :layout c-kind))
                (is (all-close-k-p expected (gpu:from-device gc) 3) (format nil "~a ~a" path c-kind))))))))))

(test sgemm-leaves-the-rest-of-the-array-alone
  (with-both-paths
    (let* ((data (random-vector 64))
           (before (copy-seq data)))
      (gpu:with-gpu-array (g (gpu:to-device data))
        (let ((whole (blas:make-matrix-view g 8 8)))
          (blas:sgemm :no-transpose :no-transpose 1f0
                      (blas:matrix-subview whole 0 0 4 4) (blas:matrix-subview whole 0 4 4 4) 0f0
                      (blas:matrix-subview whole 4 0 4 4)))
        (is (equalp (subseq before 0 32) (subseq (gpu:from-device g) 0 32)) path)
        (is (equalp (subseq before 36 40) (subseq (gpu:from-device g) 36 40)) path)))))

(test sgemm-handles-a-view-ending-in-a-short-row
  (with-both-paths
    (let* ((rows 5) (cols 4) (ld 7) (length (+ (* (1- rows) ld) cols))
           (a (random-vector length)) (b (random-vector 16))
           (c (random-vector 20))
           (expected (copy-seq c)))
      (flet ((a-view (library data)
               (funcall (if (eq library :simd) #'simd:make-matrix-view #'blas:make-matrix-view)
                        data rows cols :leading-dimension ld)))
        (simd:sgemm :no-transpose :no-transpose 1f0 (a-view :simd a) (make-view :simd b 4 4 :row-major)
                    0.5f0 (make-view :simd expected 5 4 :row-major))
        (gpu:with-gpu-array (ga (gpu:to-device a))
          (gpu:with-gpu-array (gb (gpu:to-device b))
            (gpu:with-gpu-array (gc (gpu:to-device c))
              (blas:sgemm :no-transpose :no-transpose 1f0 (a-view :gpu ga)
                          (make-view :gpu gb 4 4 :row-major) 0.5f0 (make-view :gpu gc 5 4 :row-major))
              (is (all-close-k-p expected (gpu:from-device gc) 4) path))))))))

(test sgemm-of-a-large-matrix-matches-the-oracle
  (with-both-paths
    (check-gemm 300 200 400 :no-transpose :no-transpose 1f0 0f0 '(:row-major :row-major :row-major)
                path)))

;;; gemv

(defparameter *gemv-shapes* '((1 1) (5 7) (33 17) (100 64) (257 3)))

(defparameter *increments* '((1 1) (2 3) (-1 1) (1 -2)))

(defun check-gemv (m n transpose alpha beta incx incy kind message)
  "Compare SGEMV of an M by N op(matrix) with the oracle. The oracle takes only regular layouts, so
a reversed view is compared with the reversed data in row-major order."
  (let* ((plain (eq transpose :no-transpose))
         (rows (if plain m n)) (cols (if plain n m))
         (a (random-vector (kind-length rows cols kind)))
         (x (random-vector (span-length n incx 1)))
         (y (random-vector (span-length m incy 2)))
         (expected (copy-seq y)))
    (simd:sgemv transpose alpha
                (if (eq kind :reversed)
                    (make-view :simd (reverse a) rows cols :row-major)
                    (make-view :simd a rows cols kind))
                x incx beta expected incy :x-offset 1 :y-offset 2)
    (gpu:with-gpu-array (ga (gpu:to-device a))
      (gpu:with-gpu-array (gx (gpu:to-device x))
        (gpu:with-gpu-array (gy (gpu:to-device y))
          (is (eq gy (blas:sgemv transpose alpha (make-view :gpu ga rows cols kind) gx incx beta
                                 gy incy :x-offset 1 :y-offset 2)))
          (is (all-close-k-p expected (gpu:from-device gy) n) message))))))

(test sgemv-matches-the-oracle
  (with-both-paths
    (let ((counter 0))
      (loop for (m n) in *gemv-shapes*
            do (dolist (transpose '(:no-transpose :transpose :conjugate-transpose))
                 (destructuring-bind (incx incy) (nth (mod counter (length *increments*)) *increments*)
                   (destructuring-bind (alpha beta) (nth (mod counter (length *scalars*)) *scalars*)
                     (check-gemv m n transpose alpha beta incx incy (nth (mod counter 5) *kinds*)
                                 (format nil "~a ~dx~d ~a ~d ~d ~a ~a" path m n transpose incx incy
                                         alpha beta))
                     (incf counter))))))))

(test sgemv-handles-every-layout
  (with-both-paths
    (dolist (kind *kinds*)
      (dolist (transpose *flags*)
        (check-gemv 40 30 transpose 1.5f0 0.5f0 1 1 kind (format nil "~a ~a ~a" path kind transpose))))))

(test sgemv-by-zero-beta-ignores-y
  (float-features:with-float-traps-masked (:invalid :overflow :divide-by-zero :inexact)
    (with-both-paths
      (let ((a (random-vector 12)) (x (random-vector 4))
            (y (make-array 3 :element-type 'single-float
                             :initial-element float-features:single-float-positive-infinity))
            (expected (make-array 3 :element-type 'single-float :initial-element 1f0)))
        (simd:sgemv :no-transpose 1f0 (simd:make-matrix-view a 3 4) x 1 0f0 expected 1)
        (gpu:with-gpu-array (ga (gpu:to-device a))
          (gpu:with-gpu-array (gx (gpu:to-device x))
            (gpu:with-gpu-array (gy (gpu:to-device y))
              (blas:sgemv :no-transpose 1f0 (blas:make-matrix-view ga 3 4) gx 1 0f0 gy 1)
              (is (all-close-k-p expected (gpu:from-device gy) 4) path))))))))

(test sgemv-by-zero-alpha-scales-y
  (with-both-paths
    (let ((y (random-vector 5)))
      (gpu:with-gpu-array (ga (gpu:make-gpu-array 15))
        (gpu:with-gpu-array (gx (gpu:make-gpu-array 3))
          (gpu:with-gpu-array (gy (gpu:to-device y))
            (blas:sgemv :no-transpose 0f0 (blas:make-matrix-view ga 5 3) gx 1 3f0 gy 1)
            (is (all-close-p (map 'vector (lambda (v) (* 3 v)) y) (gpu:from-device gy)) path)))))))

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

;;; Matrix validation, without a GPU

(test matrix-views-describe-their-layout
  (with-mock-arrays (device (data 24))
    (let ((view (blas:make-matrix-view data 4 5 :leading-dimension 6)))
      (is (equal '(:row-major 6 6 1 0)
                 (list (blas:matrix-view-layout view) (blas:matrix-view-leading-dimension view)
                       (blas:matrix-view-row-stride view) (blas:matrix-view-column-stride view)
                       (blas:matrix-view-offset view))))
      (is (eq data (blas:matrix-view-data view))))
    (let ((view (blas:make-matrix-view data 4 5 :layout :column-major)))
      (is (equal '(:column-major 4 1 4) (list (blas:matrix-view-layout view)
                                              (blas:matrix-view-leading-dimension view)
                                              (blas:matrix-view-row-stride view)
                                              (blas:matrix-view-column-stride view)))))
    (let ((view (blas:make-matrix-view data 4 5 :offset 19 :row-stride -5 :column-stride -1)))
      (is (eq :strided (blas:matrix-view-layout view))))
    (let ((part (blas:matrix-subview (blas:make-matrix-view data 4 5 :leading-dimension 6) 1 2 2 3)))
      (is (equal '(2 3 8 6) (list (blas:matrix-view-rows part) (blas:matrix-view-cols part)
                                  (blas:matrix-view-offset part) (blas:matrix-view-row-stride part)))))))

(test bad-matrix-views-signal-gpu-error
  (with-mock-arrays (device (data 24))
    (let ((ints (gpu:make-gpu-array 24 :device device :element-type 'gpu:i32)))
      (signals gpu:gpu-error (blas:make-matrix-view data 5 5))
      (signals gpu:gpu-error (blas:make-matrix-view data -1 5))
      (signals gpu:gpu-error (blas:make-matrix-view data 4 5 :layout :diagonal))
      (signals gpu:gpu-error (blas:make-matrix-view data 4 5 :offset 5))
      (signals gpu:gpu-error (blas:make-matrix-view data 4 5 :row-stride 5))
      (signals gpu:gpu-error (blas:make-matrix-view data 4 5 :row-stride 5 :column-stride 1
                                                              :leading-dimension 5))
      (signals gpu:gpu-error (blas:make-matrix-view data 4 5 :leading-dimension 4))
      (signals gpu:gpu-error (blas:make-matrix-view data 4 5 :offset 24 :row-stride -5
                                                              :column-stride -1))
      (signals gpu:gpu-error (blas:make-matrix-view ints 4 5))
      (signals gpu:gpu-error (blas:make-matrix-view #(1 2 3) 1 3))
      (signals gpu:gpu-error (blas:matrix-subview (blas:make-matrix-view data 4 5) 3 0 2 5))
      (finishes (blas:make-matrix-view data 0 5 :offset 24)))))

(defun mock-view (array rows cols &rest keys)
  (apply #'blas:make-matrix-view array rows cols keys))

(test bad-gemm-arguments-signal-gpu-error
  (with-mock-arrays (device (a 12) (b 12) (c 12))
    (let ((other (gpu:make-gpu-array 12 :device (make-instance 'mock-device)))
          (view (mock-view a 3 4)))
      (flet ((gemm (&key (transa :no-transpose) (transb :no-transpose) (alpha 1f0) (beta 0f0)
                      (a view) (b (mock-view b 4 3)) (c (mock-view c 3 3)))
               (blas:sgemm transa transb alpha a b beta c)))
        (signals gpu:gpu-error (gemm :transa :flip))
        (signals gpu:gpu-error (gemm :transb nil))
        (signals gpu:gpu-error (gemm :alpha "1"))
        (signals gpu:gpu-error (gemm :beta :zero))
        (signals gpu:gpu-error (gemm :a #(1 2 3)))
        (signals gpu:gpu-error (gemm :b (mock-view b 3 4)))
        (signals gpu:gpu-error (gemm :c (mock-view c 3 4)))
        (signals gpu:gpu-error (gemm :transa :transpose))
        (signals gpu:gpu-error (gemm :c (mock-view other 3 3)))
        (gpu:free b)
        (signals gpu:gpu-error (gemm :b (mock-view b 4 3)))
        (is (zerop (launches device)))))))

(test gemm-outputs-must-be-written-once-and-apart-from-inputs
  (with-mock-arrays (device (a 12) (b 12) (shared 40))
    (flet ((gemm (c &optional (a (mock-view a 3 4)) (b (mock-view b 4 3)))
             (blas:sgemm :no-transpose :no-transpose 1f0 a b 0f0 c)))
      (signals gpu:gpu-error (gemm (mock-view shared 3 3 :row-stride 0 :column-stride 1)))
      (signals gpu:gpu-error (gemm (mock-view shared 3 3 :row-stride 1 :column-stride 1)))
      (signals gpu:gpu-error (gemm (mock-view shared 3 3 :row-stride 2 :column-stride 3)))
      (signals gpu:gpu-error (gemm (mock-view shared 3 3 :column-stride 0 :row-stride 3)))
      (finishes (gemm (mock-view shared 3 3 :row-stride 3 :column-stride 1)))
      (finishes (gemm (mock-view shared 3 3 :row-stride 1 :column-stride 3)))
      (finishes (blas:sgemm :no-transpose :no-transpose 1f0 (mock-view a 1 4) (mock-view b 4 3) 0f0
                            (mock-view shared 1 3 :row-stride 0 :column-stride 1)))
      (signals gpu:gpu-error (gemm (mock-view shared 3 3) (mock-view shared 3 4 :offset 5)))
      (signals gpu:gpu-error (gemm (mock-view a 3 3) (mock-view a 3 4)))
      (finishes (gemm (mock-view shared 3 3) (mock-view shared 3 4 :offset 9) (mock-view b 4 3)))
      (finishes (gemm (mock-view shared 3 3 :offset 20))))))

(test empty-and-trivial-gemm-calls-launch-as-little-as-possible
  (with-mock-arrays (device (a 12) (b 12) (c 12))
    (flet ((gemm (m n k alpha beta)
             (blas:sgemm :no-transpose :no-transpose alpha (mock-view a m k :leading-dimension (max k 1))
                         (mock-view b k n) beta (mock-view c m n))
             (prog1 (launches device) (setf (launches device) 0))))
      (is (= 0 (gemm 0 3 3 1f0 0f0)))
      (is (= 0 (gemm 3 0 3 1f0 0f0)))
      (is (= 0 (gemm 3 3 3 0f0 1f0)))
      (is (= 0 (gemm 3 3 0 1f0 1f0)))
      (is (= 1 (gemm 3 3 0 1f0 0f0)))
      (is (= 1 (gemm 3 3 3 0f0 2f0)))
      (is (= 1 (gemm 3 3 3 1f0 0f0))))))

(test bad-gemv-arguments-signal-gpu-error
  (with-mock-arrays (device (a 12) (x 10) (y 10) (shared 30))
    (let ((view (mock-view a 3 4)))
      (signals gpu:gpu-error (blas:sgemv :flip 1f0 view x 1 0f0 y 1))
      (signals gpu:gpu-error (blas:sgemv :no-transpose "1" view x 1 0f0 y 1))
      (signals gpu:gpu-error (blas:sgemv :no-transpose 1f0 #(1) x 1 0f0 y 1))
      (signals gpu:gpu-error (blas:sgemv :no-transpose 1f0 view x 0 0f0 y 1))
      (signals gpu:gpu-error (blas:sgemv :no-transpose 1f0 view x 4 0f0 y 1))
      (signals gpu:gpu-error (blas:sgemv :no-transpose 1f0 view x 1 0f0 y 1 :y-offset 8))
      (signals gpu:gpu-error (blas:sgemv :no-transpose 1f0 view shared 1 0f0 shared 1 :y-offset 2))
      (signals gpu:gpu-error (blas:sgemv :no-transpose 1f0 (mock-view shared 3 4) shared 1 0f0
                                         shared 1 :x-offset 12 :y-offset 4))
      (finishes (blas:sgemv :no-transpose 1f0 (mock-view shared 3 4) shared 1 0f0 shared 1
                            :x-offset 8 :y-offset 20))
      (finishes (blas:sgemv :transpose 1f0 view x 1 0f0 y 1))
      (is (= 2 (launches device))))))

(test empty-and-trivial-gemv-calls-launch-as-little-as-possible
  (with-mock-arrays (device (a 12) (x 10) (y 10))
    (flet ((gemv (m n alpha beta)
             (blas:sgemv :no-transpose alpha (mock-view a m n :leading-dimension (max n 1)) x 1 beta y 1)
             (prog1 (launches device) (setf (launches device) 0))))
      (is (= 0 (gemv 0 3 1f0 0f0)))
      (is (= 0 (gemv 3 3 0f0 1f0)))
      (is (= 0 (gemv 3 0 1f0 1f0)))
      (is (= 1 (gemv 3 0 1f0 0f0)))
      (is (= 1 (gemv 3 3 0f0 2f0)))
      (is (= 1 (gemv 3 3 1f0 0f0))))))
