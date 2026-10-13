(defpackage #:cl-gpu/metal/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:gpu #:cl-gpu) (#:metal #:cl-gpu/metal))
  (:export #:run-tests))

;;; Package that kernel text is read in.
(defpackage #:cl-gpu/metal/tests/kernel
  (:use #:cl #:cl-gpu))

(in-package #:cl-gpu/metal/tests)

(def-suite :cl-gpu/metal)
(in-suite :cl-gpu/metal)

(defun run-tests ()
  (run! :cl-gpu/metal))

(defun kernel-source (text arguments)
  (let ((*package* (find-package '#:cl-gpu/metal/tests/kernel)))
    (gpu:kernel-source
     (gpu:lower-kernel (first (gpu:read-kernel-from-string text :package *package*))
                       :name 'k
                       :arguments (read-from-string arguments))
     :metal)))

(defmacro with-kernel ((pipeline text arguments) &body body)
  `(metal:with-library (library (metal:default-device) (kernel-source ,text ,arguments))
     (metal:with-pipeline (,pipeline library "k")
       ,@body)))

(defun filled (count function)
  (let ((vector (make-array count)))
    (dotimes (index count vector)
      (setf (aref vector index) (funcall function index)))))

(defun read-back (buffer type count)
  (metal:copy-from-buffer buffer (make-array count) :element-type type))

(defun close-p (a b)
  (<= (abs (- a b)) (* 1e-5 (max 1 (abs a) (abs b)))))

;;; Device and buffers

(test default-device-has-a-name
  (let ((device (metal:default-device)))
    (is (plusp (length (metal:device-name device))))
    (is (eq device (metal:default-device)))
    (is (>= (metal:device-max-threads device) 256))))

(test buffers-round-trip-elements
  (loop for (type values) in '((gpu:f32 #(1.5 -2.25 0.0))
                               (gpu:u32 #(0 7 4294967295))
                               (gpu:i8 #(-128 0 127))
                               (gpu:bool #(t nil t))
                               (gpu:f16 #(1.5 -2.0 0.25 65504.0 6.1035156e-5))
                               (gpu:bf16 #(1.5 -2.0 0.25 256.0)))
        do (metal:with-buffer (buffer (metal:default-device) 64)
             (metal:copy-to-buffer buffer values :element-type type)
             (is (equalp values (read-back buffer type (length values)))
                 "~a did not round-trip" type))))

(test buffers-start-zeroed-and-reject-overflow
  (metal:with-buffer (buffer (metal:default-device) 16)
    (is (= 16 (metal:buffer-size buffer)))
    (is (equalp #(0 0 0 0) (read-back buffer 'gpu:u32 4)))
    (signals metal:metal-error
      (metal:copy-to-buffer buffer (make-array 5 :initial-element 0) :element-type 'gpu:u32))))

(test released-objects-are-rejected
  (let ((buffer (metal:make-buffer (metal:default-device) 16)))
    (metal:release buffer)
    (metal:release buffer)
    (signals metal:metal-error (metal:buffer-size buffer))))

;;; Compiling

(test compile-errors-carry-the-log
  (handler-case (progn (metal:compile-library (metal:default-device) "kernel void k( {")
                       (fail "no error signalled"))
    (metal:metal-compile-error (condition)
      (is (search "error" (metal:metal-error-message condition))))))

(test unknown-kernel-names-are-an-error
  (metal:with-library (library (metal:default-device)
                         "#include <metal_stdlib>
kernel void present() {}")
    (signals metal:metal-error (metal:make-pipeline library "absent"))))

;;; Running

(test saxpy-runs-on-a-grid-that-is-not-a-multiple-of-the-group
  (let* ((n 1000)
         (x (filled n (lambda (i) (float i 1f0))))
         (y (filled n (lambda (i) (float (* 2 i) 1f0)))))
    (with-kernel (pipeline "(let ((i (global-id 0)))
                              (when (< i n)
                                (setf (aref out i) (+ (* a (aref x i)) (aref y i)))))"
                           "((out (ptr f32 :global)) (x (ptr f32 :global)) (y (ptr f32 :global))
                             (a f32) (n u32))")
      (metal:with-buffer (out (metal:default-device) (* 4 n))
        (metal:with-buffer (xb (metal:default-device) (* 4 n))
          (metal:with-buffer (yb (metal:default-device) (* 4 n))
            (metal:copy-to-buffer xb x)
            (metal:copy-to-buffer yb y)
            (metal:dispatch pipeline
                            (list out xb yb '(:bytes gpu:f32 3.0) `(:bytes gpu:u32 ,n))
                            n :group 64)
            (let ((result (read-back out 'gpu:f32 n)))
              (is (every #'close-p result
                         (map 'vector (lambda (a b) (+ (* 3 a) b)) x y))))))))))

(test default-group-covers-small-grids
  (with-kernel (pipeline "(setf (aref out (global-id 0)) 1)" "((out (ptr u32 :global)))")
    (metal:with-buffer (out (metal:default-device) 12)
      (metal:dispatch pipeline (list out) 3)
      (is (equalp #(1 1 1) (read-back out 'gpu:u32 3))))))

(test scalar-and-vector-arguments-are-packed
  (with-kernel (pipeline "(progn (setf (aref out 0) n)
                                 (setf (aref out 1) (convert u32 (aref v 0)))
                                 (setf (aref out 2) (convert u32 (aref v 1)))
                                 (setf (aref out 3) (convert u32 (aref v 2)))
                                 (setf (aref out 4) k))"
                         "((out (ptr u32 :global)) (n u32) (v (vec f32 3)) (k u32))")
    (metal:with-buffer (out (metal:default-device) 20)
      (metal:dispatch pipeline
                      (list out '(:bytes gpu:u32 7) '(:bytes (gpu:vec gpu:f32 3) #(1.0 2.0 3.0))
                            '(:bytes gpu:u32 9))
                      1)
      (is (equalp #(7 1 2 3 9) (read-back out 'gpu:u32 5))))))

(test four-vectors-are-packed
  (with-kernel (pipeline "(setf (aref out 0) (aref v 3))"
                         "((out (ptr f32 :global)) (v (vec f32 4)))")
    (metal:with-buffer (out (metal:default-device) 4)
      (metal:dispatch pipeline (list out '(:bytes (gpu:vec gpu:f32 4) #(1.0 2.0 3.0 4.5))) 1)
      (is (equalp #(4.5) (read-back out 'gpu:f32 1))))))

(test threadgroup-arguments-bind-between-buffers
  (with-kernel (pipeline "(progn (setf (aref tile (local-id 0)) (aref a (global-id 0)))
                                 (barrier)
                                 (setf (aref out (global-id 0))
                                       (aref tile (- (- (group-size 0) 1) (local-id 0)))))"
                         "((a (ptr u32 :global)) (tile (ptr u32 :shared)) (out (ptr u32 :global)))")
    (metal:with-buffer (a (metal:default-device) (* 4 256))
      (metal:with-buffer (out (metal:default-device) (* 4 256))
        (metal:copy-to-buffer a (filled 256 #'identity) :element-type 'gpu:u32)
        (metal:dispatch pipeline (list a '(:threadgroup 256) out) 256 :group 64)
        (is (equalp (filled 256 (lambda (i) (+ (* 64 (floor i 64)) (- 63 (mod i 64)))))
                    (read-back out 'gpu:u32 256)))))))

(test atomics-count-every-thread
  (with-kernel (pipeline "(atomic-add c 1)" "((c (ptr (atomic u32) :global)))")
    (metal:with-buffer (c (metal:default-device) 4)
      (metal:dispatch pipeline (list c) 1000 :group 64)
      (is (equalp #(1000) (read-back c 'gpu:u32 1))))))

(test commands-can-be-waited-on-later
  (with-kernel (pipeline "(setf (aref out (global-id 0)) 5)" "((out (ptr u32 :global)))")
    (metal:with-buffer (out (metal:default-device) 4)
      (let ((command (metal:dispatch pipeline (list out) 1 :wait nil)))
        (metal:wait command)
        (is (metal:command-done-p command))
        (metal:wait command)
        (is (equalp #(5) (read-back out 'gpu:u32 1)))))))

;;; define-kernel and GPU arrays

(defun eval-kernel (text)
  (let ((*package* (find-package '#:cl-gpu/metal/tests/kernel)))
    (eval (read-from-string text))))

(defun kernel-function (name)
  (symbol-function (intern (string-upcase name) '#:cl-gpu/metal/tests/kernel)))

(test define-kernel-runs-saxpy
  (eval-kernel "(define-kernel saxpy ((out (ptr f32 :global)) (x (ptr f32 :global))
                                      (y (ptr f32 :global)) (a f32) (n u32))
                  (let ((i (global-id 0)))
                    (when (< i n)
                      (setf (aref out i) (+ (* a (aref x i)) (aref y i))))))")
  (gpu:with-gpu-array (x (gpu:to-device (filled 1000 (lambda (i) (float i)))))
    (gpu:with-gpu-array (y (gpu:to-device (filled 1000 (constantly 2.0))))
      (gpu:with-gpu-array (out (gpu:make-gpu-array 1000))
        (is (null (funcall (kernel-function "saxpy") out x y 3.0 1000 :grid 1000)))
        (is (every #'close-p (filled 1000 (lambda (i) (+ (* 3.0 i) 2.0)))
                   (gpu:from-device out)))))))

(test define-kernel-runs-local-functions
  (eval-kernel "(define-kernel local-functions ((out (ptr u32 :global)) (k u32))
                  (symbol-macrolet ((cell (aref out (global-id 0))))
                    (flet ((scale (x) (* x k)))
                      (setf cell (scale (global-id 0)))
                      (incf cell 1))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 8 :element-type 'gpu:u32))
    (funcall (kernel-function "local-functions") out 3 :grid 8)
    (is (equalp (filled 8 (lambda (i) (1+ (* 3 i)))) (gpu:from-device out)))))

(test define-kernel-launches-asynchronously
  (eval-kernel "(define-kernel fill-five ((out (ptr u32 :global)))
                  (setf (aref out (global-id 0)) 5))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 64 :element-type 'gpu:u32))
    (let ((command (funcall (kernel-function "fill-five") out :grid 64 :wait nil)))
      (gpu:wait command)
      (is (every (lambda (value) (= value 5)) (gpu:from-device out))))))

(test define-kernel-binds-shared-memory-by-element-count
  (eval-kernel "(define-kernel reverse-groups ((a (ptr u32 :global)) (tile (ptr u32 :shared))
                                               (out (ptr u32 :global)))
                  (setf (aref tile (local-id 0)) (aref a (global-id 0)))
                  (barrier)
                  (setf (aref out (global-id 0))
                        (aref tile (- (- (group-size 0) 1) (local-id 0)))))")
  (gpu:with-gpu-array (a (gpu:to-device (filled 256 #'identity) :element-type 'gpu:u32))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 256 :element-type 'gpu:u32))
      (funcall (kernel-function "reverse-groups") a 64 out :grid 256 :group 64)
      (is (equalp (filled 256 (lambda (i) (+ (* 64 (floor i 64)) (- 63 (mod i 64)))))
                  (gpu:from-device out))))))

(test define-kernel-passes-vector-and-small-integer-parameters
  (eval-kernel "(define-kernel scale-vec ((out (ptr f32 :global)) (v (vec f32 3)) (k i8))
                  (setf (aref out 0) (* (convert f32 k) (aref v 2))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 1))
    (funcall (kernel-function "scale-vec") out #(1.0 2.0 4.0) -3 :grid 1)
    (is (equalp #(-12.0) (gpu:from-device out)))))

(test define-kernel-reports-bad-launches
  (eval-kernel "(define-kernel noop ((out (ptr f32 :global))) (setf (aref out 0) 1.0))")
  (let ((noop (kernel-function "noop")))
    (gpu:with-gpu-array (ints (gpu:make-gpu-array 1 :element-type 'gpu:i32))
      (gpu:with-gpu-array (floats (gpu:make-gpu-array 1))
        (signals gpu:gpu-error (funcall noop ints :grid 1))
        (signals gpu:gpu-error (funcall noop floats))
        (signals gpu:gpu-error (funcall noop floats :grid 1 :device nil :group 0))))))

(test freed-arrays-release-their-buffers
  (let ((array (gpu:make-gpu-array 4)))
    (is (plusp (metal:buffer-size (gpu::array-buffer array))))
    (gpu:free array)
    (gpu:free array)
    (signals gpu:gpu-error (gpu:from-device array))))

(test metal-errors-are-gpu-errors
  (signals gpu:gpu-error (metal:compile-library (metal:default-device) "not metal")))
