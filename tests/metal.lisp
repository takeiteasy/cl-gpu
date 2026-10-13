(defpackage #:cl-gpu/metal/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:gpu #:cl-gpu) (#:metal #:cl-gpu/metal) (#:bt #:bordeaux-threads))
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

(defun eval-kernel (text)
  (let ((*package* (find-package '#:cl-gpu/metal/tests/kernel)))
    (eval (read-from-string text))))

(defun kernel-function (name)
  (symbol-function (intern (string-upcase name) '#:cl-gpu/metal/tests/kernel)))

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

;;; Private storage

(defun private-buffer (bytes)
  (metal:make-buffer (metal:default-device) bytes :storage :private))

(test auto-storage-follows-unified-memory
  (let ((device (metal:default-device)))
    (metal:with-buffer (buffer device 16 :storage :auto)
      (is (eq (if (metal:device-unified-memory-p device) :shared :private)
              (metal:buffer-storage buffer))))
    (metal:with-buffer (buffer device 16 :storage :shared)
      (is (eq :shared (metal:buffer-storage buffer))))
    (let ((metal:*buffer-storage* :private))
      (metal:with-buffer (buffer device 16)
        (is (eq :private (metal:buffer-storage buffer)))))
    (signals metal:metal-error (metal:make-buffer device 16 :storage :managed))))

(test private-buffers-start-zeroed-and-reject-overflow
  (metal:with-buffer (buffer (metal:default-device) 6 :storage :private)
    (is (= 6 (metal:buffer-size buffer)))
    (is (equalp #(0 0 0 0 0 0) (read-back buffer 'gpu:u8 6)))
    (signals metal:metal-error
      (metal:copy-to-buffer buffer (make-array 7 :initial-element 0) :element-type 'gpu:u8))
    (signals metal:metal-error (metal:buffer-pointer buffer))))

(test private-buffers-round-trip-elements
  (loop for (type values) in '((gpu:f32 #(1.5 -2.25 0.0))
                               (gpu:u32 #(0 7 4294967295))
                               (gpu:i8 #(-128 0 127))
                               (gpu:u8 #(1 2 3))
                               (gpu:bool #(t nil t nil t))
                               (gpu:f16 #(1.5 -2.0 0.25))
                               (gpu:bf16 #(1.5 -2.0 0.25 256.0 3.0)))
        do (metal:with-buffer (buffer (metal:default-device) 64 :storage :private)
             (metal:copy-to-buffer buffer values :element-type type)
             (is (equalp values (read-back buffer type (length values)))
                 "~a did not round-trip" type))))

(defparameter *raw-samples*
  `((gpu:f32 single-float #(1.5 -2.25 0.0 3.0e10))
    (gpu:f64 double-float #(1.5d0 -2.25d-200 0.0d0 3.0d100))
    (gpu:i8 (signed-byte 8) #(-128 0 127 5))
    (gpu:i16 (signed-byte 16) #(-32768 0 32767 5))
    (gpu:i32 (signed-byte 32) #(-2147483648 0 2147483647 5))
    (gpu:i64 (signed-byte 64) #(-9223372036854775808 0 9223372036854775807 5))
    (gpu:u8 (unsigned-byte 8) #(0 1 255 5))
    (gpu:u16 (unsigned-byte 16) #(0 1 65535 5))
    (gpu:u32 (unsigned-byte 32) #(0 1 4294967295 5))
    (gpu:u64 (unsigned-byte 64) #(0 1 18446744073709551615 5))))

(test specialised-vectors-round-trip-in-bulk
  (dolist (storage '(:shared :private))
    (loop for (type lisp-type values) in *raw-samples*
          do (metal:with-buffer (buffer (metal:default-device) 64 :storage storage)
               (let ((source (coerce values `(simple-array ,lisp-type (*))))
                     (result (make-array (length values) :element-type lisp-type)))
                 (metal:copy-to-buffer buffer source :element-type type)
                 (is (equalp source (metal:copy-from-buffer buffer result :element-type type))
                     "~a ~a did not round-trip" storage type)
                 (is (equalp values (read-back buffer type (length values)))
                     "~a ~a bulk write differs from generic read" storage type))))))

(test specialised-vectors-fill-a-prefix-and-reject-overflow
  (dolist (storage '(:shared :private))
    (metal:with-buffer (buffer (metal:default-device) 16 :storage storage)
      (metal:copy-to-buffer buffer (coerce #(1 2 3 4) '(simple-array (signed-byte 32) (*)))
                            :element-type 'gpu:i32)
      (let ((prefix (make-array 2 :element-type '(signed-byte 32))))
        (is (equalp #(1 2) (metal:copy-from-buffer buffer prefix :element-type 'gpu:i32))))
      (signals metal:metal-error
        (metal:copy-to-buffer buffer (make-array 5 :element-type '(signed-byte 32))
                              :element-type 'gpu:i32))
      (is (equalp #(1 2 3 4) (read-back buffer 'gpu:i32 4))))))

(test specialised-three-vectors-skip-the-padding
  (metal:with-buffer (buffer (metal:default-device) 32 :storage :shared)
    (metal:copy-to-buffer buffer #(9 9 9 9 9 9 9 9) :element-type 'gpu:f32)
    (metal:copy-to-buffer buffer (coerce #(1.0 2.0 3.0 4.0 5.0 6.0) '(simple-array single-float (*)))
                          :element-type '(gpu:vec gpu:f32 3))
    (is (equalp #(1.0 2.0 3.0 9.0 4.0 5.0 6.0 9.0) (read-back buffer 'gpu:f32 8)))
    (let ((result (make-array 6 :element-type 'single-float)))
      (is (equalp #(1.0 2.0 3.0 4.0 5.0 6.0)
                  (metal:copy-from-buffer buffer result :element-type '(gpu:vec gpu:f32 3)))))))

(test other-sequences-copy-component-by-component
  (dolist (storage '(:shared :private))
    (metal:with-buffer (buffer (metal:default-device) 32 :storage storage)
      (let ((expected #(1.0 2.0 3.0 4.0)))
        (dolist (source (list '(1 2 3 4)
                              (coerce #(1.0d0 2.0d0 3.0d0 4.0d0) '(simple-array double-float (*)))
                              (make-array 4 :initial-contents '(1 2 3 4) :adjustable t)
                              (make-array 4 :element-type 'single-float
                                            :initial-contents '(1.0 2.0 3.0 4.0)
                                            :adjustable t)))
          (metal:copy-to-buffer buffer source)
          (is (equalp expected (read-back buffer 'gpu:f32 4))))
        (let ((list (list 0 0 0 0)))
          (is (equal '(1.0 2.0 3.0 4.0) (metal:copy-from-buffer buffer list))))
        (let ((doubles (make-array 4 :element-type 'double-float)))
          (is (equalp #(1.0d0 2.0d0 3.0d0 4.0d0) (metal:copy-from-buffer buffer doubles))))
        (metal:copy-to-buffer buffer '(1 2 3 4) :element-type 'gpu:f64)
        (let ((doubles (make-array 4 :element-type 'double-float)))
          (is (equalp #(1.0d0 2.0d0 3.0d0 4.0d0)
                      (metal:copy-from-buffer buffer doubles :element-type 'gpu:f64))))))))

(test private-vector-elements-round-trip-with-zeroed-padding
  (metal:with-buffer (buffer (metal:default-device) 32 :storage :private)
    (metal:copy-to-buffer buffer #(1 2 3 4 5 6) :element-type '(gpu:vec gpu:f32 3))
    (is (equalp #(1 2 3 4 5 6)
                (read-back buffer '(gpu:vec gpu:f32 3) 6)))
    (is (equalp #(1.0 2.0 3.0 0.0 4.0 5.0 6.0 0.0) (read-back buffer 'gpu:f32 8)))))

(test private-partial-uploads-keep-the-bytes-after-them
  (metal:with-buffer (buffer (metal:default-device) 8 :storage :private)
    (metal:copy-to-buffer buffer #(1 2 3 4 5 6 7 8) :element-type 'gpu:u8)
    (metal:copy-to-buffer buffer #(9 9 9) :element-type 'gpu:u8)
    (is (equalp #(9 9 9 4 5 6 7 8) (read-back buffer 'gpu:u8 8)))
    (metal:copy-to-buffer buffer #(7 7 7 7 7 7) :element-type 'gpu:u8)
    (is (equalp #(7 7 7 7 7 7 7 8) (read-back buffer 'gpu:u8 8)))))

(test private-transfers-reuse-and-grow-the-staging-buffer
  (let ((device (metal:default-device)))
    (metal:with-buffer (small device 16 :storage :private)
      (metal:with-buffer (large device 40000 :storage :private)
        (metal:copy-to-buffer small #(1 2 3 4) :element-type 'gpu:u32)
        (metal:copy-to-buffer small #(5 6 7 8) :element-type 'gpu:u32)
        (is (equalp #(5 6 7 8) (read-back small 'gpu:u32 4)))
        (let ((values (filled 10000 (lambda (i) (* 3 i)))))
          (metal:copy-to-buffer large values :element-type 'gpu:u32)
          (is (equalp values (read-back large 'gpu:u32 10000))))
        (metal:copy-to-buffer small #(9 9 9 9) :element-type 'gpu:u32)
        (is (equalp #(9 9 9 9) (read-back small 'gpu:u32 4)))))))

(test kernels-run-on-private-buffers
  (with-kernel (pipeline "(setf (aref out (global-id 0)) (* 2 (aref in (global-id 0))))"
                         "((out (ptr u32 :global)) (in (ptr u32 :global)))")
    (metal:with-buffer (out (metal:default-device) 16 :storage :private)
      (metal:with-buffer (in (metal:default-device) 16 :storage :private)
        (metal:copy-to-buffer in #(1 2 3 4) :element-type 'gpu:u32)
        (metal:dispatch pipeline (list out in) 4)
        (is (equalp #(2 4 6 8) (read-back out 'gpu:u32 4)))))))

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

(test dispatches-queue-until-the-device-synchronizes
  (with-kernel (pipeline "(setf (aref out (global-id 0)) 5)" "((out (ptr u32 :global)))")
    (metal:with-buffer (out (metal:default-device) 16)
      (is (null (metal:dispatch pipeline (list out) 4)))
      (is (equalp #(5 5 5 5) (read-back out 'gpu:u32 4)) "a host read sees the dispatch"))
    (metal:with-buffer (out (metal:default-device) 16)
      (is (null (metal:dispatch pipeline (list out) 4 :wait t)))
      (is (equalp #(5 5 5 5) (read-back out 'gpu:u32 4))))))

(test host-writes-wait-for-dispatches-that-read
  (with-kernel (pipeline "(setf (aref out (global-id 0)) (aref in (global-id 0)))"
                         "((in (ptr u32 :global)) (out (ptr u32 :global)))")
    (metal:with-buffer (in (metal:default-device) 16)
      (metal:with-buffer (out (metal:default-device) 16)
        (metal:copy-to-buffer in #(1 2 3 4) :element-type 'gpu:u32)
        (metal:dispatch pipeline (list in out) 4)
        (metal:copy-to-buffer in #(9 9 9 9) :element-type 'gpu:u32)
        (is (equalp #(1 2 3 4) (read-back out 'gpu:u32 4)))))))

(test dispatches-run-in-order-across-command-buffers
  (with-kernel (pipeline "(incf (aref out (global-id 0)) 1)" "((out (ptr u32 :global)))")
    (metal:with-buffer (out (metal:default-device) 16)
      (dotimes (i 1000)
        (metal:dispatch pipeline (list out) 4))
      (is (equalp #(1000 1000 1000 1000) (read-back out 'gpu:u32 4))))))

(test releasing-a-device-buffer-after-queueing-keeps-the-result
  (eval-kernel "(define-kernel copy-from ((in (ptr u32 :global)) (out (ptr u32 :global)))
                  (setf (aref out (global-id 0)) (aref in (global-id 0))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 1024 :element-type 'gpu:u32))
    (let ((in (gpu:to-device (filled 1024 (constantly 3)) :element-type 'gpu:u32)))
      (funcall (kernel-function "copy-from") in out :grid 1024)
      (gpu:free in)
      (is (every (lambda (value) (= value 3)) (gpu:from-device out))))))

(test launches-from-many-threads-share-a-device
  (eval-kernel "(define-kernel bump ((out (ptr u32 :global)))
                  (incf (aref out (global-id 0)) 1))")
  (let* ((threads (loop for id below 4
                        collect (let ((id id))
                                  (bt:make-thread
                                   (lambda ()
                                     (gpu:with-gpu-array (out (gpu:make-gpu-array 8 :element-type 'gpu:u32))
                                       (dotimes (i (+ 200 id))
                                         (funcall (kernel-function "bump") out :grid 8))
                                       (gpu:synchronize)
                                       (every (lambda (value) (= value (+ 200 id)))
                                              (gpu:from-device out))))))))
         (results (mapcar #'bt:join-thread threads)))
    (is (every #'identity results))))

;;; define-kernel and GPU arrays

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
    (is (null (funcall (kernel-function "fill-five") out :grid 64)))
    (is (null (gpu:synchronize)))
    (is (every (lambda (value) (= value 5)) (gpu:from-device out)))))

(test define-kernel-waits-on-request
  (eval-kernel "(define-kernel fill-six ((out (ptr u32 :global)))
                  (setf (aref out (global-id 0)) 6))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 64 :element-type 'gpu:u32))
    (is (null (funcall (kernel-function "fill-six") out :grid 64 :wait t)))
    (is (every (lambda (value) (= value 6)) (gpu:from-device out)))))

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

(test remainders-match-common-lisp
  (eval-kernel "(define-kernel int-remainders ((a (ptr i32 :global)) (b (ptr i32 :global))
                                               (m (ptr i32 :global)) (r (ptr i32 :global)))
                  (let ((i (global-id 0)))
                    (setf (aref m i) (mod (aref a i) (aref b i)))
                    (setf (aref r i) (rem (aref a i) (aref b i)))))")
  (eval-kernel "(define-kernel float-remainders ((a (ptr f32 :global)) (b (ptr f32 :global))
                                                 (m (ptr f32 :global)) (r (ptr f32 :global)))
                  (let ((i (global-id 0)))
                    (setf (aref m i) (mod (aref a i) (aref b i)))
                    (setf (aref r i) (rem (aref a i) (aref b i)))))")
  (flet ((check (kernel type dividends divisors)
           (let* ((pairs (loop for a in dividends append (loop for b in divisors collect (cons a b))))
                  (n (length pairs)))
             (gpu:with-gpu-array (a (gpu:to-device (mapcar #'car pairs) :element-type type))
               (gpu:with-gpu-array (b (gpu:to-device (mapcar #'cdr pairs) :element-type type))
                 (gpu:with-gpu-array (m (gpu:make-gpu-array n :element-type type))
                   (gpu:with-gpu-array (r (gpu:make-gpu-array n :element-type type))
                     (funcall (kernel-function kernel) a b m r :grid n)
                     (is (equalp (map 'vector (lambda (pair) (mod (car pair) (cdr pair))) pairs)
                                 (gpu:from-device m))
                         "mod on ~a" type)
                     (is (equalp (map 'vector (lambda (pair) (rem (car pair) (cdr pair))) pairs)
                                 (gpu:from-device r))
                         "rem on ~a" type))))))))
    (check "int-remainders" 'gpu:i32 '(7 -7 0 2147483646 -2147483647) '(3 -3 2147483647))
    (check "float-remainders" 'gpu:f32 '(7.5 -7.5 0.0 6.0) '(2.0 -2.0 0.5))))

(test pointer-arithmetic-writes-through-offset-pointers
  (eval-kernel "(define-kernel offset-writes ((out (ptr u32 :global)) (n (ptr i64 :global)))
                  (let ((q (ptr+ out 4)))
                    (incf q 2)
                    (decf q)
                    (setf (aref q (global-id 0)) (+ 10 (global-id 0)))
                    (when (= (global-id 0) 0)
                      (setf (aref n 0) (ptr- q out)))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 16 :element-type 'gpu:u32))
    (gpu:with-gpu-array (n (gpu:make-gpu-array 1 :element-type 'gpu:i64))
      (funcall (kernel-function "offset-writes") out n :grid 4)
      (is (equalp #(0 0 0 0 0 10 11 12 13 0 0 0 0 0 0 0) (gpu:from-device out)))
      (is (equalp #(5) (gpu:from-device n))))))

(test pointer-comparisons-run-on-the-device
  (eval-kernel "(define-kernel compare-pointers ((out (ptr u32 :global)))
                  (let* ((i (global-id 0))
                         (p (ptr+ out i))
                         (end (ptr+ out 4)))
                    (setf (aref out i)
                          (+ (if (< p end) 1 0)
                             (if (= p (ptr+ out 2)) 10 0)
                             (if (>= p (ptr+ out 6)) 100 0)
                             (if (/= (ptr+ p 1) end) 1000 0)))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 8 :element-type 'gpu:u32))
    (funcall (kernel-function "compare-pointers") out :grid 8)
    (is (equalp (filled 8 (lambda (i)
                            (+ (if (< i 4) 1 0) (if (= i 2) 10 0) (if (>= i 6) 100 0)
                               (if (/= (1+ i) 4) 1000 0))))
                (gpu:from-device out)))))

(test vector-broadcast-and-swizzles-run-on-the-device
  (eval-kernel "(define-kernel vector-math ((out (ptr f32 :global)) (ints (ptr i32 :global))
                                           (v (vec f32 4)) (k (vec i32 2)))
                  (let ((scaled (+ (* v 2.0) 1.0))
                        (flipped (swizzle v :wzyx)))
                    (setf (aref out 0) (aref scaled 0))
                    (setf (aref out 1) (aref scaled 3))
                    (setf (aref out 2) (aref flipped 0))
                    (setf (aref out 3) (swizzle v :b))
                    (setf (aref out 4) (aref (swizzle v :rg) 1))
                    (setf (aref ints 0) (aref (mod (vec i32 -7 7) 3) 0))
                    (setf (aref ints 1) (aref (rem k 3) 0))
                    (setf (aref ints 2) (aref (mod k -3) 1))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 5))
    (gpu:with-gpu-array (ints (gpu:make-gpu-array 3 :element-type 'gpu:i32))
      (funcall (kernel-function "vector-math") out ints #(1.0 2.0 3.0 4.0) #(-7 7) :grid 1)
      (is (equalp #(3.0 9.0 4.0 3.0 2.0) (gpu:from-device out)))
      (is (equalp (vector (mod -7 3) (rem -7 3) (mod 7 -3)) (gpu:from-device ints))))))

(test static-shared-arrays-reverse-each-group
  (eval-kernel "(define-kernel static-reverse ((a (ptr u32 :global)) (out (ptr u32 :global)))
                  (let ((tile (shared-array u32 64)))
                    (setf (aref tile (local-id 0)) (aref a (global-id 0)))
                    (barrier)
                    (setf (aref out (global-id 0))
                          (aref tile (- (- (group-size 0) 1) (local-id 0))))))")
  (gpu:with-gpu-array (a (gpu:to-device (filled 256 #'identity) :element-type 'gpu:u32))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 256 :element-type 'gpu:u32))
      (funcall (kernel-function "static-reverse") a out :grid 256 :group 64)
      (is (equalp (filled 256 (lambda (i) (+ (* 64 (floor i 64)) (- 63 (mod i 64)))))
                  (gpu:from-device out))))))

(test cleared-shared-atomics-count-per-group
  (eval-kernel "(define-kernel group-histogram ((out (ptr u32 :global)))
                  (let ((hist (shared-array (atomic u32) 4 :zero)))
                    (atomic-add (ptr+ hist (logand (local-id 0) 3)) 1)
                    (barrier)
                    (when (< (local-id 0) 4)
                      (setf (aref out (+ (* 4 (group-id 0)) (local-id 0)))
                            (atomic-load (ptr+ hist (local-id 0)))))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 16 :element-type 'gpu:u32))
    (funcall (kernel-function "group-histogram") out :grid 256 :group 64)
    (is (equalp (filled 16 (constantly 16)) (gpu:from-device out)))))

(test cleared-shared-arrays-start-at-zero-in-every-group
  (eval-kernel "(define-kernel cleared-sums ((out (ptr u32 :global)))
                  (let ((tile (shared-array u32 8 :zero)))
                    (when (= (local-id 0) 0)
                      (let ((sum 0))
                        (declare (type u32 sum))
                        (dotimes (i 8) (incf sum (aref tile i)))
                        (setf (aref out (group-id 0)) (+ sum 7))))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 8 :element-type 'gpu:u32))
    (funcall (kernel-function "cleared-sums") out :grid 512 :group 64)
    (is (equalp (filled 8 (constantly 7)) (gpu:from-device out)))))

(test local-arrays-hold-per-thread-state
  (eval-kernel "(define-kernel local-sums ((out (ptr u32 :global)))
                  (let ((acc (local-array u32 4 :zero))
                        (i (global-id 0)))
                    (dotimes (k 4)
                      (setf (aref acc k) (+ (aref acc k) (* i (convert u32 k)))))
                    (setf (aref out i) (+ (aref acc 0) (+ (aref acc 1) (+ (aref acc 2) (aref acc 3)))))))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 32 :element-type 'gpu:u32))
    (funcall (kernel-function "local-sums") out :grid 32)
    (is (equalp (filled 32 (lambda (i) (* 6 i))) (gpu:from-device out)))))

;;; Vector arrays

(test three-vectors-take-four-component-slots
  (metal:with-buffer (buffer (metal:default-device) 32)
    (metal:copy-to-buffer buffer #(1 2 3 4 5 6) :element-type '(gpu:vec gpu:f32 3))
    (is (equalp #(1.0 2.0 3.0 0.0 4.0 5.0 6.0 0.0) (read-back buffer 'gpu:f32 8)))
    (is (equalp #(1.0 2.0 3.0 4.0 5.0 6.0)
                (metal:copy-from-buffer buffer (make-array 6) :element-type '(gpu:vec gpu:f32 3))))
    (signals metal:metal-error
      (metal:copy-to-buffer buffer #(1 2 3 4) :element-type '(gpu:vec gpu:f32 3)))
    (signals metal:metal-error
      (metal:copy-to-buffer buffer (make-array 9) :element-type '(gpu:vec gpu:f32 3)))))

(test vector-arrays-run-through-kernels
  (eval-kernel "(define-kernel double-vec4 ((out (ptr (vec f32 4) :global))
                                            (in (ptr (vec f32 4) :global)))
                  (let ((i (global-id 0)))
                    (setf (aref out i) (* (aref in i) 2.0))))")
  (eval-kernel "(define-kernel double-vec3 ((out (ptr (vec i32 3) :global))
                                            (in (ptr (vec i32 3) :global)))
                  (let ((i (global-id 0)))
                    (setf (aref out i) (* (aref in i) 2))))")
  (flet ((check (kernel type input)
           (gpu:with-gpu-array (in (gpu:to-device input :element-type type))
             (gpu:with-gpu-array (out (gpu:make-gpu-array (gpu:gpu-array-length in) :element-type type))
               (funcall (kernel-function kernel) out in :grid (gpu:gpu-array-length in))
               (is (equalp (map 'vector (lambda (vector) (map 'vector (lambda (x) (* 2 x)) vector))
                                input)
                           (gpu:from-device out)))))))
    (check "double-vec4" '(gpu:vec gpu:f32 4) '(#(1.0 2.0 3.0 4.0) #(5.0 6.0 7.0 8.0)))
    (check "double-vec3" '(gpu:vec gpu:i32 3) '(#(1 2 3) #(4 5 6) #(7 8 9)))))

;;; Threadgroup memory limit

(test the-device-reports-a-threadgroup-memory-limit
  (is (plusp (metal:device-max-threadgroup-memory (metal:default-device))))
  (is (= (metal:device-max-threadgroup-memory (metal:default-device))
         (gpu:device-shared-memory-limit (metal:default-device)))))

(test oversized-shared-memory-is-a-gpu-error
  (eval-kernel "(define-kernel huge-static ((out (ptr f32 :global)))
                  (let ((tile (shared-array f32 100000)))
                    (setf (aref tile 0) 1.0)
                    (barrier)
                    (setf (aref out 0) (aref tile 0))))")
  (eval-kernel "(define-kernel huge-parameter ((out (ptr f32 :global)) (tile (ptr f32 :shared)))
                  (setf (aref tile 0) 1.0)
                  (barrier)
                  (setf (aref out 0) (aref tile 0)))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 1))
    (let ((message (handler-case (funcall (kernel-function "huge-static") out :grid 1)
                     (gpu:gpu-error (condition) (gpu:gpu-error-message condition)))))
      (is (search "huge-static" message))
      (is (search "400000" message)))
    (signals gpu:gpu-error (funcall (kernel-function "huge-parameter") out 100000 :grid 1))
    (funcall (kernel-function "huge-parameter") out 16 :grid 1)
    (is (equalp #(1.0) (gpu:from-device out)))))

(test shared-parameters-round-up-to-sixteen-bytes
  (is (equal '(:threadgroup 16) (metal::metal-argument '(:shared 12))))
  (is (equal '(:threadgroup 32) (metal::metal-argument '(:shared 32))))
  (is (equal '(:threadgroup 0) (metal::metal-argument '(:shared 0))))
  (eval-kernel "(define-kernel odd-shared ((out (ptr f32 :global)) (tile (ptr f32 :shared)))
                  (setf (aref tile 0) 1.0)
                  (barrier)
                  (setf (aref out 0) (aref tile 0)))")
  (gpu:with-gpu-array (out (gpu:make-gpu-array 1))
    (funcall (kernel-function "odd-shared") out 3 :grid 1)
    (is (equalp #(1.0) (gpu:from-device out)))))

(test precise-math-keeps-the-rounding-error-of-a-sum
  (eval-kernel "(define-kernel two-sum ((in (ptr f32 :global)) (out (ptr f32 :global)))
                  (declare (precise-math))
                  (let* ((a (aref in 0))
                         (b (aref in 1))
                         (s (+ a b))
                         (bb (- s a)))
                    (setf (aref out 0) (+ (- a (- s bb)) (- b bb)))))")
  (gpu:with-gpu-array (in (gpu:to-device #(1e8 1.0)))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 1))
      (funcall (kernel-function "two-sum") in out :grid 1)
      (is (equalp #(1.0) (gpu:from-device out))))))

;;; Array transfers

(test from-device-returns-vectors-specialised-to-the-element-type
  (flet ((type-of-result (data type &rest keys)
           (gpu:with-gpu-array (array (gpu:to-device data :element-type type))
             (type-of (apply #'gpu:from-device array keys)))))
    (is (subtypep (type-of-result #(1.0 2.0) 'gpu:f32) '(simple-array single-float (*))))
    (is (subtypep (type-of-result #(1.0 2.0) 'gpu:f16) '(simple-array single-float (*))))
    (is (subtypep (type-of-result #(1.0 2.0) 'gpu:f64) '(simple-array double-float (*))))
    (is (subtypep (type-of-result #(1 2) 'gpu:i32) '(simple-array (signed-byte 32) (*))))
    (is (subtypep (type-of-result #(1 2) 'gpu:u8) '(simple-array (unsigned-byte 8) (*))))
    (is (subtypep (type-of-result #(t nil) 'gpu:bool) 'simple-vector))
    (is (subtypep (type-of-result #(1.0 2.0 3.0 4.0) '(gpu:vec gpu:f32 2) :flat t)
                  '(simple-array single-float (*))))))

(test vector-elements-round-trip-through-specialised-vectors
  (dolist (storage '(:shared :private))
    (let ((gpu::*device* nil)
          (metal:*buffer-storage* storage))
      (gpu:with-gpu-array (array (gpu:to-device '(#(1 2 3) (4 5 6)) :element-type '(gpu:vec gpu:f32 3)))
        (let ((elements (gpu:from-device array)))
          (is (equalp #(#(1.0 2.0 3.0) #(4.0 5.0 6.0)) elements))
          (is (typep (aref elements 0) '(simple-array single-float (3)))))
        (is (equalp #(1.0 2.0 3.0 4.0 5.0 6.0) (gpu:from-device array :flat t)))))))

(test large-transfers-round-trip
  (let* ((count 100000)
         (data (make-array count :element-type 'single-float)))
    (dotimes (index count)
      (setf (aref data index) (float index 1.0)))
    (dolist (storage '(:shared :private))
      (let ((metal:*buffer-storage* storage))
        (gpu:with-gpu-array (array (gpu:to-device data))
          (is (equalp data (gpu:from-device array))))))))
