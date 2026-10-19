(defpackage #:cl-gpu/opencl/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:gpu #:cl-gpu) (#:ocl #:cl-gpu/opencl) (#:bt #:bordeaux-threads))
  (:export #:run-tests))
(in-package #:cl-gpu/opencl/tests)
(def-suite :cl-gpu/opencl)
(in-suite :cl-gpu/opencl)

(defun run-tests ()
  (unless (ocl:available-devices) (error "OpenCL runtime tests require a device"))
  (run! :cl-gpu/opencl))

(defun prepare (text arguments &optional (device (ocl:default-device)))
  (let ((ir (cl-gpu/tests::lower-text text :arguments arguments)))
    (gpu:device-prepare device (gpu:kernel-source ir :opencl) (gpu::kernel-entry-name ir :opencl) :precise-math t)))

(defun define-test-kernel (name text arguments)
  (let ((*package* (find-package :cl-gpu/tests/kernel)))
    (eval (read-from-string (format nil "(define-kernel ~a ~a ~a)" name arguments text)))
    (symbol-function (intern (string-upcase name)))))

(test discovery-and-release
  (let* ((info (first (ocl:available-devices))) (device (ocl:open-device info)))
    (is (plusp (length (ocl:device-name device))))
    (is (listp (ocl:device-extensions device)))
    (is (eq :opencl (gpu:device-target device)))
    (is (plusp (gpu:device-shared-memory-limit device)))
    (ocl:release device)
    (ocl:release device)
    (signals ocl:opencl-error (gpu:device-allocate device 4))))

(test arrays-zero-transfer-and-padding
  (let ((gpu:*device* (ocl:default-device)))
    (loop for type in '(gpu:i8 gpu:u8 gpu:i16 gpu:u16 gpu:i32 gpu:u32 gpu:i64 gpu:u64 gpu:f32)
          do (gpu:with-gpu-array (array (gpu:make-gpu-array 4 :element-type type))
               (is (every #'zerop (gpu:from-device array)))
               (gpu:copy-into array #(1 2 3 4))
               (is (equalp #(1 2 3 4) (gpu:from-device array)))))
    (gpu:with-gpu-array (array (gpu:make-gpu-array 2 :element-type '(gpu:vec gpu:f32 3)
                                                   :initial-contents #(#(1.0 2.0 3.0) #(4.0 5.0 6.0))))
      (is (equalp #(#(1.0 2.0 3.0) #(4.0 5.0 6.0)) (gpu:from-device array))))
    (gpu:with-gpu-array (array (gpu:make-gpu-array 0))
      (is (zerop (length (gpu:from-device array)))))))

(test saxpy-exact-grids-and-queued-order
  (let ((gpu:*device* (ocl:default-device))
        (kernel (define-test-kernel "opencl-saxpy"
                  "(let ((i (global-id 0))) (setf (aref out i) (+ (* a (aref x i)) (aref out i))))"
                  "((out (ptr f32 :global)) (x (ptr f32 :global)) (a f32))")))
    (gpu:with-gpu-array (x (gpu:to-device #(1.0 2.0 3.0 4.0)))
      (gpu:with-gpu-array (out (gpu:make-gpu-array 4))
        (funcall kernel out x 2.0 :grid 4 :group 2)
        (funcall kernel out x 3.0 :grid 4)
        (is (equalp #(5.0 10.0 15.0 20.0) (gpu:from-device out)))
        (signals ocl:opencl-error (funcall kernel out x 1.0 :grid 3 :group 2))
        (gpu:copy-into x #(2.0 2.0 2.0 2.0))
        (funcall kernel out x 1.0 :grid 4 :wait t)
        (is (equalp #(7.0 12.0 17.0 22.0) (gpu:from-device out)))))))

(test aligned-views-and-host-offsets
  (let* ((gpu:*device* (ocl:default-device))
         (alignment (gpu:device-offset-alignment gpu:*device* :global))
         (offset (ceiling alignment 4))
         (kernel (define-test-kernel "opencl-view-copy" "(setf (aref out (global-id 0)) (aref in (global-id 0)))"
                                     "((out (ptr f32 :global)) (in (ptr f32 :constant)))")))
    (gpu:with-gpu-array (data (gpu:make-gpu-array (+ offset 4)))
      (gpu:with-gpu-array (out (gpu:make-gpu-array 4))
        (let ((view (gpu:make-gpu-array-view data 4 :offset offset)))
          (gpu:copy-into view #(1.0 2.0 3.0 4.0))
          (funcall kernel out view :grid 4 :group 2)
          (is (equalp #(1.0 2.0 3.0 4.0) (gpu:from-device out))))
        (when (> alignment 4)
          (signals gpu:gpu-error
            (funcall kernel out (gpu:make-gpu-array-view data 4 :offset 1) :grid 4)))))))

(test shared-arrays-barriers-and-dynamic-local
  (let ((gpu:*device* (ocl:default-device))
        (kernel (define-test-kernel "opencl-shared"
                  "(let ((tile (shared-array f32 4 :zero)) (i (local-id 0)))
                     (setf (aref scratch i) (convert f32 (+ i 1))) (barrier)
                     (setf (aref out (global-id 0)) (+ (aref tile i) (aref scratch (- 3 i)))))"
                  "((out (ptr f32 :global)) (scratch (ptr f32 :shared)))")))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 8))
      (funcall kernel out 4 :grid 8 :group 4)
      (is (equalp #(4.0 3.0 2.0 1.0 4.0 3.0 2.0 1.0) (gpu:from-device out)))
      (signals gpu:gpu-error
        (funcall kernel out (gpu:device-shared-memory-limit gpu:*device*) :grid 8 :group 4)))))

(test atomics-local-state-and-mod
  (let ((gpu:*device* (ocl:default-device))
        (kernel (define-test-kernel "opencl-atomic"
                  "(let ((counts (shared-array (atomic u32) 1 :zero)) (scratch (local-array i32 2 :zero)))
                     (atomic-add counts 1) (barrier)
                     (setf (aref scratch 1) (mod -7 3))
                     (when (= (local-id 0) 0)
                       (setf (aref out (group-id 0)) (+ (atomic-load counts) (convert u32 (aref scratch 1))))))"
                  "((out (ptr u32 :global)))")))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 2 :element-type 'gpu:u32))
      (funcall kernel out :grid 8 :group 4)
      (is (equalp #(6 6) (gpu:from-device out))))))

(test concurrent-launches-bind-arguments-atomically
  (let* ((device (ocl:default-device))
         (kernel (define-test-kernel "opencl-thread" "(setf (aref out (global-id 0)) n)"
                                     "((out (ptr u32 :global)) (n u32))"))
         (arrays (loop repeat 8 collect (gpu:make-gpu-array 64 :element-type 'gpu:u32 :device device))))
    (unwind-protect
         (let ((threads (loop for array in arrays for n from 1 collect
                         (let ((array array) (n n))
                           (bt:make-thread (lambda ()
                                             (dotimes (i 8) (funcall kernel array n :grid 64 :group 8 :device device))))))))
           (mapc #'bt:join-thread threads)
           (loop for array in arrays for n from 1 do (is (every (lambda (value) (= n value)) (gpu:from-device array)))))
      (mapc #'gpu:free arrays))))

(test compilation-errors-and-optional-types
  (let ((device (ocl:default-device)))
    (signals ocl:opencl-compile-error (gpu:device-prepare device "this is not OpenCL" "k"))
    (loop for (type extension) in '(("f16" "cl_khr_fp16") ("f64" "cl_khr_fp64"))
          do (if (member extension (ocl:device-extensions device) :test #'equal)
                 (is (not (null (prepare "(setf (aref out 0) x)" (format nil "((out (ptr ~a :global)) (x ~a))" type type)))))
                 (signals ocl:opencl-compile-error
                   (prepare "(setf (aref out 0) x)" (format nil "((out (ptr ~a :global)) (x ~a))" type type)))))))

(test vectors-conversions-bitcasts-and-remainders
  (let ((gpu:*device* (ocl:default-device))
        (kernel (define-test-kernel "opencl-vector"
                  "(let ((v (vec f32 x 2.0 3.0)))
                     (setf (aref out (global-id 0)) (convert (vec i32 3) v))
                     (setf (aref bits (global-id 0)) (bitcast u32 x))
                     (setf (aref remainders (global-id 0)) (mod x 3.0)))"
                  "((out (ptr (vec i32 3) :global)) (bits (ptr u32 :global)) (remainders (ptr f32 :global)) (x f32))")))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 2 :element-type '(gpu:vec gpu:i32 3)))
      (gpu:with-gpu-array (bits (gpu:make-gpu-array 2 :element-type 'gpu:u32))
        (gpu:with-gpu-array (remainders (gpu:make-gpu-array 2))
          (funcall kernel out bits remainders -7.0 :grid 2 :group 1)
          (is (equalp #(#(-7 2 3) #(-7 2 3)) (gpu:from-device out)))
          (is (every (lambda (value) (= value (float-features:single-float-bits -7.0))) (gpu:from-device bits)))
          (is (equalp #(2.0 2.0) (gpu:from-device remainders))))))))

(test compiled-limits-and-buffer-lifetime
  (let* ((gpu:*device* (ocl:default-device))
         (prepared (prepare "(setf (aref out (global-id 0)) (convert u32 (group-size 0)))" "((out (ptr u32 :global)))"))
         (buffer (gpu:device-allocate gpu:*device* 16)))
    (unwind-protect
         (progn
           (gpu:device-launch gpu:*device* prepared (list (list :buffer buffer 0)) '(4 1 1) '(2 1 1))
           (gpu:device-free gpu:*device* buffer)
           (gpu:synchronize)
           (pass "queued commands retain freed buffer storage")
           (signals ocl:opencl-error
             (gpu:device-launch gpu:*device* prepared (list (list :buffer buffer 0)) '(4 1 1) '(2 1 1))))
      (gpu:device-free gpu:*device* buffer))
    (let ((fresh (gpu:device-allocate gpu:*device* 4)))
      (unwind-protect
           (signals ocl:opencl-error
             (gpu:device-launch gpu:*device* prepared (list (list :buffer fresh 0)) '(1048576 1 1) '(1048576 1 1)))
        (gpu:device-free gpu:*device* fresh)))))

(test larger-transfers-and-host-write-order
  (let* ((gpu:*device* (ocl:default-device))
         (data (make-array 100000 :element-type 'single-float :initial-element 3.0))
         (kernel (define-test-kernel "opencl-copy" "(setf (aref out (global-id 0)) (aref in (global-id 0)))"
                                     "((out (ptr f32 :global)) (in (ptr f32 :global)))")))
    (gpu:with-gpu-array (in (gpu:to-device data))
      (gpu:with-gpu-array (out (gpu:make-gpu-array 100000))
        (funcall kernel out in :grid 100000)
        (fill data 5.0)
        (gpu:copy-into in data)
        (is (every (lambda (value) (= value 3.0)) (gpu:from-device out)))
        (is (equalp data (gpu:from-device in)))))))

(test multidimensional-builtins-and-local-limit
  (let* ((gpu:*device* (ocl:default-device))
         (kernel (define-test-kernel "opencl-axes"
                   "(let ((tile (shared-array u32 4 :zero))
                          (i (+ (local-id 0) (* 2 (local-id 1)))))
                      (setf (aref out (+ (global-id 0) (* 2 (global-id 1))))
                            (+ (aref tile i) (grid-size 0) (grid-size 1))))"
                   "((out (ptr u32 :global)))")))
    (gpu:with-gpu-array (out (gpu:make-gpu-array 4 :element-type 'gpu:u32))
      (funcall kernel out :grid '(2 2) :group '(2 2))
      (is (equalp #(4 4 4 4) (gpu:from-device out)))))
  (let* ((device (ocl:default-device))
         (prepared (prepare "(setf (aref scratch (local-id 0)) 1) (barrier) (setf (aref out 0) (aref scratch 0))"
                            "((out (ptr u32 :global)) (scratch (ptr u32 :shared)))"))
         (buffer (gpu:device-allocate device 4)))
    (unwind-protect
         (signals ocl:opencl-error
           (gpu:device-launch device prepared
                              (list (list :buffer buffer 0) (list :shared (+ 1024 (gpu:device-shared-memory-limit device))))
                              '(1 1 1) '(1 1 1)))
      (gpu:device-free device buffer))))

(test release-cleans-up-after-execution-error
  (let ((device (ocl:open-device (first (ocl:available-devices)))))
    (unwind-protect
         (float-features:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow :inexact)
           (let ((event (ocl::created (status)
                          (cffi:foreign-funcall "clCreateUserEvent"
                                               :pointer (ocl::device-context device) :pointer status :pointer))))
             (push event (ocl::device-events device))
             (ocl::check (cffi:foreign-funcall "clSetUserEventStatus" :pointer event :int32 -1 :int32)
                         "set user event failure"))
           (signals ocl:opencl-error (ocl:release device))
           (is (null (ocl::device-queue device)))
           (is (null (ocl::device-context device))))
      (ocl:release device))))
