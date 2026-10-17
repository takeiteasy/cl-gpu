(defpackage #:cl-gpu/reductions/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:gpu #:cl-gpu) (#:red #:cl-gpu/reductions))
  (:export #:run-tests #:run-metal-tests))

(in-package #:cl-gpu/reductions/tests)

(def-suite :cl-gpu/reductions)
(def-suite :cl-gpu/reductions/metal)
(in-suite :cl-gpu/reductions)

(defun run-tests () (run! :cl-gpu/reductions))
(defun run-metal-tests () (run! :cl-gpu/reductions/metal))

(defclass mock-device () ((launches :initform 0 :accessor launches)))
(defmethod gpu:device-allocate ((device mock-device) bytes)
  (make-array bytes :element-type '(unsigned-byte 8) :initial-element 0))
(defmethod gpu:device-free ((device mock-device) buffer) (declare (ignore buffer)))
(defmethod gpu:device-store ((device mock-device) buffer sequence type offset)
  (declare (ignore buffer sequence type offset)))
(defmethod gpu:device-target ((device mock-device)) :metal)
(defmethod gpu:device-prepare ((device mock-device) source entry &key precise-math)
  (declare (ignore entry precise-math)) source)
(defmethod gpu:device-launch ((device mock-device) prepared arguments grid group)
  (declare (ignore prepared arguments grid group)) (incf (launches device)))

(defmacro with-mock-data ((device data length &optional (element 'gpu:f32)) &body body)
  `(let ((,device (make-instance 'mock-device)))
     (gpu:with-gpu-array (,data (gpu:make-gpu-array ,length :element-type ',element :device ,device))
       (locally ,@body))))

(test views-copy-and-validate-metadata
  (with-mock-data (device data 30)
    (let* ((shape (vector 2 3)) (strides (vector 6 -2))
           (view (red:make-strided-view data shape :strides strides :offset 5)))
      (setf (aref shape 0) 100 (aref strides 1) 100)
      (is (equalp #(2 3) (red:strided-view-shape view)))
      (is (equalp #(6 -2) (red:strided-view-strides view)))
      (is (= 5 (red:strided-view-offset view))))
    (is (equalp #(3 1) (red:strided-view-strides (red:make-strided-view data '(2 3)))))
    (is (equalp #() (red:strided-view-shape (red:make-strided-view data #()))))
    (dolist (thunk (list (lambda () (red:make-strided-view data '(-1)))
                        (lambda () (red:make-strided-view data '(2 3) :strides '(1)))
                        (lambda () (red:make-strided-view data '(2 3) :strides '(6 -2)))
                        (lambda () (red:make-strided-view data '(31)))
                        (lambda () (red:make-strided-view data '(0) :offset 31))
                        (lambda () (red:make-strided-view data '(2) :strides (list (expt 2 64))))))
      (signals gpu:gpu-error (funcall thunk)))))

(test invalid-calls-launch-nothing
  (with-mock-data (device data 40)
    (let ((input (red:make-strided-view data '(2 3)))
          (out (red:make-strided-view data '(2) :offset 20))
          (bad (red:make-strided-view data '(2) :strides '(0) :offset 20)))
      (dolist (axis '(2 -3 (0 -2) :bad #(1 1)))
        (signals gpu:gpu-error (red:sum input :axis axis)))
      (signals gpu:gpu-error (red:sum! out input))
      (signals gpu:gpu-error (red:sum! bad input :axis 1))
      (signals gpu:gpu-error (red:rmsnorm input :epsilon -1))
      (signals gpu:gpu-error (red:rmsnorm input :epsilon -1d-100))
      (signals gpu:gpu-error (red:rmsnorm input :weights "bad"))
      (signals gpu:gpu-error (red:rmsnorm input :weights out))
      (signals gpu:gpu-error (red:sum! input input :axis 1))
      (let* ((other (make-instance 'mock-device))
             (storage (gpu:make-gpu-array 2 :device other)))
        (unwind-protect
             (signals gpu:gpu-error (red:sum! (red:make-strided-view storage '(2)) input :axis 1))
          (gpu:free storage)))
      (gpu:free data)
      (signals gpu:gpu-error (red:sum input))
      (is (zerop (launches device))))))

(test wrong-types-and-freed-weights-are-rejected
  (with-mock-data (device data 6)
    (let ((input (red:make-strided-view data '(2 3))))
      (gpu:with-gpu-array (half (gpu:make-gpu-array 6 :element-type 'gpu:f16 :device device))
        (signals gpu:gpu-error (red:softmax! (red:make-strided-view half '(2 3)) input))
        (signals gpu:gpu-error (red:rmsnorm input :weights (red:make-strided-view half '(2 3)))))
      (gpu:with-gpu-array (weights (gpu:make-gpu-array 3 :device device))
        (let ((view (red:make-strided-view weights '(3))))
          (gpu:free weights)
          (signals gpu:gpu-error (red:rmsnorm input :weights view))))
      (is (zerop (launches device))))))

(test empty-outputs-validate-without-launches
  (with-mock-data (device data 0)
    (let ((input (red:make-strided-view data '(0 3))))
      (dolist (function (list #'red:sum #'red:mean #'red:prod #'red:minimum #'red:maximum
                             #'red:argmin #'red:argmax #'red:softmax #'red:rmsnorm))
        (let ((result (funcall function input :axis 1)))
          (is (zerop (gpu:gpu-array-length (red:strided-view-data result))))
          (gpu:free (red:strided-view-data result))))
      (signals gpu:gpu-error (red:mean input :axis 0))
      (signals gpu:gpu-error (red:argmin input :axis 0))
      (signals gpu:gpu-error (red:softmax input :axis 2))
      (signals gpu:gpu-error (red:rmsnorm input :epsilon -1))
      (is (zerop (launches device))))))

(test duplicate-and-irregular-output-addresses
  (with-mock-data (device data 40)
    (is (cl-gpu/reductions::unique-view-p
         (red:make-strided-view data '(3 2) :strides '(2 3))))
    (is (not (cl-gpu/reductions::unique-view-p
              (red:make-strided-view data '(3 3) :strides '(2 2)))))
    (is (cl-gpu/reductions::unique-view-p
         (red:make-strided-view data '(2 3) :strides '(-6 2) :offset 6)))))

(test unsupported-element-types-and-nested-view-bounds
  (with-mock-data (device data 20 gpu:i32)
    (signals gpu:gpu-error (red:sum (red:make-strided-view data '(20)))))
  (with-mock-data (device data 20)
    (let* ((window (gpu:make-gpu-array-view data 10 :offset 5))
           (nested (gpu:make-gpu-array-view window 4 :offset 2))
           (view (red:make-strided-view nested '(4) :strides '(-1) :offset 3)))
      (is (= 10 (cl-gpu/reductions::view-base view)))
      (signals gpu:gpu-error (red:make-strided-view nested '(5)))
      (signals gpu:gpu-error (red:make-strided-view nested '(4) :offset 1)))))
