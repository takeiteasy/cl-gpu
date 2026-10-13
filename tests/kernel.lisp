(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

;;; A device that stores arrays in Lisp vectors and records launches.

(defclass mock-device ()
  ((prepared :initform '() :accessor prepared)
   (launches :initform '() :accessor launches)
   (freed :initform 0 :accessor freed)))

(defmethod gpu:device-target ((device mock-device)) :metal)
(defmethod gpu:device-allocate ((device mock-device) bytes) (make-array bytes :initial-element 0))
(defmethod gpu:device-free ((device mock-device) buffer) (incf (freed device)))
(defmethod gpu:device-store ((device mock-device) buffer sequence element-type)
  (replace buffer sequence))
(defmethod gpu:device-load ((device mock-device) buffer sequence element-type)
  (replace sequence buffer))
(defmethod gpu:device-prepare ((device mock-device) source entry-name)
  (push (list source entry-name) (prepared device))
  entry-name)
(defmethod gpu:device-launch ((device mock-device) prepared arguments grid group wait)
  (push (list prepared arguments grid group wait) (launches device))
  :command)

(defmacro with-mock-device ((device) &body body)
  `(let* ((,device (make-instance 'mock-device))
          (gpu:*device* ,device))
     ,@body))

(defun eval-kernel-text (text)
  (let ((*package* (find-package '#:cl-gpu/tests/kernel)))
    (eval (read-from-string text))))

(defun kernel-function (name)
  (symbol-function (intern (string-upcase name) '#:cl-gpu/tests/kernel)))

(defun define-copy-kernel (name)
  (eval-kernel-text
   (format nil "(define-kernel ~a ((out (ptr f32 :global)) (in (ptr f32 :global)) (n u32))
                  \"Copy N numbers.\"
                  (let ((i (global-id 0)))
                    (when (< i n)
                      (setf (aref out i) (aref in i)))))"
           name)))

;;; Arrays

(test arrays-round-trip-through-the-device
  (with-mock-device (device)
    (let ((array (gpu:to-device #(1 2 3) :element-type 'gpu:i32)))
      (is (= 3 (gpu:gpu-array-length array)))
      (is (eq (type-of-spec "i32") (gpu:gpu-array-element-type array)))
      (is (equalp #(1 2 3) (gpu:from-device array)))
      (is (equalp #(1 2) (gpu:from-device array :into (make-array 2 :initial-element 0))))
      (gpu:free array))))

(test free-is-idempotent
  (with-mock-device (device)
    (let ((array (gpu:make-gpu-array 4)))
      (gpu:free array)
      (gpu:free array)
      (is (= 1 (freed device)))
      (signals gpu:gpu-error (gpu:from-device array)))))

(test with-gpu-array-frees-on-exit
  (with-mock-device (device)
    (ignore-errors
     (gpu:with-gpu-array (array (gpu:make-gpu-array 4))
       (error "unwind")))
    (is (= 1 (freed device)))))

(test arrays-reject-bad-contents
  (with-mock-device (device)
    (signals gpu:gpu-error (gpu:make-gpu-array 2 :initial-contents #(1 2 3)))
    (is (= 1 (freed device)) "a failed fill frees the array")
    (dolist (type '((gpu:ptr gpu:f32 :global) (gpu:atomic gpu:u32)))
      (signals gpu:gpu-error (gpu:make-gpu-array 2 :element-type type)))))

(test vector-arrays-take-nested-or-flat-contents
  (with-mock-device (device)
    (loop for (type nested flat) in '(((gpu:vec gpu:f32 4) (#(1 2 3 4) (5 6 7 8)) #(1 2 3 4 5 6 7 8))
                                      ((gpu:vec gpu:i32 3) (#(1 2 3) (4 5 6)) #(1 2 3 4 5 6)))
          do (let ((from-nested (gpu:to-device nested :element-type type))
                   (from-flat (gpu:to-device flat :element-type type)))
               (is (= 2 (gpu:gpu-array-length from-nested)))
               (is (= 2 (gpu:gpu-array-length from-flat)))
               (is (eq (gpu:parse-type type) (gpu:gpu-array-element-type from-nested)))
               (is (equalp (map 'vector (lambda (element) (coerce element 'vector)) nested)
                           (gpu:from-device from-nested)))
               (is (equalp flat (gpu:from-device from-flat :flat t)))))))

(test vector-arrays-fill-given-sequences
  (with-mock-device (device)
    (let ((array (gpu:to-device #(#(1 2) #(3 4) #(5 6)) :element-type '(gpu:vec gpu:f32 2))))
      (let ((into (make-array 2 :initial-element 0)))
        (is (eq into (gpu:from-device array :into into)))
        (is (equalp #(#(1 2) #(3 4)) into)))
      (is (equalp #(1 2 3 4) (gpu:from-device array :into (make-array 4) :flat t)))
      (signals gpu:gpu-error (gpu:from-device array :into (make-array 4)))
      (signals gpu:gpu-error (gpu:from-device array :into (make-array 3) :flat t))
      (signals gpu:gpu-error (gpu:from-device array :into (make-array 8) :flat t)))))

(test scalar-arrays-ignore-flat
  (with-mock-device (device)
    (let ((array (gpu:to-device #(1 2 3))))
      (is (equalp #(1.0 2.0 3.0) (gpu:from-device array :flat t))))))

(test vector-arrays-reject-bad-contents
  (with-mock-device (device)
    (let ((type '(gpu:vec gpu:f32 3)))
      (signals gpu:gpu-error (gpu:to-device '(#(1 2 3) #(4 5)) :element-type type))
      (signals gpu:gpu-error (gpu:to-device #(1 2 3 4) :element-type type))
      (signals gpu:gpu-error (gpu:make-gpu-array 1 :element-type type
                                                   :initial-contents #(1 2 3 4 5 6)))
      (signals gpu:gpu-error (gpu:make-gpu-array 1 :element-type type
                                                   :initial-contents '(#(1 2 3) #(4 5 6)))))))

(test vector-arrays-launch-on-matching-pointers
  (with-mock-device (device)
    (eval-kernel-text
     "(define-kernel vector-copy ((out (ptr (vec f32 4) :global)))
        (setf (aref out 0) (aref out 1)))")
    (let ((function (kernel-function "vector-copy"))
          (same (gpu:make-gpu-array 2 :element-type '(gpu:vec gpu:f32 4)))
          (other (gpu:make-gpu-array 2 :element-type '(gpu:vec gpu:f32 2))))
      (is (null (funcall function same :grid 1)))
      (signals gpu:gpu-error (funcall function other :grid 1))
      (is (= 1 (length (launches device)))))))

(test no-backend-is-an-error
  (let ((gpu::*backends* '())
        (gpu:*device* nil))
    (signals gpu:gpu-error (gpu:make-gpu-array 1))))

;;; Defining kernels

(test define-kernel-defines-a-function
  (with-mock-device (device)
    (define-copy-kernel "copy-a")
    (is (gpu:find-kernel (intern "COPY-A" '#:cl-gpu/tests/kernel)))
    (is (search "Copy N" (documentation (kernel-function "copy-a") 'function)))
    (let ((in (gpu:to-device #(1 2))) (out (gpu:make-gpu-array 2)))
      (is (null (funcall (kernel-function "copy-a") out in 2 :grid 2)))
      (destructuring-bind ((entry arguments grid group wait)) (launches device)
        (is (equal "copy_a" entry))
        (is (equal '(:buffer :bytes) (mapcar #'first (list (first arguments) (third arguments)))))
        (is (eq (type-of-spec "u32") (second (third arguments))))
        (is (equal '(2 1 1) grid))
        (is (null group))
        (is (eq t wait))))))

(test pipelines-are-prepared-once-per-device
  (with-mock-device (device)
    (define-copy-kernel "copy-b")
    (let ((in (gpu:to-device #(1))) (out (gpu:make-gpu-array 1)))
      (dotimes (i 3)
        (funcall (kernel-function "copy-b") out in 1 :grid 1))
      (is (= 1 (length (prepared device))))
      (is (= 3 (length (launches device))))
      (is (search "kernel void copy_b" (first (first (prepared device))))))))

(test launch-keywords-reach-the-device
  (with-mock-device (device)
    (define-copy-kernel "copy-c")
    (let ((in (gpu:to-device #(1))) (out (gpu:make-gpu-array 1)))
      (is (eq :command (funcall (kernel-function "copy-c") out in 1
                                :grid '(4 2) :group '(2) :wait nil)))
      (destructuring-bind ((entry arguments grid group wait)) (launches device)
        (declare (ignore entry arguments))
        (is (equal '(4 2 1) grid))
        (is (equal '(2 1 1) group))
        (is (null wait))))))

(test launch-checks-its-arguments
  (with-mock-device (device)
    (define-copy-kernel "copy-d")
    (let ((function (kernel-function "copy-d"))
          (f32 (gpu:to-device #(1)))
          (i32 (gpu:to-device #(1) :element-type 'gpu:i32))
          (freed-array (gpu:to-device #(1)))
          (other (gpu:to-device #(1) :device (make-instance 'mock-device))))
      (gpu:free freed-array)
      (signals gpu:gpu-error (funcall function f32 f32 1))
      (signals gpu:gpu-error (funcall function f32 f32 1 :grid 0))
      (signals gpu:gpu-error (funcall function f32 f32 1 :grid '(1 1 1 1)))
      (signals gpu:gpu-error (funcall function f32 #(1) 1 :grid 1))
      (signals gpu:gpu-error (funcall function f32 i32 1 :grid 1))
      (signals gpu:gpu-error (funcall function f32 freed-array 1 :grid 1))
      (signals gpu:gpu-error (funcall function f32 other 1 :grid 1))
      (signals gpu:gpu-error (funcall function f32 f32 -1 :grid 1))
      (signals gpu:gpu-error (funcall function f32 f32 1.5 :grid 1))
      (is (null (launches device)) "no bad launch reaches the device"))))

(test shared-and-vector-parameters-convert
  (with-mock-device (device)
    (eval-kernel-text
     "(define-kernel scratch ((out (ptr f32 :global)) (tile (ptr f32 :shared)) (v (vec f32 2)))
        (setf (aref out 0) (aref v 0)))")
    (let ((out (gpu:make-gpu-array 1)))
      (funcall (kernel-function "scratch") out 16 #(1.0 2.0) :grid 1)
      (destructuring-bind ((entry arguments &rest rest)) (launches device)
        (declare (ignore entry rest))
        (is (equal '(:shared 64) (second arguments)))
        (is (equalp #(1.0 2.0) (third (third arguments)))))
      (signals gpu:gpu-error (funcall (kernel-function "scratch") out 16 #(1.0) :grid 1)))))

(test parameters-may-share-names-with-launch-keywords
  (with-mock-device (device)
    (eval-kernel-text
     "(define-kernel named ((out (ptr f32 :global)) (grid u32) (wait f32) (group u32) (pi f32))
        (setf (aref out grid) (+ wait pi (convert f32 group))))")
    (funcall (kernel-function "named") (gpu:make-gpu-array 4) 1 2.0 3 3.0 :grid 1)
    (is (= 1 (length (launches device))))))

;;; Compile-time behaviour

(test type-errors-surface-at-macroexpansion
  (let ((*package* (find-package '#:cl-gpu/tests/kernel)))
    (signals gpu:kernel-type-error
      (macroexpand-1 (read-from-string
                      "(define-kernel broken ((out (ptr f32 :global)))
                         (setf (aref out 0) (global-id 0)))")))
    (signals gpu:kernel-type-error
      (macroexpand-1 (read-from-string "(define-kernel broken ((out (ptr nonsense :global))) nil)")))))

(test kernel-macros-are-usable-in-the-file-that-defines-them
  (let ((source (merge-pathnames "cl-gpu-kernel-macro-test.lisp" (uiop:temporary-directory))))
    (remhash (intern "TWICE-FOR-FILE" '#:cl-gpu/tests/kernel) gpu::*kernel-macros*)
    (unwind-protect
         (progn
           (with-open-file (out source :direction :output :if-exists :supersede)
             (write-string "(in-package #:cl-gpu/tests/kernel)
(define-kernel-macro twice-for-file (x) `(* 2 ,x))
(define-kernel doubled ((out (ptr f32 :global)))
  (setf (aref out 0) (twice-for-file 2.0)))"
                           out))
           (multiple-value-bind (fasl warnings failure)
               (compile-file source :output-file (merge-pathnames "cl-gpu-kernel-macro-test.fasl"
                                                                  (uiop:temporary-directory)))
             (declare (ignore warnings))
             (is (not failure))
             (is (and fasl (load fasl)))
             (is (gpu:find-kernel (intern "DOUBLED" '#:cl-gpu/tests/kernel)))))
      (ignore-errors (delete-file source)))))
