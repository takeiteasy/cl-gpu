(defpackage #:cl-gpu/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:gpu #:cl-gpu))
  (:export #:run-tests))

;;; Package that kernel text is read in.
(defpackage #:cl-gpu/tests/kernel
  (:use #:cl #:cl-gpu))

(in-package #:cl-gpu/tests)

(def-suite :cl-gpu)

(defun run-tests ()
  (run! :cl-gpu))

(defun read-forms (text)
  (gpu:read-kernel-from-string text :package '#:cl-gpu/tests/kernel))

(defun read-one (text)
  (first (read-forms text)))

(defun ksym (name)
  (intern (string-upcase name) '#:cl-gpu/tests/kernel))

(defun walk-text (text &key variables)
  (gpu:walk-kernel (read-one text) :variables (mapcar #'ksym variables)))

(defun nodes-of-type (type node)
  (let ((found '()))
    (gpu:map-nodes (lambda (child) (when (typep child type) (push child found)))
                   node)
    (nreverse found)))

(defun error-position (condition)
  (let ((location (gpu:kernel-error-location condition)))
    (list (gpu:source-location-line location)
          (gpu:source-location-column location))))

(defmacro error-position-of (type &body body)
  "Run BODY, which must signal TYPE, and return the (line column) of the error."
  `(handler-case (progn ,@body (error "no error signalled"))
     (,type (condition) (error-position condition))))
