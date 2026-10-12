(defpackage #:cl-gpu/tests
  (:use #:cl #:fiveam)
  (:export #:run-tests))

(in-package #:cl-gpu/tests)

(def-suite :cl-gpu)

(defun run-tests ()
  (run! :cl-gpu))
