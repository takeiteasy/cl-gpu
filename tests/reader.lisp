(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(test reader-locates-nested-forms
  (let* ((cst (read-one (format nil "(let ((a 1))~%  (foo~%   a))")))
         (location (cst:source (cst:third cst))))
    (is (equal '(2 3) (list (gpu:source-location-line location)
                            (gpu:source-location-column location))))
    (is (= 15 (gpu:source-location-start location)))
    (is (= 25 (gpu:source-location-end location)))))

(test reader-reads-every-form
  (is (= 3 (length (read-forms "a b (c d)")))))

(test reader-error-reports-position
  (is (equal '(2 9) (error-position-of gpu:kernel-read-error
                      (read-forms (format nil "(let ((a 1))~%  (foo a")))))
  (is (= 2 (first (error-position-of gpu:kernel-read-error
                    (read-forms (format nil "(a~%  #<bad)")))))))

(test reader-error-report-includes-position
  (handler-case (gpu:read-kernel-from-string (format nil "(a~%  (b") :file "k.lisp")
    (gpu:kernel-read-error (condition)
      (is (eql 0 (search "k.lisp:2:5: " (princ-to-string condition)))))))

(test reader-reads-files
  (uiop:with-temporary-file (:stream out :pathname path)
    (write-string (format nil "(a)~%(b)") out)
    (finish-output out)
    (let* ((forms (gpu:read-kernel-file path :package '#:cl-gpu/tests/kernel))
           (location (cst:source (second forms))))
      (is (= 2 (length forms)))
      (is (= 2 (gpu:source-location-line location)))
      (is (equal (namestring path) (gpu:source-location-file location))))))
