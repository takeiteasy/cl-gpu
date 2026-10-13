(in-package #:cl-gpu)

(defvar *current-source* nil
  "Source of the nearest enclosing form that has one.")

(defstruct (source-location (:constructor make-source-location
                                (file start end line column)))
  "A character range in kernel text. LINE and COLUMN are 1-based and locate START."
  file start end line column)

(defmethod print-object ((location source-location) stream)
  (format stream "~@[~a:~]~d:~d"
          (source-location-file location)
          (source-location-line location)
          (source-location-column location)))

(define-condition kernel-error (error)
  ((message :initarg :message :reader kernel-error-message)
   (form :initarg :form :initform nil :reader kernel-error-form)
   (location :initarg :location :initform nil :reader kernel-error-location))
  (:report (lambda (condition stream)
            (format stream "~@[~a: ~]~a"
                    (kernel-error-location condition)
                    (kernel-error-message condition)))))

(define-condition kernel-read-error (kernel-error) ())

(define-condition kernel-syntax-error (kernel-error) ())

(define-condition kernel-type-error (kernel-error) ())
