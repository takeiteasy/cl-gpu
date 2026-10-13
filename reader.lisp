(in-package #:cl-gpu)

(defclass kernel-client (eclector.concrete-syntax-tree:cst-client)
  ((file :initarg :file :reader client-file)
   (line-starts :initarg :line-starts :reader client-line-starts)))

(defun line-starts (text)
  (coerce (cons 0 (loop for char across text
                        for index from 1
                        when (char= char #\Newline) collect index))
          'vector))

(defun offset-line-column (line-starts offset)
  "Return the 1-based line and column of OFFSET."
  (let ((line (or (position-if (lambda (start) (> start offset)) line-starts)
                  (length line-starts))))
    (values line (1+ (- offset (aref line-starts (1- line)))))))

(defun client-location (client start end)
  (multiple-value-bind (line column)
      (offset-line-column (client-line-starts client) start)
    (make-source-location (client-file client) start end line column)))

(defmethod eclector.base:make-source-range ((client kernel-client) start end)
  (client-location client start end))

(defun make-kernel-client (text file)
  (make-instance 'kernel-client :file file :line-starts (line-starts text)))

(defun read-kernel-from-string (text &key file (package *package*))
  "Read every form in TEXT into a list of CSTs carrying SOURCE-LOCATIONs."
  (let* ((client (make-kernel-client text file))
         (eclector.reader:*client* client)
         (*package* (find-package package))
         (stream (make-string-input-stream text)))
    (handler-case
        (loop for cst = (eclector.concrete-syntax-tree:read stream nil stream)
              until (eq cst stream)
              collect cst)
      (error (condition)
        (let ((offset (if (typep condition '(or eclector.base:stream-position-reader-error
                                     eclector.base:end-of-file))
                          (+ (eclector.base:stream-position condition)
                             (eclector.base:position-offset condition))
                          (file-position stream))))
          (error 'kernel-read-error
                 :message (princ-to-string condition)
                 :location (client-location client offset offset)))))))

(defun read-kernel (stream &key file (package *package*))
  "Read every form from STREAM; see READ-KERNEL-FROM-STRING."
  (let ((text (with-output-to-string (out)
                (loop for char = (read-char stream nil)
                      while char do (write-char char out)))))
    (read-kernel-from-string text :file file :package package)))

(defun read-kernel-file (pathname &key (package *package*))
  "Read every form from the file at PATHNAME; see READ-KERNEL-FROM-STRING."
  (with-open-file (stream pathname)
    (read-kernel stream :file (namestring pathname) :package package)))
