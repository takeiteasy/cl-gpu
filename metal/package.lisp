(defpackage #:cl-gpu/metal
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu))
  (:export
   #:metal-error #:metal-compile-error #:metal-error-message
   #:default-device #:device-name #:device-max-threads #:device-max-threadgroup-memory
   #:make-buffer #:buffer-size #:buffer-pointer #:copy-to-buffer #:copy-from-buffer
   #:compile-library #:make-pipeline #:pipeline-max-threads #:pipeline-execution-width
   #:dispatch
   #:*use-mps*
   #:release #:with-buffer #:with-library #:with-pipeline))
