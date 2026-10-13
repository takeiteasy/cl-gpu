(defpackage #:cl-gpu/metal
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu))
  (:import-from #:cl-gpu #:wait)
  (:export
   #:metal-error #:metal-compile-error #:metal-error-message
   #:default-device #:device-name #:device-max-threads
   #:make-buffer #:buffer-size #:buffer-pointer #:copy-to-buffer #:copy-from-buffer
   #:compile-library #:make-pipeline #:pipeline-max-threads #:pipeline-execution-width
   #:dispatch #:wait #:command-done-p
   #:release #:with-buffer #:with-library #:with-pipeline))
