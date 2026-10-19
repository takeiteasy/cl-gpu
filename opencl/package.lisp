(defpackage #:cl-gpu/opencl
  (:use #:cl #:cl-gpu/transfer)
  (:local-nicknames (#:gpu #:cl-gpu) (#:bt #:bordeaux-threads))
  (:export #:available-devices #:open-device #:default-device #:device-name #:device-extensions
           #:opencl-device #:opencl-error #:opencl-compile-error #:release))
