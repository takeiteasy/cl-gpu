(defpackage #:cl-gpu/blas
  (:use #:cl #:cl-gpu)
  (:import-from #:cl-gpu #:gpu-fail)
  (:export #:saxpy #:sscal #:scopy #:sswap
           #:sdot #:snrm2 #:sasum #:isamax
           #:make-matrix-view #:matrix-subview #:matrix-view-p
           #:matrix-view-data #:matrix-view-rows #:matrix-view-cols #:matrix-view-layout
           #:matrix-view-leading-dimension #:matrix-view-row-stride #:matrix-view-column-stride
           #:matrix-view-offset
           #:sgemv #:sgemm #:sgemm-batch-strided))
