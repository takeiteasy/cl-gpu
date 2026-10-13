(defpackage #:cl-gpu/blas
  (:use #:cl #:cl-gpu)
  (:import-from #:cl-gpu #:gpu-fail)
  (:export #:saxpy #:sscal #:scopy #:sswap
           #:sdot #:snrm2 #:sasum #:isamax))
