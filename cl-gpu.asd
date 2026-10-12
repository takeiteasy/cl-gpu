;;;; cl-gpu.asd

(asdf:defsystem "cl-gpu"
  :description "S-expression GPU kernel DSL with Metal, CUDA and OpenCL backends"
  :author "George Watson <gigolo@hotmail.co.uk>"
  :license "GPL-3.0-only"
  :version "0.0.0"
  :serial t
  :components ((:file "package"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/tests"))))

(asdf:defsystem "cl-gpu/tests"
  :depends-on ("cl-gpu" "fiveam")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/smoke"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/tests :run-tests)
               (error "cl-gpu tests failed"))))
