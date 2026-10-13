;;;; cl-gpu.asd

(asdf:defsystem "cl-gpu"
  :description "S-expression GPU kernel DSL with Metal, CUDA and OpenCL backends"
  :author "George Watson <gigolo@hotmail.co.uk>"
  :license "GPL-3.0-only"
  :version "0.0.0"
  :depends-on ("eclector" "eclector-concrete-syntax-tree" "concrete-syntax-tree")
  :serial t
  :components ((:module "src"
                :serial t
                :components ((:file "package")
                             (:file "conditions")
                             (:file "types")
                             (:file "ast")
                             (:file "reader")
                             (:file "env")
                             (:file "walker")
                             (:file "macros")
                             (:file "ir")
                             (:file "intrinsics")
                             (:file "lower")
                             (:file "printer")
                             (:file "msl"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/tests"))))

(asdf:defsystem "cl-gpu/tests"
  :depends-on ("cl-gpu" "fiveam")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/smoke")
               (:file "tests/reader")
               (:file "tests/walker")
               (:file "tests/types")
               (:file "tests/intrinsics")
               (:file "tests/lower")
               (:file "tests/printer")
               (:file "tests/msl"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/tests :run-tests)
               (error "cl-gpu tests failed"))))
