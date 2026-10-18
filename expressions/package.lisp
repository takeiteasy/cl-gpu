(defpackage #:cl-gpu/expressions
  (:use #:cl)
  (:local-nicknames (#:gpu #:cl-gpu) (#:ew #:cl-gpu/elementwise)
                    (#:red #:cl-gpu/reductions) (#:bt #:bordeaux-threads))
  (:export #:expression-plan #:make-expression-plan #:expression-result-info
           #:execute-expression #:execute-expression!))
