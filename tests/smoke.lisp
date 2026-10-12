(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(test system-loads
  (is (find-package '#:cl-gpu)))
