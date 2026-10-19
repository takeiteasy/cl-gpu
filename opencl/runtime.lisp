(in-package #:cl-gpu/opencl)

(defconstant +device-type-all+ #xffffffff)
(defconstant +device-available+ #x1027)
(defconstant +device-version+ #x102f)
(defconstant +device-name+ #x102b)
(defconstant +device-extensions+ #x1030)
(defconstant +device-type+ #x1000)
(defconstant +event-execution-status+ #x11d3)
(defconstant +device-local-memory+ #x1023)
(defconstant +device-base-alignment+ #x1019)
(defconstant +memory-read-write-copy-host+ #x21)
(defconstant +program-build-log+ #x1183)
(defconstant +device-max-dimensions+ #x1003)
(defconstant +device-max-item-sizes+ #x1005)
(defconstant +device-max-group-size+ #x1004)
(defconstant +kernel-max-group-size+ #x11b0)
(defconstant +buffer-region+ #x1220)
(defconstant +kernel-local-memory+ #x11b2)

(define-condition opencl-error (gpu:gpu-error) ())
(define-condition opencl-compile-error (opencl-error) ())

(defun fail (control &rest arguments)
  (error 'opencl-error :message (apply #'format nil control arguments)))

(defun check (status operation)
  (unless (zerop status) (fail "~a failed (OpenCL ~d)" operation status)))

(defmacro created ((status) &body body)
  `(cffi:with-foreign-object (,status :int32)
     (let ((result (progn ,@body)))
       (check (cffi:mem-ref ,status :int32) "create resource")
       result)))

(defstruct device-info id name extensions kind)

(defclass opencl-device ()
  ((info :initarg :info :reader device-info)
   (context :initarg :context :accessor device-context)
   (queue :initarg :queue :accessor device-queue)
   ;; TODO: one queue lock serializes encoding; per-thread kernels/queues if throughput matters (#88).
   (lock :initform (bt:make-lock "OpenCL queue") :reader device-lock)
   (events :initform nil :accessor device-events)
   (kernels :initform nil :accessor device-kernels)))

(defstruct buffer handle device size)
(defstruct prepared handle device local-memory)

(defun device-id (device)
  (device-info-id (if (typep device 'opencl-device) (device-info device) device)))

(defun device-name (device)
  (device-info-name (if (typep device 'opencl-device) (device-info device) device)))

(defun device-extensions (device)
  (copy-list (device-info-extensions (if (typep device 'opencl-device) (device-info device) device))))

(defun info-number (device parameter type)
  (cffi:with-foreign-object (value type)
    (check (%get-device-info (device-id device) parameter (cffi:foreign-type-size type) value (cffi:null-pointer)) "device info")
    (cffi:mem-ref value type)))

(defun info-string (id parameter)
  (cffi:with-foreign-object (size :size)
    (check (%get-device-info id parameter 0 (cffi:null-pointer) size) "device info size")
    (cffi:with-foreign-object (text :char (cffi:mem-ref size :size))
      (check (%get-device-info id parameter (cffi:mem-ref size :size) text (cffi:null-pointer)) "device info")
      (cffi:foreign-string-to-lisp text))))

(defun available-devices ()
  "Descriptions of available OpenCL 1.2 devices, in platform order."
  (cffi:with-foreign-object (count :uint32)
    (let ((status (%get-platform-ids 0 (cffi:null-pointer) count)))
      (when (= status -1001) (return-from available-devices nil))
      (check status "enumerate platforms"))
    (let ((n (cffi:mem-ref count :uint32)) (result nil))
      (cffi:with-foreign-object (platforms :pointer (max 1 n))
        (when (plusp n) (check (%get-platform-ids n platforms (cffi:null-pointer)) "enumerate platforms"))
        (dotimes (i n)
          (let* ((platform (cffi:mem-aref platforms :pointer i))
                 (status (%get-device-ids platform +device-type-all+ 0 (cffi:null-pointer) count)))
            (unless (= status -1)
              (check status "enumerate devices")
              (let ((m (cffi:mem-ref count :uint32)))
                (cffi:with-foreign-object (ids :pointer (max 1 m))
                  (when (plusp m) (check (%get-device-ids platform +device-type-all+ m ids (cffi:null-pointer)) "enumerate devices"))
                  (dotimes (j m)
                    (let* ((id (cffi:mem-aref ids :pointer j))
                           (version (info-string id +device-version+)))
                      (unless (or (search "OpenCL 1.0 " version) (search "OpenCL 1.1 " version))
                        (let ((info (make-device-info :id id :name (info-string id +device-name+)
                                                      :extensions (uiop:split-string (info-string id +device-extensions+) :separator " "))))
                          (setf (device-info-kind info) (info-number info +device-type+ :uint64))
                          (when (plusp (info-number info +device-available+ :uint32))
                            (push info result))))))))))))
      (nreverse result))))

(defun open-device (info)
  "Open an AVAILABLE-DEVICES description with a context and ordered queue."
  (let ((context nil) (done nil))
    (unwind-protect
         (cffi:with-foreign-object (ids :pointer)
           (setf (cffi:mem-ref ids :pointer) (device-id info)
                 context (created (status)
                           (%create-context (cffi:null-pointer) 1 ids (cffi:null-pointer) (cffi:null-pointer) status)))
           (let ((queue (created (status) (%create-command-queue context (device-id info) 0 status))))
             (setf done t)
             (make-instance 'opencl-device :info info :context context :queue queue)))
      (when (and context (not done)) (%release-context context)))))

(defvar *default-device* nil)
(defvar *default-lock* (bt:make-lock "OpenCL default"))

(defun default-device ()
  (bt:with-lock-held (*default-lock*)
    (or *default-device*
        (let* ((devices (available-devices))
               (info (or (find-if (lambda (d) (logtest 4 (device-info-kind d))) devices)
                         (find-if (lambda (d) (logtest 2 (device-info-kind d))) devices))))
          (unless info (fail "no OpenCL 1.2 GPU or CPU device"))
          (setf *default-device* (open-device info))))))

(defun live-queue (device)
  (or (device-queue device) (fail "OpenCL device was released")))

(defun reap-events (device &optional wait)
  (let ((failure nil) (pending nil))
    (when wait
      (let ((status (%finish (live-queue device))))
        (unless (zerop status) (setf failure status))))
    (dolist (event (reverse (device-events device)))
      (cffi:with-foreign-object (value :int32)
        (check (%get-event-info event +event-execution-status+ 4 value (cffi:null-pointer)) "event status")
        (let ((status (cffi:mem-ref value :int32)))
          (cond ((plusp status) (push event pending))
                (t (%release-event event)
                   (when (and (minusp status) (not failure)) (setf failure status)))))))
    (setf (device-events device) pending)
    (when failure (check failure "queued execution"))))

(defgeneric release (object))

(defmethod release ((object buffer))
  (when (buffer-handle object)
    (%release-mem-object (buffer-handle object))
    (setf (buffer-handle object) nil))
  nil)

(defmethod release ((device opencl-device))
  (unwind-protect
       (bt:with-lock-held ((device-lock device))
         (when (device-queue device)
           (unwind-protect (reap-events device t)
             (mapc #'%release-event (device-events device))
             (setf (device-events device) nil)
             (dolist (kernel (device-kernels device))
               (%release-kernel (prepared-handle kernel))
               (setf (prepared-handle kernel) nil))
             (setf (device-kernels device) nil)
             (%release-command-queue (device-queue device))
             (%release-context (device-context device))
             (setf (device-queue device) nil (device-context device) nil))))
    (bt:with-lock-held (*default-lock*)
      (when (eq device *default-device*) (setf *default-device* nil))))
  nil)

(defmethod gpu:device-target ((device opencl-device)) :opencl)
(defmethod gpu:device-shared-memory-limit ((device opencl-device))
  (info-number device +device-local-memory+ :uint64))
(defmethod gpu:device-offset-alignment ((device opencl-device) space)
  (declare (ignore space))
  (ceiling (info-number device +device-base-alignment+ :uint32) 8))

(defmethod gpu:device-allocate ((device opencl-device) bytes)
  (bt:with-lock-held ((device-lock device))
    (live-queue device)
    ;; TODO: O(n) host staging; reuse storage or device-side zeroing for large allocations (#88).
    (let ((host (cffi:foreign-alloc :uint8 :count (max 1 bytes) :initial-element 0)))
      (unwind-protect
           (make-buffer :device device :size bytes
                        :handle (created (status)
                                  (%create-buffer (device-context device) +memory-read-write-copy-host+ (max 1 bytes) host status)))
        (cffi:foreign-free host)))))

(defmethod gpu:device-free ((device opencl-device) buffer) (release buffer))

(defun transfer (device buffer sequence type offset writep)
  (unless (and (eq device (buffer-device buffer)) (buffer-handle buffer))
    (fail "buffer is released or belongs to another device"))
  (multiple-value-bind (name count stride) (element-layout type)
    (unless (and (integerp offset) (not (minusp offset)) (zerop (mod (length sequence) count)))
      (fail "invalid transfer offset or component count"))
    (let* ((bytes (* (/ (length sequence) count) stride))
           (width (element-bytes name)))
      (when (> (+ offset bytes) (buffer-size buffer)) (fail "transfer exceeds buffer"))
      (bt:with-lock-held ((device-lock device))
        (reap-events device t)
        (when (plusp bytes)
          ;; TODO: O(n) staging and per-value encoding; bulk copies and reusable storage (#88).
          (cffi:with-foreign-pointer (host bytes)
            (when writep
              (cffi:foreign-funcall "memset" :pointer host :int 0 :size bytes :pointer)
              (let ((next (component-offsets count stride width)))
                (map nil (lambda (value) (write-element host name value (funcall next))) sequence)))
            (check (funcall (if writep #'%enqueue-write-buffer #'%enqueue-read-buffer)
                            (live-queue device) (buffer-handle buffer) 1 offset bytes host
                            0 (cffi:null-pointer) (cffi:null-pointer)) "buffer transfer")
            (unless writep
              (let ((next (component-offsets count stride width))
                    (decode (sequence-decoder sequence name)))
                (map-into sequence (lambda () (funcall decode (cffi:mem-ref host (element-ctype name) (funcall next))))))))))))
  sequence)

(defmethod gpu:device-store ((device opencl-device) buffer sequence type offset)
  (transfer device buffer sequence type offset t))
(defmethod gpu:device-load ((device opencl-device) buffer sequence type offset)
  (transfer device buffer sequence type offset nil))
(defmethod gpu:device-synchronize ((device opencl-device))
  (bt:with-lock-held ((device-lock device)) (reap-events device t))
  nil)

(defun build-log (program device)
  (cffi:with-foreign-object (size :size)
    (check (%get-program-build-info program (device-id device) +program-build-log+ 0 (cffi:null-pointer) size) "build log size")
    (cffi:with-foreign-object (text :char (max 1 (cffi:mem-ref size :size)))
      (check (%get-program-build-info program (device-id device) +program-build-log+ (cffi:mem-ref size :size) text (cffi:null-pointer)) "build log")
      (cffi:foreign-string-to-lisp text))))

(defun compile-program (program device precise-math)
  (let ((status (%build-program program 0 (cffi:null-pointer)
                                (if precise-math "-cl-std=CL1.2" "-cl-std=CL1.2 -cl-fast-relaxed-math")
                                (cffi:null-pointer) (cffi:null-pointer))))
    (unless (zerop status)
      (error 'opencl-compile-error
             :message (format nil "OpenCL build ~d: ~a" status (build-log program device))))))

(defun program-kernel (program device entry-name)
  (cffi:with-foreign-object (status :int32)
    (let ((handle (%create-kernel program entry-name status)))
      (unless (zerop (cffi:mem-ref status :int32))
        (error 'opencl-compile-error
               :message (format nil "kernel ~a failed (~d): ~a"
                                entry-name (cffi:mem-ref status :int32) (build-log program device))))
      (let ((prepared (make-prepared :device device :handle handle)))
        (handler-case
            (progn
              (setf (prepared-local-memory prepared) (kernel-info device prepared +kernel-local-memory+ :uint64))
              prepared)
          (error (condition)
            (%release-kernel handle)
            (error condition)))))))

(defmethod gpu:device-prepare ((device opencl-device) source entry-name &key precise-math)
  (bt:with-lock-held ((device-lock device))
    (live-queue device)
    (dolist (extension '("cl_khr_fp16" "cl_khr_fp64"))
      (when (and (search (format nil "#pragma OPENCL EXTENSION ~a : enable" extension) source)
                 (not (member extension (device-extensions device) :test #'equal)))
        (error 'opencl-compile-error
               :message (format nil "device ~a lacks required extension ~a" (device-name device) extension))))
    (cffi:with-foreign-string (text source)
      (cffi:with-foreign-object (strings :pointer)
        (setf (cffi:mem-ref strings :pointer) text)
        (let ((program (created (status)
                         (%create-program-with-source (device-context device) 1 strings (cffi:null-pointer) status))))
          (unwind-protect
               (progn
                 (compile-program program device precise-math)
                 (let ((prepared (program-kernel program device entry-name)))
                   (push prepared (device-kernels device))
                   prepared))
            (%release-program program)))))))

(defun kernel-info (device prepared parameter type)
  (cffi:with-foreign-object (value type)
    (check (%get-kernel-work-group-info (prepared-handle prepared) (device-id device) parameter
                                       (cffi:foreign-type-size type) value (cffi:null-pointer)) "kernel info")
    (cffi:mem-ref value type)))

(defun check-group (device prepared grid group)
  (when group
    (let ((dimensions (info-number device +device-max-dimensions+ :uint32)))
      (cffi:with-foreign-object (limits :size dimensions)
        (check (%get-device-info (device-id device) +device-max-item-sizes+ (* dimensions (cffi:foreign-type-size :size)) limits (cffi:null-pointer)) "work-item limits")
        (loop for n in grid for g in group for axis from 0
              unless (and (zerop (mod n g)) (< axis dimensions) (<= g (cffi:mem-aref limits :size axis)))
                do (fail "group ~s must divide grid ~s and fit device dimensions" group grid))))
    (when (> (reduce #'* group) (min (info-number device +device-max-group-size+ :size) (kernel-info device prepared +kernel-max-group-size+ :size)))
      (fail "group exceeds device or kernel work-group limit"))))

(defun argument-buffer (device buffer offset)
  (unless (and (eq device (buffer-device buffer)) (buffer-handle buffer))
    (fail "buffer is released or belongs to another device"))
  (unless (and (integerp offset) (<= 0 offset) (< offset (max 1 (buffer-size buffer)))
               (zerop (mod offset (gpu:device-offset-alignment device :global))))
    (fail "invalid or unaligned buffer offset ~s" offset))
  (if (zerop offset) (values (buffer-handle buffer) nil)
      (cffi:with-foreign-object (region :size 2)
        (setf (cffi:mem-aref region :size 0) offset
              (cffi:mem-aref region :size 1) (- (buffer-size buffer) offset))
        (values (created (status) (%create-sub-buffer (buffer-handle buffer) 0 +buffer-region+ region status)) t))))

(defmethod gpu:device-launch ((device opencl-device) prepared arguments grid group)
  (bt:with-lock-held ((device-lock device))
    (live-queue device)
    (unless (and (eq device (prepared-device prepared)) (prepared-handle prepared)) (fail "invalid prepared kernel"))
    (unless (and (= 3 (length grid)) (every (lambda (n) (and (integerp n) (plusp n))) grid)
                 (or (null group) (and (= 3 (length group)) (every (lambda (n) (and (integerp n) (plusp n))) group))))
      (fail "grid and group need three positive dimensions"))
    (reap-events device)
    (check-group device prepared grid group)
    (when (> (+ (prepared-local-memory prepared)
                (loop for argument in arguments when (eq (first argument) :shared) sum (second argument)))
             (gpu:device-shared-memory-limit device))
      (fail "kernel exceeds device local-memory limit"))
    (let ((owned nil))
      (unwind-protect
           (progn
             (loop for argument in arguments for index from 0
                   do (ecase (first argument)
                        (:buffer
                         (multiple-value-bind (handle temporary) (argument-buffer device (second argument) (third argument))
                           (when temporary (push handle owned))
                           (cffi:with-foreign-object (value :pointer)
                             (setf (cffi:mem-ref value :pointer) handle)
                             (check (%set-kernel-arg (prepared-handle prepared) index (cffi:foreign-type-size :pointer) value) "buffer argument"))))
                        (:bytes
                         (multiple-value-bind (pointer size) (pack-bytes (second argument) (third argument))
                           (unwind-protect (check (%set-kernel-arg (prepared-handle prepared) index size pointer) "value argument")
                             (cffi:foreign-free pointer))))
                        (:shared
                         (check (%set-kernel-arg (prepared-handle prepared) index (second argument) (cffi:null-pointer)) "local argument"))))
             (when (> (kernel-info device prepared +kernel-local-memory+ :uint64) (gpu:device-shared-memory-limit device))
               (fail "kernel exceeds device local-memory limit"))
             (cffi:with-foreign-objects ((global :size 3) (local :size 3) (event :pointer))
               (loop for n in grid for i from 0 do (setf (cffi:mem-aref global :size i) n))
               (loop for n in group for i from 0 do (setf (cffi:mem-aref local :size i) n))
               (check (%enqueue-nd-range-kernel (device-queue device) (prepared-handle prepared) 3
                                               (cffi:null-pointer) global (if group local (cffi:null-pointer))
                                               0 (cffi:null-pointer) event) "enqueue kernel")
               (push (cffi:mem-ref event :pointer) (device-events device))
               (check (%flush (device-queue device)) "flush queue")))
        (mapc #'%release-mem-object owned))))
  nil)

(gpu:register-backend :opencl #'default-device)
