(in-package #:cl-gpu/metal)

;;; Apple frameworks raise floating-point exceptions that SBCL would otherwise trap, so every
;;; shim call runs with the traps masked.
(defmacro defshim ((c-name lisp-name) return-type &rest parameters)
  (let ((raw (intern (format nil "~a/RAW" lisp-name)))
        (names (mapcar #'first parameters)))
    `(progn
       (cffi:defcfun (,c-name ,raw) ,return-type ,@parameters)
       (defun ,lisp-name ,names
         (float-features:with-float-traps-masked
             (:overflow :invalid :divide-by-zero :inexact :underflow)
           (,raw ,@names))))))

(defshim ("clgpu_free_string" %free-string) :void (string :pointer))
(defshim ("clgpu_release" %release) :void (object :pointer))

(defshim ("clgpu_device_default" %device-default) :pointer)
(defshim ("clgpu_device_name" %device-name) :pointer (device :pointer))
(defshim ("clgpu_device_max_threads" %device-max-threads) :unsigned-long (device :pointer))
(defshim ("clgpu_device_max_threadgroup_memory" %device-max-threadgroup-memory) :unsigned-long
  (device :pointer))
(defshim ("clgpu_queue_new" %queue-new) :pointer (device :pointer))

(defshim ("clgpu_device_unified_memory" %device-unified-memory) :int (device :pointer))
(defshim ("clgpu_buffer_new" %buffer-new) :pointer
  (device :pointer) (length :unsigned-long) (private :int))
(defshim ("clgpu_buffer_contents" %buffer-contents) :pointer (buffer :pointer))

(defshim ("clgpu_library_new" %library-new) :pointer
  (device :pointer) (source :string) (precise :int) (error :pointer))
(defshim ("clgpu_pipeline_new" %pipeline-new) :pointer
  (device :pointer) (library :pointer) (name :string) (error :pointer))
(defshim ("clgpu_pipeline_max_threads" %pipeline-max-threads) :unsigned-long
  (pipeline :pointer))
(defshim ("clgpu_pipeline_execution_width" %pipeline-execution-width) :unsigned-long
  (pipeline :pointer))

(defshim ("clgpu_stream_new" %stream-new) :pointer (queue :pointer))
(defshim ("clgpu_stream_dispatch" %stream-dispatch) :int
  (stream :pointer) (pipeline :pointer) (count :int) (kinds :pointer) (values :pointer)
  (lengths :pointer) (grid :pointer) (group :pointer) (error :pointer))
(defshim ("clgpu_stream_copy" %stream-copy) :int
  (stream :pointer) (source :pointer) (source-offset :unsigned-long) (target :pointer)
  (target-offset :unsigned-long) (length :unsigned-long) (error :pointer))
(defshim ("clgpu_stream_fill" %stream-fill) :int
  (stream :pointer) (buffer :pointer) (length :unsigned-long) (error :pointer))
(defshim ("clgpu_stream_wait" %stream-wait) :int (stream :pointer) (error :pointer))

(defshim ("clgpu_mps_supported" %mps-supported) :int (device :pointer))
(defshim ("clgpu_mps_gemm" %mps-gemm) :int
  (stream :pointer) (transpose-left :int) (transpose-right :int)
  (m :unsigned-long) (n :unsigned-long) (k :unsigned-long) (alpha :float) (beta :float)
  (count :unsigned-long) (buffers :pointer) (shapes :pointer) (error :pointer))
(defshim ("clgpu_mps_gemv" %mps-gemv) :int
  (stream :pointer) (transpose :int) (rows :unsigned-long) (columns :unsigned-long)
  (alpha :float) (beta :float) (buffers :pointer) (shapes :pointer) (error :pointer))
