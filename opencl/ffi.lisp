(in-package #:cl-gpu/opencl)

(cffi:define-foreign-library opencl-library
  (:darwin (:framework "OpenCL"))
  (:unix (:or "libOpenCL.so.1" "libOpenCL.so")))
(cffi:use-foreign-library opencl-library)

(defmacro defclfun ((foreign name) result &rest arguments)
  (let ((raw (gensym (string name)))
        (names (mapcar #'first arguments)))
    `(progn
       (cffi:defcfun (,foreign ,raw) ,result ,@arguments)
       (defun ,name ,names
         (float-features:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow :inexact)
           (,raw ,@names))))))

(defclfun ("clGetPlatformIDs" %get-platform-ids) :int32
  (count :uint32) (platforms :pointer) (found :pointer))
(defclfun ("clGetDeviceIDs" %get-device-ids) :int32
  (platform :pointer) (kind :uint64) (count :uint32) (devices :pointer) (found :pointer))
(defclfun ("clGetDeviceInfo" %get-device-info) :int32
  (device :pointer) (param :uint32) (size :size) (value :pointer) (found :pointer))
(defclfun ("clCreateContext" %create-context) :pointer
  (properties :pointer) (count :uint32) (devices :pointer) (notify :pointer) (data :pointer) (status :pointer))
(defclfun ("clCreateCommandQueue" %create-command-queue) :pointer
  (context :pointer) (device :pointer) (properties :uint64) (status :pointer))
(defclfun ("clCreateBuffer" %create-buffer) :pointer
  (context :pointer) (flags :uint64) (size :size) (host :pointer) (status :pointer))
(defclfun ("clCreateSubBuffer" %create-sub-buffer) :pointer
  (buffer :pointer) (flags :uint64) (kind :uint32) (region :pointer) (status :pointer))
(defclfun ("clCreateProgramWithSource" %create-program-with-source) :pointer
  (context :pointer) (count :uint32) (strings :pointer) (lengths :pointer) (status :pointer))
(defclfun ("clBuildProgram" %build-program) :int32
  (program :pointer) (count :uint32) (devices :pointer) (options :string) (notify :pointer) (data :pointer))
(defclfun ("clGetProgramBuildInfo" %get-program-build-info) :int32
  (program :pointer) (device :pointer) (param :uint32) (size :size) (value :pointer) (found :pointer))
(defclfun ("clCreateKernel" %create-kernel) :pointer
  (program :pointer) (name :string) (status :pointer))
(defclfun ("clSetKernelArg" %set-kernel-arg) :int32
  (kernel :pointer) (index :uint32) (size :size) (value :pointer))
(defclfun ("clGetKernelWorkGroupInfo" %get-kernel-work-group-info) :int32
  (kernel :pointer) (device :pointer) (param :uint32) (size :size) (value :pointer) (found :pointer))
(defclfun ("clEnqueueNDRangeKernel" %enqueue-nd-range-kernel) :int32
  (queue :pointer) (kernel :pointer) (dimensions :uint32) (offset :pointer) (grid :pointer) (group :pointer) (count :uint32) (events :pointer) (event :pointer))
(defclfun ("clGetEventInfo" %get-event-info) :int32
  (event :pointer) (param :uint32) (size :size) (value :pointer) (found :pointer))
(defclfun ("clEnqueueWriteBuffer" %enqueue-write-buffer) :int32
  (queue :pointer) (buffer :pointer) (blocking :uint32) (offset :size) (size :size) (host :pointer) (count :uint32) (events :pointer) (event :pointer))
(defclfun ("clEnqueueReadBuffer" %enqueue-read-buffer) :int32
  (queue :pointer) (buffer :pointer) (blocking :uint32) (offset :size) (size :size) (host :pointer) (count :uint32) (events :pointer) (event :pointer))
(defclfun ("clReleaseContext" %release-context) :int32
  (object :pointer))
(defclfun ("clReleaseCommandQueue" %release-command-queue) :int32
  (object :pointer))
(defclfun ("clReleaseMemObject" %release-mem-object) :int32
  (object :pointer))
(defclfun ("clReleaseProgram" %release-program) :int32
  (object :pointer))
(defclfun ("clReleaseKernel" %release-kernel) :int32
  (object :pointer))
(defclfun ("clReleaseEvent" %release-event) :int32
  (object :pointer))
(defclfun ("clFinish" %finish) :int32
  (queue :pointer))
(defclfun ("clFlush" %flush) :int32
  (queue :pointer))
