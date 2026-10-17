(defpackage #:cl-gpu/reductions
  (:use #:cl #:cl-gpu)
  (:import-from #:cl-gpu #:gpu-fail #:shape-size #:row-strides #:view-bounds #:check-view
                #:view-element #:float-view-element #:view-root #:view-base
                #:map-view-offsets #:unique-view-p #:views-overlap-p)
  (:export #:strided-view #:strided-view-p #:make-strided-view
           #:strided-view-data #:strided-view-shape #:strided-view-strides #:strided-view-offset
           #:sum #:sum! #:mean #:mean! #:prod #:prod! #:minimum #:minimum! #:maximum #:maximum!
           #:argmin #:argmin! #:argmax #:argmax! #:softmax #:softmax! #:rmsnorm #:rmsnorm!))
