(defpackage #:cl-gpu/elementwise
  (:use #:cl)
  (:shadow #:min #:max #:abs #:sqrt #:exp #:log #:sin #:cos #:tanh)
  (:local-nicknames (#:gpu #:cl-gpu))
  (:import-from #:cl-gpu #:gpu-fail #:check-view #:shape-size #:view-element
                #:view-root #:view-base #:unique-view-p #:views-overlap-p)
  (:export #:add #:add! #:subtract #:subtract! #:multiply #:multiply! #:divide #:divide!
           #:min #:min! #:max #:max! #:negate #:negate! #:abs #:abs!
           #:sqrt #:sqrt! #:reciprocal #:reciprocal! #:exp #:exp! #:log #:log!
           #:sin #:sin! #:cos #:cos! #:tanh #:tanh! #:sigmoid #:sigmoid!
           #:silu #:silu! #:gelu #:gelu! #:clamp #:clamp! #:compare #:compare!
           #:select #:select! #:convert #:convert!))
