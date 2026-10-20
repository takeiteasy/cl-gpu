(in-package #:cl-gpu/elementwise)

(defparameter *elements* '(gpu:f16 gpu:bf16 gpu:f32 gpu:f64 gpu:i8 gpu:u8 gpu:i16 gpu:u16
                          gpu:i32 gpu:u32 gpu:i64 gpu:u64))

(defun float-element-p (element)
  (member element '(gpu:f16 gpu:bf16 gpu:f32 gpu:f64)))

(defun element-code (element)
  (ecase element
    (gpu:f16 16) (gpu:bf16 17) (gpu:f32 32) (gpu:f64 64)
    (gpu:i8 108) (gpu:u8 208) (gpu:i16 116) (gpu:u16 216)
    (gpu:i32 132) (gpu:u32 232) (gpu:i64 164) (gpu:u64 264)))

(defparameter *conversion-helper*
"inline ulong ew_mask(uint bits) { return bits == 64 ? ~0ul : (1ul << bits) - 1ul; }
inline ulong ew_round(ulong value, int shift, bool negative, uint mode) {
    if (shift <= 0) return value << (uint)(-shift);
    ulong q = shift >= 64 ? 0ul : value >> (uint)(shift);
    ulong r = shift >= 64 ? value : value & ew_mask((uint)(shift));
    bool up = false;
    if (mode == 0 && shift <= 64) {
        ulong midpoint = 1ul << (uint)(shift - 1);
        up = r > midpoint || (r == midpoint && (q & 1ul));
    } else if (mode == 2) up = negative && r != 0;
    else if (mode == 3) up = !negative && r != 0;
    return q + (ulong)(up);
}
inline ulong ew_overflow(bool negative, uint mode, uint frac, uint ebits) {
    ulong inf = ew_mask(ebits) << frac;
    return mode == 0 || (mode == 2 && negative) || (mode == 3 && !negative)
        ? inf : inf - 1ul;
}
inline ulong cl_gpu_ew_convert(ulong raw, ulong source, ulong destination, ulong rounding) {
    if (source == destination) return raw;
    bool floating = source < 100;
    bool negative = false;
    ulong sig = 0ul, payload = 0ul;
    int exponent = 0;
    uint source_frac = 0;
    bool infinity = false, nan = false;
    if (floating) {
        source_frac = source == 64 ? 52 : (source == 32 ? 23 : (source == 16 ? 10 : 7));
        uint ebits = source == 64 ? 11 : (source == 16 ? 5 : 8);
        uint bias = (1u << (ebits - 1)) - 1;
        negative = (raw >> (source_frac + ebits)) != 0;
        ulong field = (raw >> source_frac) & ew_mask(ebits);
        payload = raw & ew_mask(source_frac);
        infinity = field == ew_mask(ebits) && payload == 0;
        nan = field == ew_mask(ebits) && payload != 0;
        sig = payload | (field != 0 ? 1ul << source_frac : 0ul);
        exponent = (int)(field == 0 ? 1ul : field) - (int)(bias) - (int)(source_frac);
    } else {
        uint width = (uint)(source % 100);
        raw &= ew_mask(width);
        negative = source < 200 && (raw & (1ul << (width - 1))) != 0;
        sig = negative ? ((~raw + 1ul) & ew_mask(width)) : raw;
    }
    if (destination >= 100) {
        uint width = (uint)(destination % 100);
        bool signed_output = destination < 200;
        ulong limit = signed_output ? ((1ul << (width - 1)) - (ulong)(!negative)) : ew_mask(width);
        if (nan || (negative && !signed_output)) return 0ul;
        bool overflow = infinity;
        ulong magnitude = sig;
        if (exponent > 0) {
            overflow = overflow || exponent >= 64 || sig > (limit >> (uint)(min(exponent, 63)));
            if (!overflow) magnitude = sig << (uint)(exponent);
        } else magnitude = ew_round(sig, -exponent, negative, (uint)(rounding));
        if (overflow || magnitude > limit) magnitude = limit;
        return (negative ? ~magnitude + 1ul : magnitude) & ew_mask(width);
    }
    uint frac = destination == 64 ? 52 : (destination == 32 ? 23 : (destination == 16 ? 10 : 7));
    uint ebits = destination == 64 ? 11 : (destination == 16 ? 5 : 8);
    int bias = (int)((1u << (ebits - 1)) - 1);
    ulong sign = (ulong)(negative) << (frac + ebits);
    uint mode = floating ? (uint)(rounding) : 0;
    if (nan) {
        ulong p = source_frac > frac ? payload >> (source_frac - frac) : payload << (frac - source_frac);
        return sign | (ew_mask(ebits) << frac) | p | (1ul << (frac - 1));
    }
    if (infinity) return sign | (ew_mask(ebits) << frac);
    if (sig == 0) return sign;
    int leading = 63 - (int)(clz(sig));
    int e = leading + exponent;
    int emin = 1 - bias, emax = bias;
    if (e > emax) return sign | ew_overflow(negative, mode, frac, ebits);
    int unit = max(e, emin) - (int)(frac);
    ulong rounded = ew_round(sig, unit - exponent, negative, mode);
    if (e < emin) return sign | rounded;
    if (rounded == (1ul << (frac + 1))) { rounded >>= 1; ++e; }
    if (e > emax) return sign | ew_overflow(negative, mode, frac, ebits);
    return sign | ((ulong)(e + bias) << frac) | (rounded & ew_mask(frac));
}
")

(gpu::define-intrinsic %convert-bits (:number 4)
  :metal (:helper "cl_gpu_ew_convert")
  :opencl (:helper "cl_gpu_ew_convert"))

(gpu::register-msl-helper "cl_gpu_ew_convert" *conversion-helper*)
(gpu::register-msl-helper
 "cl_gpu_ew_error"
 "inline void cl_gpu_ew_error(device uint* status) {
    atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(status), 1u, memory_order_relaxed);
}")

(defun unsigned-element (element)
  (ecase (gpu:scalar-type-bits (gpu:parse-type element))
    (8 'gpu:u8) (16 'gpu:u16) (32 'gpu:u32) (64 'gpu:u64)))

(defun raw-code (element expression)
  `(gpu:convert gpu:u64 (gpu:bitcast ,(unsigned-element element) ,expression)))

(defun converted-code (source destination expression mode)
  (if (eq source destination) expression
      `(gpu:bitcast ,(conversion-storage destination)
         (gpu:convert ,(unsigned-element destination)
           (%convert-bits ,(raw-code (conversion-storage source) expression)
                          (gpu:convert gpu:u64 ,(element-code source))
                          (gpu:convert gpu:u64 ,(element-code destination))
                          (gpu:convert gpu:u64 ,mode))))))

(gpu::define-intrinsic %record-error (:typed ((gpu:ptr gpu:u32 :global)) nil)
  :impure t :metal (:helper "cl_gpu_ew_error")
  :opencl (:helper "cl_gpu_ew_error"))

(gpu::register-opencl-helper "cl_gpu_ew_convert" *conversion-helper*)

(gpu::register-opencl-helper "cl_gpu_ew_error"
 "inline void cl_gpu_ew_error(__global uint* status) {
    atomic_or((volatile __global uint*)status, 1u);
}")

(defun compute-element (element)
  (if (eq element 'gpu:f64) 'gpu:f64 'gpu:f32))

(defun complex-element-p (element)
  (member element '(gpu:c32 gpu:c64)))

(defun component-element (element)
  (case element (gpu:c32 'gpu:f32) (gpu:c64 'gpu:f64) (otherwise element)))

(defun conversion-storage (element)
  (cond ((member element '(gpu:f16 gpu:bf16)) 'gpu:u16)
        ((complex-element-p element) `(gpu:vec ,(component-element element) 2))
        (t element)))

(defun conversion-root (view)
  (let ((root (view-root view)))
    (if (member (view-element view) '(gpu:f16 gpu:bf16))
        (make-instance 'gpu:gpu-array-view :owner root :offset 0
                       :device (gpu:gpu-array-device root) :length (gpu:gpu-array-length root)
                       :element-type (gpu:parse-type 'gpu:u16))
        (gpu:gpu-array-storage-view root))))

(defun operation-storage (element operation)
  (if (or (eq operation :convert) (complex-element-p element))
      (conversion-storage element) element))

(defun operation-root (view operation)
  (if (or (eq operation :convert) (complex-element-p (view-element view)))
      (conversion-root view) (view-root view)))
