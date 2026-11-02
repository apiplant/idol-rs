// Fused elementwise kernels for the denoiser (f16 activations, f32 math). Built to PTX by build.rs and launched
// through candle's CUDA stream. Every kernel replaces a chain of memory-bound candle ops with one pass.
#include <cuda_fp16.h>
#include <curand_kernel.h>

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  v = lane < nw ? red[lane] : 0.f;
  return warp_sum(v);
}

__device__ __forceinline__ float rh(float v) { return __half2float(__float2half(v)); }

// out = LayerNorm(x) * (1 + scale) + shift, one block per row of C (<= 2048) elements.
// m holds the adaLN output [6, C]: shift = m[slot], scale = m[slot + 1].
extern "C" __global__ void mod_ln_f16(const __half* __restrict__ x, const __half* __restrict__ m, __half* __restrict__ out,
                                      int C, int slot, float eps) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  const __half* xr = x + row * C;
  __half* o = out + row * C;
  const __half* shift = m + (size_t)slot * C;
  const __half* scale = m + (size_t)(slot + 1) * C;
  float v[16];
  int n = 0;
  float s = 0.f;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float a = __half2float(xr[i]);
    v[n++] = a;
    s += a;
  }
  const float mean = block_sum(s, red) / C;
  float q = 0.f;
  for (int k = 0; k < n; k++) {
    float d = v[k] - mean;
    q += d * d;
  }
  const float inv = rsqrtf(block_sum(q, red) / C + eps);
  n = 0;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float y = (v[n++] - mean) * inv;
    o[i] = __float2half(y * (1.f + __half2float(scale[i])) + __half2float(shift[i]));
  }
}

// qkv [R, 3, H, 64] f16 (a linear output without its bias) -> out [3, R, H, 64]: q and k get the bias, the rotary
// rotation, an RMS norm over the head and their gamma; v gets the bias. One warp per (row, head), lane = rotary pair.
// cs is [2, R, H, 32] (cos then sin), gq/gk are [H, 64] f32.
extern "C" __global__ void rope_qkv_f16(const __half* __restrict__ qkv, const __half* __restrict__ bias, const float* __restrict__ cs,
                                        const float* __restrict__ gq, const float* __restrict__ gk, __half* __restrict__ out,
                                        int R, int H, float eps) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= R * H) return;
  const int row = warp / H, h = warp % H;
  const size_t HD = (size_t)H * 64;
  const size_t ci = ((size_t)row * H + h) * 32 + lane;
  const float c = cs[ci], s = cs[(size_t)R * H * 32 + ci];
  #pragma unroll
  for (int t = 0; t < 3; t++) {
    const size_t base = ((size_t)row * 3 + t) * HD + (size_t)h * 64 + 2 * lane;
    const size_t bi = t * HD + (size_t)h * 64 + 2 * lane;
    const __half2 xv = *reinterpret_cast<const __half2*>(qkv + base);
    const __half2 bv = *reinterpret_cast<const __half2*>(bias + bi);
    // the linear's output is rounded to f16 with its bias, like the reference
    float x0 = rh(__low2float(xv) + __low2float(bv));
    float x1 = rh(__high2float(xv) + __high2float(bv));
    const size_t oi = (((size_t)t * R + row) * H + h) * 64 + 2 * lane;
    if (t == 2) {
      *reinterpret_cast<__half2*>(out + oi) = __floats2half2_rn(x0, x1);
      continue;
    }
    const float r0 = rh(x0 * c - x1 * s), r1 = rh(x0 * s + x1 * c);
    const float inv = rsqrtf(warp_sum(r0 * r0 + r1 * r1) / 64.f + eps);
    const float* g = (t == 0 ? gq : gk) + (size_t)h * 64 + 2 * lane;
    *reinterpret_cast<__half2*>(out + oi) = __floats2half2_rn(r0 * inv * g[0], r1 * inv * g[1]);
  }
}

// out = x + (y + bias) * gate, with bias [C] and gate = m[slot] ([6, C]); C is even.
extern "C" __global__ void bias_gate_res_f16(const __half* __restrict__ x, const __half* __restrict__ y, const __half* __restrict__ bias,
                                             const __half* __restrict__ m, __half* __restrict__ out, int C, int slot, size_t n2) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const int c = (int)((i * 2) % C);
  const __half2 xv = reinterpret_cast<const __half2*>(x)[i], yv = reinterpret_cast<const __half2*>(y)[i];
  const __half2 bv = *reinterpret_cast<const __half2*>(bias + c);
  const __half2 gv = *reinterpret_cast<const __half2*>(m + (size_t)slot * C + c);
  const float a = rh(__low2float(yv) + __low2float(bv)), b = rh(__high2float(yv) + __high2float(bv));
  reinterpret_cast<__half2*>(out)[i] = __floats2half2_rn(__low2float(xv) + a * __low2float(gv), __high2float(xv) + b * __high2float(gv));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// out = gelu_tanh(x + bias), bias [N], N even.
extern "C" __global__ void bias_gelu_f16(const __half* __restrict__ x, const __half* __restrict__ bias, __half* __restrict__ out, int N, size_t n2) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const int c = (int)((i * 2) % N);
  const __half2 xv = reinterpret_cast<const __half2*>(x)[i];
  const __half2 bv = *reinterpret_cast<const __half2*>(bias + c);
  const float a = rh(__low2float(xv) + __low2float(bv)), b = rh(__high2float(xv) + __high2float(bv));
  reinterpret_cast<__half2*>(out)[i] = __floats2half2_rn(gelu_tanh(a), gelu_tanh(b));
}

// RePo angles: o [R, H, 3] f32 offsets (x, y, z per head), freqs [32] f32 (10 for x, 10 for y, 12 for z) ->
// out [2, R, H, 32] = cos, sin of o[axis] * freq * pi.
extern "C" __global__ void repo_angles_f32(const float* __restrict__ o, const float* __restrict__ freqs, float* __restrict__ out, int R, int H) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)R * H * 32;
  if (i >= total) return;
  const int j = (int)(i % 32);
  const size_t rh_ = i / 32;  // row * H + h
  const int axis = j < 10 ? 0 : (j < 20 ? 1 : 2);
  const float ang = (o[rh_ * 3 + axis] * freqs[j]) * 3.14159265358979323846f;
  float sn, cs;
  sincosf(ang, &sn, &cs);
  out[i] = cs;
  out[total + i] = sn;
}

// PyTorch's CUDA `randn` / `rand` for float32 (aten distribution_elementwise_grid_stride_kernel with unroll 4):
// the same curand Philox4x32-10 calls with the same thread -> element mapping, so launched with torch's grid
// (see `calc_execution_policy`) it reproduces `torch.manual_seed(seed)` streams exactly. `dummy` only carries the
// device for the launcher. kind 0: curand_normal4 (randn), 1: curand_uniform4 (rand, with 1.0 mapped to 0).
extern "C" __global__ void philox_fill_f32(const float* __restrict__ dummy, float* __restrict__ out, long long numel,
                                           unsigned long long seed, unsigned long long offset, int kind) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
  const long long grid_threads = (long long)blockDim.x * gridDim.x;
  const long long stride = grid_threads * 4;
  const long long rounded = ((numel - 1) / stride + 1) * stride;
  for (long long base = idx; base < rounded; base += stride) {
    const float4 r = kind == 0 ? curand_normal4(&state) : curand_uniform4(&state);
    const float v[4] = {r.x, r.y, r.z, r.w};
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
      const long long li = base + grid_threads * ii;
      if (li < numel) {
        float x = v[ii];
        if (kind == 1 && x == 1.0f) x = 0.0f;
        out[li] = x;
      }
    }
  }
}
