// Versioned fixed-order FP32 categorical selection. No atomics or Torch RNG.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ uint64_t sm64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

extern "C" __global__ void categorical_kernel(const float* p, int64_t nr, int64_t nc,
    int64_t s0, int64_t s1, uint64_t key, int64_t* out, int32_t* ubits,
    float* mass_out, float* target_out, int32_t* status) {
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    const uint64_t word = sm64(key ^ sm64((uint64_t)r));
    const uint32_t bits = (uint32_t)(word >> 40);
    ubits[r] = (int32_t)bits;
    const float u = __fmul_rn((float)bits, 0x1p-24f);
    float total = 0.0f;
    int64_t last_positive = -1;
    int code = 0;
    for (int64_t c = 0; c < nc; ++c) {
        const float v = p[r*s0+c*s1];
        if (!isfinite(v) || v < 0.0f) code = 1;
        if (v > 0.0f) last_positive = c;
        total = __fadd_rn(total, v);
    }
    if (!isfinite(total) || !(total > 0.0f) || last_positive < 0) code = 2;
    mass_out[r] = total;
    const float target = __fmul_rn(u, total);
    target_out[r] = target;
    int64_t selected = last_positive;
    if (!code) {
        float cumulative = 0.0f;
        for (int64_t c = 0; c < nc; ++c) {
            cumulative = __fadd_rn(cumulative, p[r*s0+c*s1]);
            // Strict boundary: U=0 never selects a leading zero-weight bin.
            if (cumulative > target) { selected = c; break; }
        }
    }
    out[r] = code ? -1 : selected;
    status[r] = code;
}

extern "C" int affiguide_portable_categorical_f32_v1(
    const float* p, int64_t nr, int64_t nc, int64_t s0, int64_t s1,
    uint64_t key, int64_t* out, int32_t* ubits, float* mass, float* target,
    int32_t* status, void* stream) {
    if (nr <= 0 || nc <= 0 || s0 < 0 || s1 <= 0) return (int)cudaErrorInvalidValue;
    categorical_kernel<<<(unsigned int)((nr+127)/128), 128, 0,
                        (cudaStream_t)stream>>>(p,nr,nc,s0,s1,key,out,ubits,mass,target,status);
    return (int)cudaPeekAtLastError();
}
