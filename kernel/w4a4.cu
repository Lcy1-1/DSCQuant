#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int THREADS = 256;

void check_cuda(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Expected contiguous CUDA tensor");
}
void check_float(const torch::Tensor& x) {
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat,
              "Only FP16 and FP32 inputs are supported");
}

template <typename scalar_t, bool FUSED>
__global__ void quantize_rows(const scalar_t* x, unsigned char* packed,
                             unsigned char* codes, float* scales, int* zeros,
                             int M, int K, int KP, float clip) {
  int row = blockIdx.x, tid = threadIdx.x;
  __shared__ float lower[THREADS], upper[THREADS];
  float lo = 0.f, hi = 0.f;
  if (row < M) {
    for (int k = tid; k < K; k += THREADS) {
      float value = static_cast<float>(x[(int64_t)row * K + k]);
      lo = fminf(lo, value); hi = fmaxf(hi, value);
    }
  }
  lower[tid] = lo; upper[tid] = hi;
  __syncthreads();
  for (int stride = THREADS / 2; stride; stride >>= 1) {
    if (tid < stride) {
      lower[tid] = fminf(lower[tid], lower[tid + stride]);
      upper[tid] = fmaxf(upper[tid], upper[tid + stride]);
    }
    __syncthreads();
  }
  float minimum = lower[0] * clip, maximum = upper[0] * clip;
  float scale = fmaxf(__fdiv_rn(maximum - minimum, 15.f), 1e-8f);
  int zero = max(0, min(15, __float2int_rn(__fdiv_rn(-minimum, scale))));
  if (tid == 0) { scales[row] = scale; zeros[row] = zero; }
  if (FUSED) {
    for (int pair = tid; pair < KP / 2; pair += THREADS) {
      int k = pair * 2, q0 = zero, q1 = zero;
      if (row < M && k < K)
        q0 = max(0, min(15, __float2int_rn(__fdiv_rn(static_cast<float>(x[(int64_t)row*K+k]), scale) + zero)));
      if (row < M && k + 1 < K)
        q1 = max(0, min(15, __float2int_rn(__fdiv_rn(static_cast<float>(x[(int64_t)row*K+k+1]), scale) + zero)));
      packed[(int64_t)row * (KP/2) + pair] = ((q0 - 8) & 15) | (((q1 - 8) & 15) << 4);
    }
  } else {
    for (int k = tid; k < KP; k += THREADS) {
      int q = zero;
      if (row < M && k < K)
        q = max(0, min(15, __float2int_rn(__fdiv_rn(static_cast<float>(x[(int64_t)row*K+k]), scale) + zero)));
      codes[(int64_t)row * KP + k] = q;
    }
  }
}

__global__ void pack_rows(const unsigned char* codes, unsigned char* packed, int64_t count) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) packed[i] = ((codes[2*i] - 8) & 15) | (((codes[2*i+1] - 8) & 15) << 4);
}

template <typename scalar_t>
__device__ scalar_t restore_value(int value, int row, int col, const float* sx,
                                 const int* zx, const float* sw, const int* sums,
                                 const scalar_t* bias) {
  int corrected = value + (8 - zx[row]) * sums[col];
  float result = static_cast<float>(corrected) * sx[row] * sw[col];
  if (bias) result += static_cast<float>(bias[col]);
  return static_cast<scalar_t>(result);
}

template <typename scalar_t, bool FUSED>
__global__ void tensor_gemm(const unsigned char* a, const unsigned char* b,
                            const float* sx, const int* zx, const float* sw,
                            const int* sums, const scalar_t* bias,
                            scalar_t* out, int* raw, int M, int N, int KP) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900
  namespace wmma = nvcuda::wmma;
  using s4 = wmma::experimental::precision::s4;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int row0 = blockIdx.y * 8, col0 = (blockIdx.x * 4 + warp) * 8;
  if (col0 >= N) return;
  __shared__ __align__(32) unsigned char as[4][128];
  __shared__ __align__(32) unsigned char bs[4][128];
  __shared__ __align__(32) int cs[4][64];
  wmma::fragment<wmma::matrix_a, 8, 8, 32, s4, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 8, 8, 32, s4, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 8, 8, 32, int> cf;
  wmma::fill_fragment(cf, 0);
  for (int k = 0; k < KP; k += 32) {
    for (int i = lane; i < 128; i += 32) {
      int r = i / 16, c = i % 16;
      as[warp][i] = a[(int64_t)(row0+r)*(KP/2) + k/2+c];
      bs[warp][i] = b[(int64_t)(col0+r)*(KP/2) + k/2+c];
    }
    __syncwarp();
    wmma::load_matrix_sync(af, reinterpret_cast<const int*>(as[warp]), 32);
    wmma::load_matrix_sync(bf, reinterpret_cast<const int*>(bs[warp]), 32);
    wmma::mma_sync(cf, af, bf, cf);
    __syncwarp();
  }
  wmma::store_matrix_sync(cs[warp], cf, 8, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 64; i += 32) {
    int row = row0 + i/8, col = col0 + i%8;
    if (row < M && col < N) {
      int value = cs[warp][i];
      if (FUSED) out[(int64_t)row*N+col] = restore_value(value,row,col,sx,zx,sw,sums,bias);
      else raw[(int64_t)row*N+col] = value;
    }
  }
#endif
}

__device__ int sign4(int value) { return value >= 8 ? value - 16 : value; }
__device__ int bytes4(const unsigned char* p, bool high) {
  unsigned int result = 0;
  int nibble_start = high ? 4 : 0;
  for (int j = 0; j < 4; ++j) {
    int n = nibble_start + j;
    int value = sign4((p[n/2] >> ((n&1)*4)) & 15);
    result |= static_cast<unsigned int>(value & 255) << (8*j);
  }
  return static_cast<int>(result);
}

template <typename scalar_t, bool FUSED>
__global__ void decode_gemm(const unsigned char* a, const unsigned char* b,
                           const float* sx, const int* zx, const float* sw,
                           const int* sums, const scalar_t* bias,
                           scalar_t* out, int* raw, int M, int N, int KP) {
  int lane = threadIdx.x & 31, col = blockIdx.x*4 + (threadIdx.x>>5), row = blockIdx.y;
  if (col >= N) return;
  int value = 0;
  for (int k = lane*8; k < KP; k += 256) {
    const unsigned char* ap = a + (int64_t)row*(KP/2) + k/2;
    const unsigned char* bp = b + (int64_t)col*(KP/2) + k/2;
    value = __dp4a(bytes4(ap,false), bytes4(bp,false), value);
    value = __dp4a(bytes4(ap,true), bytes4(bp,true), value);
  }
  for (int delta = 16; delta; delta >>= 1) value += __shfl_down_sync(0xffffffff, value, delta);
  if (lane == 0) {
    if (FUSED) out[(int64_t)row*N+col] = restore_value(value,row,col,sx,zx,sw,sums,bias);
    else raw[(int64_t)row*N+col] = value;
  }
}

template <typename scalar_t>
__global__ void epilogue(const int* raw, scalar_t* out, const float* sx, const int* zx,
                         const float* sw, const int* sums, const scalar_t* bias, int M, int N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (int64_t)M*N) out[i] = restore_value(raw[i],i/N,i%N,sx,zx,sw,sums,bias);
}
} // namespace

std::vector<torch::Tensor> quantize_cuda(torch::Tensor x, double clip, bool fused) {
  check_cuda(x); check_float(x);
  TORCH_CHECK(x.dim()==2 && x.size(0)>0 && x.size(0)<=65535 && x.size(1)>0 && x.size(1)<=1048576,
              "Require 1<=M<=65535 and 1<=K<=1048576");
  TORCH_CHECK(clip>=0.5 && clip<=1., "Activation clip must be in [0.5,1]");
  c10::cuda::CUDAGuard guard(x.device());
  int M=x.size(0), K=x.size(1), MP=(M+7)/8*8, KP=(K+63)/64*64;
  auto packed=torch::empty({MP,KP/2},x.options().dtype(at::kByte));
  auto scales=torch::empty({MP},x.options().dtype(at::kFloat));
  auto zeros=torch::empty({MP},x.options().dtype(at::kInt));
  auto codes=fused ? torch::empty({0},x.options().dtype(at::kByte)) : torch::empty({MP,KP},x.options().dtype(at::kByte));
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(),"quantize_w4a4",[&] {
    if (fused) quantize_rows<scalar_t,true><<<MP,THREADS,0,stream>>>(x.data_ptr<scalar_t>(),packed.data_ptr<unsigned char>(),nullptr,scales.data_ptr<float>(),zeros.data_ptr<int>(),M,K,KP,clip);
    else quantize_rows<scalar_t,false><<<MP,THREADS,0,stream>>>(x.data_ptr<scalar_t>(),nullptr,codes.data_ptr<unsigned char>(),scales.data_ptr<float>(),zeros.data_ptr<int>(),M,K,KP,clip);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (!fused) {
    int64_t count=packed.numel();
    pack_rows<<<(count+255)/256,256,0,stream>>>(codes.data_ptr<unsigned char>(),packed.data_ptr<unsigned char>(),count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed,scales,zeros};
}

std::vector<torch::Tensor> matmul_cuda(torch::Tensor a, torch::Tensor b,
    torch::Tensor sx, torch::Tensor zx, torch::Tensor sw, torch::Tensor sums,
    torch::Tensor bias, int64_t M, int64_t N, bool fused, bool force_decode, bool fp32) {
  for (auto t : {a,b,sx,zx,sw,sums,bias}) {
    check_cuda(t); TORCH_CHECK(t.device()==a.device(),"All buffers must be on the same device");
  }
  TORCH_CHECK(a.dim()==2 && b.dim()==2 && a.scalar_type()==at::kByte && b.scalar_type()==at::kByte,"Packed operands must be byte matrices");
  int64_t KP=a.size(1)*2, MP=(M+7)/8*8, NP=(N+7)/8*8;
  TORCH_CHECK(M>0 && M<=65535 && N>0 && KP>0 && KP<=1048576 && KP%64==0,"Invalid dimensions");
  TORCH_CHECK(a.size(0)>=MP && b.size(0)>=NP && b.size(1)==a.size(1),"Invalid padded operand shapes");
  TORCH_CHECK(sx.scalar_type()==at::kFloat && sw.scalar_type()==at::kFloat && zx.scalar_type()==at::kInt && sums.scalar_type()==at::kInt,"Invalid metadata dtypes");
  TORCH_CHECK(sx.numel()>=MP && zx.numel()>=MP && sw.numel()>=NP && sums.numel()>=NP,"Metadata too short");
  auto dtype=fp32 ? at::kFloat : at::kHalf;
  TORCH_CHECK(bias.scalar_type()==dtype && (bias.numel()==0 || bias.numel()==N),"Bias shape/dtype mismatch");
  c10::cuda::CUDAGuard guard(a.device());
  const auto* properties=at::cuda::getCurrentDeviceProperties();
  int capability=properties->major*10+properties->minor;
  TORCH_CHECK(capability>=75 && capability<90,"This s4 WMMA backend targets SM75 through SM89 (including RTX 4090)");
  auto out=torch::empty({M,N},a.options().dtype(dtype));
  auto raw=fused ? torch::empty({0},a.options().dtype(at::kInt)) : torch::empty({M,N},a.options().dtype(at::kInt));
  bool decode=force_decode || M<8;
  dim3 grid(decode ? (N+3)/4 : (N+31)/32, decode ? M : (M+7)/8);
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.scalar_type(),"matmul_w4a4",[&] {
    auto bp=bias.numel() ? bias.data_ptr<scalar_t>() : nullptr;
    if (decode) {
      if (fused) decode_gemm<scalar_t,true><<<grid,128,0,stream>>>(a.data_ptr<unsigned char>(),b.data_ptr<unsigned char>(),sx.data_ptr<float>(),zx.data_ptr<int>(),sw.data_ptr<float>(),sums.data_ptr<int>(),bp,out.data_ptr<scalar_t>(),nullptr,M,N,KP);
      else decode_gemm<scalar_t,false><<<grid,128,0,stream>>>(a.data_ptr<unsigned char>(),b.data_ptr<unsigned char>(),sx.data_ptr<float>(),zx.data_ptr<int>(),sw.data_ptr<float>(),sums.data_ptr<int>(),bp,out.data_ptr<scalar_t>(),raw.data_ptr<int>(),M,N,KP);
    } else {
      if (fused) tensor_gemm<scalar_t,true><<<grid,128,0,stream>>>(a.data_ptr<unsigned char>(),b.data_ptr<unsigned char>(),sx.data_ptr<float>(),zx.data_ptr<int>(),sw.data_ptr<float>(),sums.data_ptr<int>(),bp,out.data_ptr<scalar_t>(),nullptr,M,N,KP);
      else tensor_gemm<scalar_t,false><<<grid,128,0,stream>>>(a.data_ptr<unsigned char>(),b.data_ptr<unsigned char>(),sx.data_ptr<float>(),zx.data_ptr<int>(),sw.data_ptr<float>(),sums.data_ptr<int>(),bp,out.data_ptr<scalar_t>(),raw.data_ptr<int>(),M,N,KP);
    }
    if (!fused) epilogue<scalar_t><<<(M*N+255)/256,256,0,stream>>>(raw.data_ptr<int>(),out.data_ptr<scalar_t>(),sx.data_ptr<float>(),zx.data_ptr<int>(),sw.data_ptr<float>(),sums.data_ptr<int>(),bp,M,N);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,raw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize_cuda);
  m.def("matmul", &matmul_cuda);
}
