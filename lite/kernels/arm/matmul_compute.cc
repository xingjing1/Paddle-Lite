// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/arm/matmul_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void MatMulCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

template <>
void MatMulCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

template <>
void MatMulCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.X->data<float>();
  const auto* y_data = param.Y->data<float>();
  auto* o_data = param.Out->mutable_data<float>();

  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  auto o_dims = param.Out->dims();
  bool x_transpose = param.transpose_X;
  bool y_transpose = param.transpose_Y;
  float alpha = param.alpha;
  auto& ctx = this->ctx_->template As<ARMContext>();

  operators::ActivationParam act_param;
  act_param.has_active = false;

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "x_dims size:" << x_dims.size() << std::endl;
  std::cout << "y_dims size:" << y_dims.size() << std::endl;
  for (int i = 0; i < x_dims[0] * x_dims[1] * x_dims[2]; i++) {
    std::cout << "matmul_x: " << x_data[i] << ",";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  for (int i = 0; i < y_dims[0] * y_dims[1]; i++) {
    std::cout << "matmul_y: " << float(((const int8_t*)y_data)[i]) << ",";
  }
  std::cout << std::endl;

  if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&
      (x_dims.size() != 2 || y_dims.size() != 2)) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    // or
    // x: [M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [M, K], y: [B, K, N], out: [B, M, N]
    int lda, ldb, ldc;
    if (!x_transpose) {
      m_ = x_dims[x_dims.size() - 2];
      k_ = x_dims[x_dims.size() - 1];
      lda = k_;
    } else {
      m_ = x_dims[x_dims.size() - 1];
      k_ = x_dims[x_dims.size() - 2];
      lda = m_;
    }

    if (!y_transpose) {
      n_ = y_dims[y_dims.size() - 1];
      ldb = n_;
    } else {
      n_ = y_dims[y_dims.size() - 2];
      ldb = k_;
    }

    ldc = n_;

    int x_inner = x_dims[x_dims.size() - 2] * x_dims[x_dims.size() - 1];
    int y_inner = y_dims[y_dims.size() - 2] * y_dims[y_dims.size() - 1];
    int out_inner = o_dims[o_dims.size() - 2] * o_dims[o_dims.size() - 1];

    if (x_dims.size() > 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
        lite::arm::math::sgemm(x_transpose,
                               y_transpose,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data + i * x_inner,
                               lda,
                               y_data + i * y_inner,
                               ldb,
                               0.f,
                               o_data + i * out_inner,
                               ldc,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    } else if (x_dims.size() > 2 && y_dims.size() == 2) {
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
        lite::arm::math::sgemm(x_transpose,
                               y_transpose,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data + i * x_inner,
                               lda,
                               y_data,
                               ldb,
                               0.f,
                               o_data + i * out_inner,
                               ldc,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    } else if (x_dims.size() == 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < y_dims.count(0, y_dims.size() - 2); ++i) {
        lite::arm::math::sgemm(x_transpose,
                               y_transpose,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data,
                               lda,
                               y_data + i * y_inner,
                               ldb,
                               0.f,
                               o_data + i * out_inner,
                               ldc,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    }
  } else if (x_dims.size() == 2 && y_dims.size() == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    int lda, ldb, ldc;
    if (!x_transpose) {
      m_ = x_dims[0];
      k_ = x_dims[1];
      lda = k_;
    } else {
      m_ = x_dims[1];
      k_ = x_dims[0];
      lda = m_;
    }
    if (!y_transpose) {
      n_ = y_dims[1];
      ldb = n_;
    } else {
      n_ = y_dims[0];
      ldb = k_;
    }
    ldc = n_;

    lite::arm::math::sgemm(x_transpose,
                           y_transpose,
                           m_,
                           n_,
                           k_,
                           alpha,
                           x_data,
                           lda,
                           y_data,
                           ldb,
                           0.f,
                           o_data,
                           ldc,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
  } else if (x_dims.size() > 2 && y_dims.size() == 1) {
    // x: [B, M, K], y: [K], out: [B, M]
    CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[0])
        << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
        << ")";
    for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 1); ++i) {
      o_data[i] = 0;
      for (size_t j = 0; j < y_dims[0]; ++j) {
        o_data[i] += x_data[i * y_dims[0] + j] * y_data[j] * alpha;
      }
    }
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    // x: [K], y: [K], out: [1]
    if (x_dims[0] == y_dims[0] && x_transpose == false &&
        y_transpose == false) {
      o_data[0] = 0.;
      for (size_t i = 0; i < x_dims[0]; ++i) {
        o_data[0] += x_data[i] * y_data[i] * alpha;
      }
    }
    // x: [M], y: [N], x_transpose: true, y_transpose: true, out: [M, N]
    if (x_transpose == true && y_transpose == true) {
      m_ = x_dims[0];
      k_ = 1;
      n_ = y_dims[0];
      int lda = k_;
      int ldb = n_;
      int ldc = n_;
      if (n_ == 1) {
        lite::arm::math::sgemv(x_data,
                               y_data,
                               o_data,
                               false,
                               m_,
                               k_,
                               0.f,
                               false,
                               nullptr,
                               false,
                               lite_api::ActivationType::kIndentity,
                               &ctx);
        if (fabsf(alpha - 1.f) > 1e-8f) {
          for (size_t i = 0; i < param.Out->dims().production(); ++i) {
            o_data[i] *= alpha;
          }
        }
      } else {
        lite::arm::math::sgemm(false,
                               false,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data,
                               lda,
                               y_data,
                               ldb,
                               0.f,
                               o_data,
                               ldc,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }

  for (int i = 0; i < 100; i++) {
    std::cout << "matmul_o:" << o_data[i] << ",";
  }
}

template <>
void MatMulCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.X->data<int8_t>();
  const auto* y_data = param.Y->data<int8_t>();
  auto* o_data = param.Out->mutable_data<float>();
  const float alpha = param.alpha;

  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  auto o_dims = param.Out->dims();
  bool x_transpose = param.transpose_X;
  bool y_transpose = param.transpose_Y;
  auto& ctx = this->ctx_->template As<ARMContext>();

  operators::ActivationParam act_param;
  act_param.has_active = false;

  std::cout << std::endl;
  std::cout << "x_dims size:" << x_dims.size() << std::endl;
  std::cout << "y_dims size:" << y_dims.size() << std::endl;
  std::vector<float> scale_;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&
      (x_dims.size() != 2 || y_dims.size() != 2)) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    // or
    // x: [M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [M, K], y: [B, K, N], out: [B, M, N]
    int lda, ldb, ldc;
    if (!x_transpose) {
      m_ = x_dims[x_dims.size() - 2];
      k_ = x_dims[x_dims.size() - 1];
      lda = k_;
    } else {
      m_ = x_dims[x_dims.size() - 1];
      k_ = x_dims[x_dims.size() - 2];
      lda = m_;
    }
    if (!y_transpose) {
      n_ = y_dims[y_dims.size() - 1];
      ldb = n_;
    } else {
      n_ = y_dims[y_dims.size() - 2];
      ldb = k_;
    }
    scale_.resize(n_);
    for (int i = 0; i < n_; i++) {
      param.output_scale = param.input_scale * param.weight_scale[i];
      scale_[i] = param.output_scale;
      std::cout << "scale_" << scale_[i] << std::endl;
    }

    ldc = n_;

    int x_inner = x_dims[x_dims.size() - 2] * x_dims[x_dims.size() - 1];
    int y_inner = y_dims[y_dims.size() - 2] * y_dims[y_dims.size() - 1];
    int out_inner = o_dims[o_dims.size() - 2] * o_dims[o_dims.size() - 1];

    if (x_dims.size() > 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
        lite::arm::math::gemm_s8(x_transpose,
                                 y_transpose,
                                 m_,
                                 n_,
                                 k_,
                                 x_data + i * x_inner,
                                 y_data + i * y_inner,
                                 o_data + i * out_inner,
                                 nullptr,
                                 false,
                                 scale_.data(),
                                 act_param,
                                 &ctx);
      }
    } else if (x_dims.size() > 2 && y_dims.size() == 2) {
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
        lite::arm::math::gemm_s8_n_scale_bias(x_transpose,
                                              y_transpose,
                                              m_,
                                              n_,
                                              k_,
                                              x_data + i * x_inner,
                                              y_data,
                                              o_data + i * out_inner,
                                              nullptr,
                                              false,
                                              scale_.data(),
                                              act_param,
                                              &ctx);
        // for (int i = 0; i < m_; i++) {
        //  for (int j = 0; j < n_; j++) {
        //    o_data[i * n_ + j] *= param.input_scale * param.weight_scale[j];
        //  }
        //}
      }
    } else if (x_dims.size() == 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < y_dims.count(0, y_dims.size() - 2); ++i) {
        lite::arm::math::gemm_s8(x_transpose,
                                 y_transpose,
                                 m_,
                                 n_,
                                 k_,
                                 x_data,
                                 y_data + i * y_inner,
                                 o_data + i * out_inner,
                                 nullptr,
                                 false,
                                 scale_.data(),
                                 act_param,
                                 &ctx);
      }
    }
  } else if (x_dims.size() == 2 && y_dims.size() == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    int lda, ldb, ldc;
    if (!x_transpose) {
      m_ = x_dims[0];
      k_ = x_dims[1];
      lda = k_;
    } else {
      m_ = x_dims[1];
      k_ = x_dims[0];
      lda = m_;
    }
    if (!y_transpose) {
      n_ = y_dims[1];
      ldb = n_;
    } else {
      n_ = y_dims[0];
      ldb = k_;
    }
    ldc = n_;
    scale_.resize(m_);
    for (int i = 0; i < m_; i++) {
      scale_[i] = param.output_scale;
    }

    lite::arm::math::gemm_s8(x_transpose,
                             y_transpose,
                             m_,
                             n_,
                             k_,
                             x_data,
                             y_data,
                             o_data,
                             nullptr,
                             false,
                             scale_.data(),
                             act_param,
                             &ctx);
  } else if (x_dims.size() > 2 && y_dims.size() == 1) {
    // x: [B, M, K], y: [K], out: [B, M]
    CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[0])
        << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
        << ")";
    for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 1); ++i) {
      o_data[i] = 0;
      for (size_t j = 0; j < y_dims[0]; ++j) {
        o_data[i] += x_data[i * y_dims[0] + j] * y_data[j] * alpha;
      }
    }
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    // x: [K], y: [K], out: [1]
    if (x_dims[0] == y_dims[0] && x_transpose == false &&
        y_transpose == false) {
      o_data[0] = 0.;
      for (size_t i = 0; i < x_dims[0]; ++i) {
        o_data[0] += x_data[i] * y_data[i] * alpha;
      }
    }
    // x: [M], y: [N], x_transpose: true, y_transpose: true, out: [M, N]
    if (x_transpose == true && y_transpose == true) {
      m_ = x_dims[0];
      k_ = 1;
      n_ = y_dims[0];
      int lda = k_;
      int ldb = n_;
      int ldc = n_;
      scale_.resize(m_);
      for (int i = 0; i < m_; i++) {
        scale_[i] = param.output_scale;
      }
      if (n_ == 1) {
        scale_.resize(m_);
        for (int i = 0; i < m_; i++) {
          scale_[i] = param.output_scale;
        }
        lite::arm::math::gemv_int8(x_data,
                                   y_data,
                                   o_data,
                                   false,
                                   m_,
                                   k_,
                                   scale_.data(),
                                   false,
                                   nullptr,
                                   act_param,
                                   &ctx);
        if (fabsf(alpha - 1.f) > 1e-8f) {
          for (size_t i = 0; i < param.Out->dims().production(); ++i) {
            o_data[i] *= alpha;
          }
        }
      } else {
        scale_.resize(m_);
        for (int i = 0; i < m_; i++) {
          scale_[i] = param.output_scale;
        }
        lite::arm::math::gemm_s8(false,
                                 false,
                                 m_,
                                 n_,
                                 k_,
                                 x_data,
                                 y_data,
                                 o_data,
                                 nullptr,
                                 false,
                                 scale_.data(),
                                 act_param,
                                 &ctx);
      }
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }

  for (int i = 0; i < 100; i++) {
    std::cout << "matmul_o:" << o_data[i] << std::endl;
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::MatMulCompute<PRECISION(kFloat),
                                                  PRECISION(kFloat)>
    Matmul_f32_f32;

typedef paddle::lite::kernels::arm::MatMulCompute<PRECISION(kInt8),
                                                  PRECISION(kFloat)>
    Matmul_int8_f32;

REGISTER_LITE_KERNEL(matmul, kARM, kFloat, kNCHW, Matmul_f32_f32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul, kARM, kInt8, kNCHW, Matmul_int8_f32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
