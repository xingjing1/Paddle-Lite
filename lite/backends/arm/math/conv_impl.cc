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

#include "lite/backends/arm/math/conv_impl.h"
#include <arm_neon.h>
#include <sys/time.h>
#include <algorithm>
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemv_arm_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/backends/arm/math/sgemv.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

/**
 * \brief neon implementation to add bias
 * @param tensor
 * @param bias
 * @param channel
 * @param channel_size
 */
void fill_bias(float* tensor,
               const float* bias,
               int channel,
               int channel_size) {
  if (tensor == nullptr) {
    return;
  }
  float* data = tensor;

  for (int j = 0; j < channel; ++j) {
    float32x4_t vdata = vdupq_n_f32(bias[j]);
    int i = 0;
    for (; i < channel_size - 3; i += 4) {
      vst1q_f32(data + i, vdata);
    }
    for (; i < channel_size; i++) {
      data[i] = bias[j];
    }
    data += channel_size;
  }
}

void fill_bias_int8(int* tensor,
                    const int* bias,
                    int channel,
                    int channel_size) {
  if (tensor == nullptr) {
    return;
  }
  int* data = tensor;
  for (int j = 0; j < channel; ++j) {
    int32x4_t vdata = vdupq_n_s32(bias[j]);
    int i = 0;
    for (; i < channel_size - 3; i += 4) {
      vst1q_s32(data + i, vdata);
    }
    for (; i < channel_size; i++) {
      data[i] = bias[j];
    }
    data += channel_size;
  }
}

/**
 * \brief inline funcs used in im2col
 * @param a
 * @param b
 * @return
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/**
 * \brief normal im2col function for gemm conv
 * @tparam dtype
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <typename Dtype>
void im2col_common(const Dtype* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   Dtype* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
          stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_top + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_left + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <>
void im2col_s1<float>(const float* data_im,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int pad_top,
                      int pad_bottom,
                      int pad_left,
                      int pad_right,
                      int dilation_h,
                      int dilation_w,
                      float* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(float));
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    int data_im_z = c * in_channel_size;
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * out_channel_size * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * out_channel_size;
        int data_col_z = data_col_z1 + data_col_z2 + data_col_z3;
        int oh_begin = std::max(((pad_top - h_offset)), 0);
        int oh_end = std::min(((height + pad_bottom - h_offset)), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset)), 0);
        int ow_end = std::min(((width + pad_right - w_offset)), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ++ih) {
          int iw = ow_begin - pad_left + w_offset;
          int ow = ow_begin;
          int data_im_offset = data_im_z + ih * width;
          int data_col_offset = data_col_z + oh * output_w;
          const float* data_im_ptr = data_im + data_im_offset;
          float* data_col_ptr = data_col + data_col_offset;
          for (; ow + 3 < ow_end; ow += 4, iw += 4) {
            float32x4_t tmp = vld1q_f32(data_im_ptr + iw);
            vst1q_f32(data_col_ptr + ow, tmp);
          }
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
}

template <>
void im2col_s1<int8_t>(const int8_t* data_im,
                       int channels,
                       int height,
                       int width,
                       int kernel_h,
                       int kernel_w,
                       int pad_top,
                       int pad_bottom,
                       int pad_left,
                       int pad_right,
                       int dilation_h,
                       int dilation_w,
                       int8_t* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(int8_t));
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    int data_im_z = c * in_channel_size;
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * out_channel_size * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * out_channel_size;
        int data_col_z = data_col_z1 + data_col_z2 + data_col_z3;
        int oh_begin = std::max(((pad_top - h_offset)), 0);
        int oh_end = std::min(((height + pad_bottom - h_offset)), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset)), 0);
        int ow_end = std::min(((width + pad_right - w_offset)), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ++ih) {
          int iw = ow_begin - pad_left + w_offset;
          int ow = ow_begin;
          int data_im_offset = data_im_z + ih * width;
          int data_col_offset = data_col_z + oh * output_w;
          const int8_t* data_im_ptr = data_im + data_im_offset;
          int8_t* data_col_ptr = data_col + data_col_offset;
          for (; ow + 15 < ow_end; ow += 16, iw += 16) {
            int8x16_t tmp = vld1q_s8(data_im_ptr + iw);
            vst1q_s8(data_col_ptr + ow, tmp);
          }
          for (; ow + 7 < ow_end; ow += 8, iw += 8) {
            int8x8_t tmp = vld1_s8(data_im_ptr + iw);
            vst1_s8(data_col_ptr + ow, tmp);
          }
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
}

template <>
void im2col_s2<float>(const float* data_im,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int pad_top,
                      int pad_bottom,
                      int pad_left,
                      int pad_right,
                      int dilation_h,
                      int dilation_w,
                      float* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
      1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(float));
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    int data_im_z = c * in_channel_size;
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * output_h * output_w * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * output_h * output_w;
        int data_col_z = data_col_z1 + data_col_z2 + data_col_z3;
        int oh_begin = std::max(((pad_top - h_offset + 1) / 2), 0);
        int oh_end =
            std::min(((height + pad_bottom - h_offset + 1) / 2), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset + 1) / 2), 0);
        int ow_end =
            std::min(((width + pad_right - w_offset + 1) / 2), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin * 2 - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ih += 2) {
          int iw = ow_begin * 2 - pad_left + w_offset;
          int ow = ow_begin;
          int data_im_offset = data_im_z + ih * width;
          int data_col_offset = data_col_z + oh * output_w;
          const float* data_im_ptr = data_im + data_im_offset;
          float* data_col_ptr = data_col + data_col_offset;
          for (; ow + 3 < ow_end; ow += 4, iw += 8) {
            float32x4x2_t tmp = vld2q_f32(data_im_ptr + iw);
            vst1q_f32(data_col_ptr + ow, tmp.val[0]);
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
}

template <>
void im2col_s2<int8_t>(const int8_t* data_im,
                       int channels,
                       int height,
                       int width,
                       int kernel_h,
                       int kernel_w,
                       int pad_top,
                       int pad_bottom,
                       int pad_left,
                       int pad_right,
                       int dilation_h,
                       int dilation_w,
                       int8_t* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
      1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(int8_t));
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    int data_im_z = c * in_channel_size;
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * output_h * output_w * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * output_h * output_w;
        int data_col_z = data_col_z1 + data_col_z2 + data_col_z3;
        int oh_begin = std::max(((pad_top - h_offset + 1) / 2), 0);
        int oh_end =
            std::min(((height + pad_bottom - h_offset + 1) / 2), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset + 1) / 2), 0);
        int ow_end =
            std::min(((width + pad_right - w_offset + 1) / 2), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin * 2 - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ih += 2) {
          int iw = ow_begin * 2 - pad_left + w_offset;
          int ow = ow_begin;
          int data_im_offset = data_im_z + ih * width;
          int data_col_offset = data_col_z + oh * output_w;
          const int8_t* data_im_ptr = data_im + data_im_offset;
          int8_t* data_col_ptr = data_col + data_col_offset;
          for (; ow + 15 < ow_end; ow += 16, iw += 32) {
            int8x16x2_t tmp = vld2q_s8(data_im_ptr + iw);
            vst1q_s8(data_col_ptr + ow, tmp.val[0]);
          }
          for (; ow + 7 < ow_end; ow += 8, iw += 16) {
            int8x8x2_t tmp = vld2_s8(data_im_ptr + iw);
            vst1_s8(data_col_ptr + ow, tmp.val[0]);
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
}

/**
 * \brief normal im2col function for gemm conv
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <>
void im2col<float>(const float* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   float* data_col) {
  bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
  bool pads_all_equal = (pads_equal && pad_top == pad_left);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
  bool kspd = pads_all_equal && ks_equal && no_dilation;
  if (kspd && stride_h == 1) {
    im2col_s1<float>(data_im,
                     channels,
                     height,
                     width,
                     kernel_h,
                     kernel_w,
                     pad_top,
                     pad_bottom,
                     pad_left,
                     pad_right,
                     dilation_h,
                     dilation_w,
                     data_col);
  } else if (kspd && stride_h == 2) {
    im2col_s2<float>(data_im,
                     channels,
                     height,
                     width,
                     kernel_h,
                     kernel_w,
                     pad_top,
                     pad_bottom,
                     pad_left,
                     pad_right,
                     dilation_h,
                     dilation_w,
                     data_col);
  } else {
    im2col_common<float>(data_im,
                         channels,
                         height,
                         width,
                         kernel_h,
                         kernel_w,
                         pad_top,
                         pad_bottom,
                         pad_left,
                         pad_right,
                         stride_h,
                         stride_w,
                         dilation_h,
                         dilation_w,
                         data_col);
  }
}

/**
 * \brief normal im2col function for gemm conv
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <>
void im2col<int8_t>(const int8_t* data_im,
                    int channels,
                    int height,
                    int width,
                    int kernel_h,
                    int kernel_w,
                    int pad_top,
                    int pad_bottom,
                    int pad_left,
                    int pad_right,
                    int stride_h,
                    int stride_w,
                    int dilation_h,
                    int dilation_w,
                    int8_t* data_col) {
  bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
  bool pads_all_equal = (pads_equal && pad_top == pad_left);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
  bool kspd = pads_all_equal && ks_equal && no_dilation;
  if (kspd && stride_h == 1) {
    std::cout << "int im2col_s1" << std::endl;
    im2col_s1<int8_t>(data_im,
                      channels,
                      height,
                      width,
                      kernel_h,
                      kernel_w,
                      pad_top,
                      pad_bottom,
                      pad_left,
                      pad_right,
                      dilation_h,
                      dilation_w,
                      data_col);
  } else if (kspd && stride_h == 2) {
    im2col_s2<int8_t>(data_im,
                      channels,
                      height,
                      width,
                      kernel_h,
                      kernel_w,
                      pad_top,
                      pad_bottom,
                      pad_left,
                      pad_right,
                      dilation_h,
                      dilation_w,
                      data_col);
  } else {
    im2col_common<int8_t>(data_im,
                          channels,
                          height,
                          width,
                          kernel_h,
                          kernel_w,
                          pad_top,
                          pad_bottom,
                          pad_left,
                          pad_right,
                          stride_h,
                          stride_w,
                          dilation_h,
                          dilation_w,
                          data_col);
  }
}

/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm
 * implementation
 */
void conv1x1s1_gemm(const float* i_data,
                    float* o_data,
                    int num,
                    int oc,
                    int oh,
                    int ow,
                    int ic,
                    int ih,
                    int win,
                    const float* weights,
                    const float* bias,
                    const operators::ConvParam& param,
                    ARMContext* ctx) {
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;

  const int group = param.groups;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float* dout_group =
          static_cast<float*>(o_data) + (b * oc + g * m) * channel_size_out;
      const float* din_group = static_cast<const float*>(i_data) +
                               (b * ic + g * k) * channel_size_in;
      const float* weights_group =
          static_cast<const float*>(weights) + g * weights_size_per_group;
      const float* bias_group = static_cast<const float*>(bias) + g * m;

      if (n == 1) {
        sgemv(weights_group,
              din_group,
              dout_group,
              false,
              m,
              k,
              0.f,
              flag_bias,
              bias_group,
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
      } else if (m == 1) {
        float bias_ptr[n];  // NOLINT
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }

        sgemv(din_group,
              weights_group,
              dout_group,
              true,
              n,
              k,
              0.f,
              flag_bias,
              bias_ptr,
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
      } else {
        sgemm_prepack(false,
                      m,
                      n,
                      k,
                      weights_group,
                      din_group,
                      n,
                      0.f,
                      dout_group,
                      n,
                      bias_group,
                      flag_bias,
                      act_param,
                      ctx);
      }
    }
  }
}

template <typename Dtype>
void conv1x1s1_gemm_int8(const int8_t* i_data,
                         Dtype* o_data,
                         int num,
                         int oc,
                         int oh,
                         int ow,
                         int ic,
                         int ih,
                         int win,
                         const int8_t* weights,
                         const float* bias,
                         const operators::ConvParam& param,
                         ARMContext* ctx,
                         const float* scale) {
  int group = param.groups;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;
  int hblock = get_hblock_int8(ctx);
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      Dtype* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const int8_t* din_group = i_data + (b * ic + g * k) * channel_size_in;
      const int8_t* weights_group = weights + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      const float* scale_group = scale + g * m;
      if (n == 1) {
        gemv_int8(weights_group,
                  din_group,
                  dout_group,
                  false,
                  m,
                  k,
                  scale_group,
                  flag_bias,
                  bias_group,
                  act_param,
                  ctx);
      } else if (m == 1) {
        float bias_ptr[n];   // NOLINT
        float scale_ptr[n];  // NOLINT
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }
        for (int i = 0; i < n; i++) {
          scale_ptr[i] = scale_group[0];
        }
        gemv_int8(din_group,
                  weights_group,
                  dout_group,
                  true,
                  n,
                  k,
                  scale_ptr,
                  flag_bias,
                  bias_ptr,
                  act_param,
                  ctx);
      } else {
        gemm_prepack_int8(weights_group,
                          din_group,
                          bias_group,
                          dout_group,
                          m,
                          n,
                          k,
                          flag_bias,
                          false,
                          scale_group,
                          act_param,
                          ctx);
      }
    }
  }
}

template void conv1x1s1_gemm_int8<int8_t>(const int8_t* i_data,
                                          int8_t* o_data,
                                          int num,
                                          int oc,
                                          int oh,
                                          int ow,
                                          int ic,
                                          int ih,
                                          int win,
                                          const int8_t* weights,
                                          const float* bias,
                                          const operators::ConvParam& param,
                                          ARMContext* ctx,
                                          const float* scale);

template void conv1x1s1_gemm_int8<float>(const int8_t* i_data,
                                         float* o_data,
                                         int num,
                                         int oc,
                                         int oh,
                                         int ow,
                                         int ic,
                                         int ih,
                                         int win,
                                         const int8_t* weights,
                                         const float* bias,
                                         const operators::ConvParam& param,
                                         ARMContext* ctx,
                                         const float* scale);

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm
 * implementation
 */
void conv_im2col_gemm(const float* i_data,
                      float* o_data,
                      int num,
                      int oc,
                      int oh,
                      int ow,
                      int ic,
                      int ih,
                      int win,
                      const float* weights,
                      const float* bias,
                      const operators::ConvParam& param,
                      ARMContext* ctx) {
  const int group = param.groups;
  auto filter_dims = param.filter->dims();
  const int kernel_h = filter_dims[2];
  const int kernel_w = filter_dims[3];  // nchw
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  int hblock = get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;

  auto act_param = param.activation_param;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const float* din_group =
          i_data + (b * ic + g * chin_per_group) * channel_size_in;
      const float* weights_group = weights + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      float* dB = tmp_work_space;
      im2col<float>(din_group,
                    chin_per_group,
                    ih,
                    win,
                    kernel_h,
                    kernel_w,
                    paddings[0],
                    paddings[1],
                    paddings[2],
                    paddings[3],
                    param.strides[0],
                    param.strides[1],
                    dilations[0],
                    dilations[1],
                    dB);
      if (n == 1) {
        sgemv(weights_group,
              dB,
              dout_group,
              false,
              m,
              k,
              0.f,
              flag_bias,
              bias_group,
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
      } else if (m == 1) {
        float bias_ptr[n];  // NOLINT
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }
        sgemv(dB,
              weights_group,
              dout_group,
              true,
              n,
              k,
              0.f,
              flag_bias,
              bias_ptr,
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
      } else {
        int ldb = n;
        sgemm_prepack(false,
                      m,
                      n,
                      k,
                      weights_group,
                      dB,
                      ldb,
                      0.f,
                      dout_group,
                      n,
                      bias_group,
                      flag_bias,
                      act_param,
                      ctx);
      }
    }
  }
}
double get_current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);

  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void im2col_tile(const int8_t* data_im,
                 int channels,
                 int height,
                 int width,
                 int kernel_h,
                 int kernel_w,
                 int pad_top,
                 int pad_bottom,
                 int pad_left,
                 int pad_right,
                 int stride_h,
                 int stride_w,
                 int dilation_h,
                 int dilation_w,
                 int8_t* data_col) {
  // int in_channel_size = (height + kernel_h - 1) * (width + kernel_w - 1);
  int in_channel_size = (height) * (width);
  int8_t* inptr[8];
  int8_t* outptr = data_col;
  //
  int tile_len = 8;
  for (int i = 0; i < tile_len; i++) {
    inptr[i] = data_col + i * channels * kernel_h * kernel_w;
  }

  int8x16_t tmp[3][8];
  for (int c = 0; c < channels; c += 8) {
    for (int i = 0; i < kernel_h; i++) {
      tmp[i][0] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 0);
      tmp[i][1] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 1);
      tmp[i][2] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 2);
      tmp[i][3] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 3);
      tmp[i][4] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 4);
      tmp[i][5] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 5);
      tmp[i][6] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 6);
      tmp[i][7] = vld1q_s8(data_im + i * width + c * in_channel_size +
                           in_channel_size * 7);
    }

    for (int i = 0; i < tile_len; i++) {
      int index0 = 0 + i;
      int index1 = 1 + i;
      int index2 = 2 + i;

      data_col[i * channels * kernel_h * kernel_w + 0] = tmp[0][0 + i];
      data_col[i * channels * kernel_h * kernel_w + 1] = tmp[0][1 + i];
      data_col[i * channels * kernel_h * kernel_w + 2] = tmp[0][2 + i];
      data_col[i * channels * kernel_h * kernel_w + 3] = tmp[1][0 + i];
      data_col[i * channels * kernel_h * kernel_w + 4] = tmp[1][1 + i];
      data_col[i * channels * kernel_h * kernel_w + 5] = tmp[1][2 + i];
      data_col[i * channels * kernel_h * kernel_w + 6] = tmp[2][0 + i];
      data_col[i * channels * kernel_h * kernel_w + 7] = tmp[2][1 + i];
      data_col[i * channels * kernel_h * kernel_w + 8] = tmp[2][2 + i];

      // int8x16_t data_col_r;
      // data_col_r[0] = tmp[0][0 + i];
      // data_col_r[1] = tmp[0][1 + i];
      // data_col_r[2] = tmp[0][2 + i];
      // data_col_r[3] = tmp[1][0 + i];
      // data_col_r[4] = tmp[1][1 + i];
      // data_col_r[5] = tmp[1][2 + i];
      // data_col_r[6] = tmp[2][0 + i];
      // data_col_r[7] = tmp[2][1 + i];
      // data_col_r[8] = tmp[2][2 + i];
      // vst1q_s8(data_col + i * channels * kernel_h * kernel_w, data_col_r);
    }
    data_col += kernel_h * kernel_w;
  }

  int x = channels * kernel_h * kernel_w;

  for (; x >= 8; x -= 8) {
    asm volatile(
        "ld1 {v0.8b}, [%[inptr0]], #8 \n"  // v0=a0a1a2a3a4a5a6a7
        "ld1 {v1.8b}, [%[inptr1]], #8 \n"  // v1=b0b1b2b3b4b5b6b7
        "ld1 {v2.8b}, [%[inptr2]], #8 \n"  // v2=c0c1c2c3c4c5c6c7
        "ld1 {v3.8b}, [%[inptr3]], #8 \n"  // v3=d0d1d2d3d4d5d6d7

        "ld1 {v4.8b}, [%[inptr4]], #8 \n"  // v0=e0e1a2a3a4a5a6a7
        "ld1 {v5.8b}, [%[inptr5]], #8 \n"  // v1=f0f1b2b3b4b5b6b7
        "ld1 {v6.8b}, [%[inptr6]], #8 \n"  // v2=g0g1c2c3c4c5c6c7
        "ld1 {v7.8b}, [%[inptr7]], #8 \n"  // v3=h0h1d2d3d4d5d6d7

        "trn1 v12.2s, v0.2s, v1.2s \n"  // v0=a0a1a2a3b0b1b2b3
        "trn2 v13.2s, v0.2s, v1.2s \n"  // v0=a4a5a6a7b4b5b6b7
        "trn1 v14.2s, v2.2s, v3.2s \n"  // v0=c0c1c2c3d0d1d2d3
        "trn2 v15.2s, v2.2s, v3.2s \n"  // v0=c4c5c6c7d4d5d6d7

        "trn1 v16.2s, v4.2s, v5.2s \n"  // v0=e0e1e2e3f0f1f2f3
        "trn2 v17.2s, v4.2s, v5.2s \n"  // v0=e4e5e6e7f4f5f6f7
        "trn1 v18.2s, v6.2s, v7.2s \n"  // v0=g0g1g2g3h0h1h2h3
        "trn2 v19.2s, v6.2s, v7.2s \n"  // v0=g4g5g6g7h4h5h6h7

        "st1 {v12.2s}, [%[outptr]], #8\n"
        "st1 {v14.2s}, [%[outptr]], #8\n"
        "st1 {v16.2s}, [%[outptr]], #8\n"
        "st1 {v18.2s}, [%[outptr]], #8\n"

        "st1 {v13.2s}, [%[outptr]], #8\n"
        "st1 {v15.2s}, [%[outptr]], #8\n"
        "st1 {v17.2s}, [%[outptr]], #8\n"
        "st1 {v19.2s}, [%[outptr]], #8\n"
        : [inptr0] "+r"(inptr[0]),
          [inptr1] "+r"(inptr[1]),
          [inptr2] "+r"(inptr[2]),
          [inptr3] "+r"(inptr[3]),
          [inptr4] "+r"(inptr[4]),
          [inptr5] "+r"(inptr[5]),
          [inptr6] "+r"(inptr[6]),
          [inptr7] "+r"(inptr[7]),
          [outptr] "+r"(outptr)
        :
        : "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "v16",
          "v17",
          "v18",
          "v19",
          "v20",
          "v21",
          "v22",
          "v23",
          "cc",
          "memory");
  }
}

template <typename Dtype>
void conv_3x3s1_im2col_gemm_int8(const int8_t* i_data,
                                 Dtype* o_data,
                                 int num,
                                 int oc,
                                 int oh,
                                 int ow,
                                 int ic,
                                 int ih,
                                 int win,
                                 const int8_t* weights,
                                 const float* bias,
                                 const operators::ConvParam& param,
                                 ARMContext* ctx,
                                 const float* scale) {
  int group = param.groups;
  auto filter_dims = param.filter->dims();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock_int8(ctx);
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }

  int tile_size = 8;
  int num_tile = ROUNDUP(n, tile_size);
  int8_t* tmp_work_space =
      ctx->workspace_data<int8_t>() + ctx->llc_size() / sizeof(int8_t);
  int index = 0;
  for (int b = 0; b < num; ++b) {
    for (int g = 0; g < group; ++g) {
      Dtype* dout_group = o_data + (b * oc + g * m) * channel_size_out;

      // int8_t* data_tmp =  (int8_t*)malloc(56 * 56 * 64 * sizeof(int8_t));
      // for (int i = 0; i < 56 * 56 * 64; i++) {
      //  data_tmp[i] = i % 56;
      //}
      // const int8_t* din_group = data_tmp;
      const int8_t* din_group = static_cast<const int8_t*>(i_data) +
                                (b * ic + g * chin_per_group) * channel_size_in;
      // for (int ic = 0; ic < 1; ic++) {
      //  for (int ih = 0; ih < 56; ih++) {
      //    for (int iw = 0; iw < 56; iw++) {
      //      std::cout<<(int)din_group[iw + ih * 56 + ic * 56*56]<<" ";
      //    }
      //    std::cout<<std::endl;
      //  }
      //    std::cout<<std::endl;
      //    std::cout<<std::endl;
      //}
      const int8_t* weights_group =
          static_cast<const int8_t*>(weights) + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      int8_t* dB = tmp_work_space;
      const float* scale_group = scale + g * m;

      const int output_h =
          (ih + paddings[0] + paddings[1] - (dila_h * (kernel_h - 1) + 1)) + 1;
      const int output_w =
          (win + paddings[2] + paddings[3] - (dila_w * (kernel_w - 1) + 1)) + 1;

      for (int i = 0; i < output_h; i += 1) {
        const int8_t* data_in_tile = din_group;
        for (int j = 0; j < output_w; j += tile_size) {
          // std::cout<<"output_h:" << output_h<<std::endl;
          // std::cout<<"output_w:" << output_w<<std::endl;
          int tile_len = tile_size;
          im2col_tile(data_in_tile,
                      chin_per_group,
                      ih,
                      win,
                      kernel_h,
                      kernel_w,
                      pad_h,
                      paddings[1],
                      pad_w,
                      paddings[3],
                      stride_h,
                      stride_w,
                      dila_h,
                      dila_w,
                      dB);
          // for (int it = 0; it < tile_size; it++) {
          //  for (int iw = 0; iw < chin_per_group * kernel_h * kernel_w; iw++)
          //  {
          //    std::cout<<(int)dB[iw + it * chin_per_group * kernel_h *
          //    kernel_w]<<" ";
          //  }
          //  std::cout<<std::endl;
          //  std::cout<<std::endl;
          //}
          //  std::cout<<std::endl;
          //  std::cout<<std::endl;
          //  std::cout<<std::endl;
          //  std::cout<<std::endl;
          //  std::cout<<std::endl;
          //  std::cout<<std::endl;

          // for (int m_idx = 0; m_idx < m; m_idx += MBLOCK_INT8_DOT) {
          //  int k = chin_per_group * kernel_h * kernel_w;
          //  gemm_int8_im2col(weights_group,
          //                    dB,
          //                    bias_group,
          //                    dout_group,
          //                    output_h,
          //                    output_w,
          //                    k,
          //                    flag_bias,
          //                    false,
          //                    scale_group,
          //                    act_param,
          //                    ctx,
          //                    i,
          //                    j,
          //                    m_idx);
          //}
          data_in_tile += tile_size;

          // for (int i = 0; i < tile_size * kernel_h * kernel_w *
          // chin_per_group; i++) {
          //  //std::cout<<(int)dout_group[i]<<" "<<std::endl;
          //}
        }
        din_group += win;
      }
    }
  }
}

template void conv_3x3s1_im2col_gemm_int8<int8_t>(
    const int8_t* i_data,
    int8_t* o_data,
    int num,
    int oc,
    int oh,
    int ow,
    int ic,
    int ih,
    int win,
    const int8_t* weights,
    const float* bias,
    const operators::ConvParam& param,
    ARMContext* ctx,
    const float* scale);

template void conv_3x3s1_im2col_gemm_int8<float>(
    const int8_t* i_data,
    float* o_data,
    int num,
    int oc,
    int oh,
    int ow,
    int ic,
    int ih,
    int win,
    const int8_t* weights,
    const float* bias,
    const operators::ConvParam& param,
    ARMContext* ctx,
    const float* scale);

template <typename Dtype>
void conv_im2col_gemm_int8(const int8_t* i_data,
                           Dtype* o_data,
                           int num,
                           int oc,
                           int oh,
                           int ow,
                           int ic,
                           int ih,
                           int win,
                           const int8_t* weights,
                           const float* bias,
                           const operators::ConvParam& param,
                           ARMContext* ctx,
                           const float* scale) {
  int group = param.groups;
  auto filter_dims = param.filter->dims();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock_int8(ctx);
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }

  int8_t* tmp_work_space =
      ctx->workspace_data<int8_t>() + ctx->llc_size() / sizeof(int8_t);

  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      std::cout << "group:" << group << std::endl;
      Dtype* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const int8_t* din_group = static_cast<const int8_t*>(i_data) +
                                (b * ic + g * chin_per_group) * channel_size_in;
      const int8_t* weights_group =
          static_cast<const int8_t*>(weights) + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      int8_t* dB = tmp_work_space;
      const float* scale_group = scale + g * m;
      double time0 = get_current_time();
      im2col<int8_t>(din_group,
                     chin_per_group,
                     ih,
                     win,
                     kernel_h,
                     kernel_w,
                     pad_h,
                     paddings[1],
                     pad_w,
                     paddings[3],
                     stride_h,
                     stride_w,
                     dila_h,
                     dila_w,
                     dB);

      double time3 = get_current_time();
      std::cout << "img2col time:" << time3 - time0 << std::endl;
      if (n == 1) {
        gemv_int8(weights_group,
                  dB,
                  dout_group,
                  false,
                  m,
                  k,
                  scale_group,
                  flag_bias,
                  bias_group,
                  act_param,
                  ctx);
      } else if (m == 1) {
        float bias_ptr[n];   // NOLINT
        float scale_ptr[n];  // NOLINT
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }
        memset(scale_ptr, scale_group[0], sizeof(float) * n);
        gemv_int8(din_group,
                  weights_group,
                  dout_group,
                  true,
                  n,
                  k,
                  scale_ptr,
                  flag_bias,
                  bias_ptr,
                  act_param,
                  ctx);
      } else {
        double time1 = get_current_time();
        gemm_prepack_int8(weights_group,
                          dB,
                          bias_group,
                          dout_group,
                          m,
                          n,
                          k,
                          flag_bias,
                          false,
                          scale_group,
                          act_param,
                          ctx);
        double time2 = get_current_time();
        std::cout << "gemm time:" << time2 - time1 << std::endl;
      }
    }
  }
}

template void conv_im2col_gemm_int8<int8_t>(const int8_t* i_data,
                                            int8_t* o_data,
                                            int num,
                                            int oc,
                                            int oh,
                                            int ow,
                                            int ic,
                                            int ih,
                                            int win,
                                            const int8_t* weights,
                                            const float* bias,
                                            const operators::ConvParam& param,
                                            ARMContext* ctx,
                                            const float* scale);

template void conv_im2col_gemm_int8<float>(const int8_t* i_data,
                                           float* o_data,
                                           int num,
                                           int oc,
                                           int oh,
                                           int ow,
                                           int ic,
                                           int ih,
                                           int win,
                                           const int8_t* weights,
                                           const float* bias,
                                           const operators::ConvParam& param,
                                           ARMContext* ctx,
                                           const float* scale);

void conv_depthwise_3x3_fp32(const void* din,
                             void* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const void* weights,
                             const float* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx,
                             const float* scale) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];
  int stride = param.strides[1];
  int pad = pad_w;
  bool flag_bias = param.bias != nullptr;
  bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
  if (stride == 1) {
    if (pads_less && (pad_h == pad_w) && (pad < 2)) {  // support pad = [0, 1]
      conv_depthwise_3x3s1_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                pad,
                                flag_bias,
                                act_param,
                                ctx);
    } else {
      conv_3x3s1_depthwise_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else if (stride == 2) {
    if (pads_less && pad_h == pad_w && (pad < 2)) {  // support pad = [0, 1]
      conv_depthwise_3x3s2_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                pad,
                                flag_bias,
                                act_param,
                                ctx);
    } else {
      conv_3x3s2_depthwise_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else {
    LOG(FATAL) << "fp32 depthwise conv3x3 stride: " << stride << " unsupported";
  }
}

void conv_depthwise_5x5_fp32(const void* din,
                             void* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const void* weights,
                             const float* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx,
                             const float* scale) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  ctx->ExtendWorkspace((w_in + w_out) * sizeof(float));
  if (stride == 2) {
    conv_depthwise_5x5s2_fp32(reinterpret_cast<const float*>(din),
                              reinterpret_cast<float*>(dout),
                              num,
                              ch_out,
                              h_out,
                              w_out,
                              ch_in,
                              h_in,
                              w_in,
                              reinterpret_cast<const float*>(weights),
                              bias,
                              param,
                              act_param,
                              ctx);
  } else if (stride == 1) {
    conv_depthwise_5x5s1_fp32(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const float*>(din),
                              reinterpret_cast<const float*>(weights),
                              bias,
                              flag_bias,
                              flag_relu,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              param,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv";
  }
}

void conv_depthwise_3x3_int8_fp32(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  bool support_act_type = flag_act <= 1;
  bool support_pad_type =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
      (paddings[0] == paddings[2]) && (paddings[0] == 0 || paddings[0] == 1);
  bool support_stride_type = (param.strides[0] == 1 && param.strides[1] == 1);
  bool support_width_type = w_in > 9 ? true : false;
  if (stride == 1) {
    if (!support_act_type || !support_pad_type || !support_stride_type ||
        !support_width_type) {
      conv_depthwise_3x3s1_int8(reinterpret_cast<float*>(dout),
                                reinterpret_cast<const int8_t*>(din),
                                reinterpret_cast<const int8_t*>(weights),
                                scale,
                                bias,
                                flag_bias,
                                flag_act,
                                alpha,
                                num,
                                ch_in,
                                h_in,
                                w_in,
                                h_out,
                                w_out,
                                pad_w,
                                pad_h,
                                ctx);
    } else {
      conv_depthwise_3x3s1_int8_float_impl(
          reinterpret_cast<float*>(dout),
          reinterpret_cast<const int8_t*>(din),
          reinterpret_cast<const int8_t*>(weights),
          scale,
          bias,
          flag_bias,
          flag_act,
          alpha,
          num,
          ch_in,
          h_in,
          w_in,
          h_out,
          w_out,
          pad_w,
          pad_h,
          ctx);
    }
  } else if (stride == 2) {
    conv_depthwise_3x3s2_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 3x3 dw conv int8";
  }
}

void conv_depthwise_3x3_int8_int8(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  bool support_act_type = flag_act <= 1;
  bool support_pad_type =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
      (paddings[0] == paddings[2]) && (paddings[0] == 0 || paddings[0] == 1);
  bool support_stride_type = (param.strides[0] == 1 && param.strides[1] == 1);
  bool support_width_type = w_in > 9 ? true : false;
  if (stride == 1) {
    if (!support_act_type || !support_pad_type || !support_stride_type ||
        !support_width_type) {
      conv_depthwise_3x3s1_int8(reinterpret_cast<int8_t*>(dout),
                                reinterpret_cast<const int8_t*>(din),
                                reinterpret_cast<const int8_t*>(weights),
                                scale,
                                bias,
                                flag_bias,
                                flag_act,
                                alpha,
                                num,
                                ch_in,
                                h_in,
                                w_in,
                                h_out,
                                w_out,
                                pad_w,
                                pad_h,
                                ctx);
    } else {
      conv_depthwise_3x3s1_int8_int8_impl(
          reinterpret_cast<int8_t*>(dout),
          reinterpret_cast<const int8_t*>(din),
          reinterpret_cast<const int8_t*>(weights),
          scale,
          bias,
          flag_bias,
          flag_act,
          alpha,
          num,
          ch_in,
          h_in,
          w_in,
          h_out,
          w_out,
          pad_w,
          pad_h,
          ctx);
    }
  } else if (stride == 2) {
    conv_depthwise_3x3s2_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 3x3 dw conv int8";
  }
}

void conv_depthwise_5x5_int8_fp32(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  if (stride == 1) {
    conv_depthwise_5x5s1_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else if (stride == 2) {
    conv_depthwise_5x5s2_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv int8";
  }
}

void conv_depthwise_5x5_int8_int8(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  if (stride == 1) {
    conv_depthwise_5x5s1_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else if (stride == 2) {
    conv_depthwise_5x5s2_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv int8";
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
