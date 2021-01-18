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
#include "lite/kernels/host/tile_compute.h"
#include <algorithm>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void TileCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::TileParam>();
  auto rank = std::max(param.X->dims().size(), param.repeat_times.size());
  auto repeat_times = param.repeat_times;
  auto in_dims = param.X->dims();
  auto vec_in_dims = in_dims.Vectorize();
  // broadcast for vec_in_dims.size() equal to repeat_times.size()
  if (repeat_times.size() < vec_in_dims.size()) {
    int diff = vec_in_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  }
  DDim new_in_dims{vec_in_dims};
  DDim out_dims(new_in_dims);
  std::vector<int> bcast_dims(vec_in_dims.size());
  std::vector<int> repeat_stride(vec_in_dims.size());
  std::vector<int> in_stride(vec_in_dims.size());

  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
    out_dims[i] *= repeat_times[i];
    if (i > 0) {
      repeat_stride[i] = out_dims.production() / out_dims[i - 1];
      in_stride[i] = new_in_dims.production() / new_in_dims[i - 1];
    }
  }

  auto& in = param.X;
  auto& out = param.Out;
  out->Resize(out_dims);
  Tensor* tmp_src_tensor;
  Tensor* tmp_dst_tensor;
  auto out_data = out->template mutable_data<T>();
  auto in_data = in->template data<T>();

  tmp_src_tensor->Resize(out_dims);
  tmp_dst_tensor->Resize(out_dims);
  auto tmp_src = tmp_src_tensor->mutable_data<float>();
  auto tmp_dst = tmp_dst_tensor->mutable_data<float>();
  for (int i = 0; i < in->dims().production(); i++) {
    tmp_src[i] = in_data[i];
  }

  for (int i = bcast_dims.size() - 1; i >= 0; i--) {
    for (int m = 0; m < in_dims.production() / in_stride[i]; m++) {
      if (bcast_dims[i] > 1) {
        for (int j = 0; j < bcast_dims[i]; j++) {
          int dst_stride = repeat_stride[i] * sizeof(float);
          std::memcpy(tmp_dst + j * dst_stride + m * bcast_dims[i] * dst_stride,
                      tmp_src + m * in_stride[i],
                      dst_stride);
        }
        tmp_src_tensor->CopyDataFrom(*tmp_dst_tensor);
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using tile_float =
    paddle::lite::kernels::host::TileCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kAny, tile_float, def_float)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .Finalize();
using tile_int32 =
    paddle::lite::kernels::host::TileCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(tile, kHost, kInt32, kAny, tile_int32, def_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .Finalize();

using tile_int64 =
    paddle::lite::kernels::host::TileCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kAny, tile_int64, def_int64)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .Finalize();

using tile_int8 =
    paddle::lite::kernels::host::TileCompute<int8_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kAny, tile_int8, def_int8)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt8), DATALAYOUT(kAny), -1)})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt8), DATALAYOUT(kAny), -1)})
    .Finalize();

using tile_bool =
    paddle::lite::kernels::host::TileCompute<bool, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kAny, tile_bool, def_bool)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();
