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
/*
template <int Rank>
void Tile(operators::TileParam param) {
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
  //fliud::Eigen::DSizes<int, Rank> bcast_dims;

  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
  }
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_dims[i] *= repeat_times[i];
  }

  auto& in = param.X;
  auto& out = param.Out;
  out->Resize(out_dims);
  out->mutable_data<T>();
 // auto x = fluid::EigenTensor<T, Rank>::From(*in, new_in_dims);
 // auto y = fluid::EigenTensor<T, Rank>::From(*out, out_dims);
 // y = x.broadcast(bcast_dims);
}
*/
template <typename T, PrecisionType PType>
void TileCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::TileParam>();
  auto rank = std::max(param.X->dims().size(), param.repeat_times.size());
  // Tile<rank>(param);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using tile_float =
    paddle::lite::kernels::host::TileCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(stack, kHost, kFloat, kAny, tile_float, def)
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
REGISTER_LITE_KERNEL(stack, kHost, kInt32, kAny, tile_int32, def)
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

/*
REGISTER_LITE_KERNEL(tile,
                     kHost,
                     kFloat,
                     kAny,
                     paddle::lite::kernels::host::TileCompute<float,
PRECISION(kFloat)>,
                     def)
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

REGISTER_LITE_KERNEL(tile,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::TileCompute<int32_t,
PRECISION(kInt)>,
                     def)
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

REGISTER_LITE_KERNEL(tile,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::TileCompute<int64_t>,
                     def_int64)
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

REGISTER_LITE_KERNEL(tile,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::TileCompute<int8_t>,
                     def_int8)
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
REGISTER_LITE_KERNEL(tile,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::TileCompute<bool>,
                     def_bool)
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
    */
