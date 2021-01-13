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

REGISTER_LITE_KERNEL(tile,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::TileCompute<float>,
                     def_float)
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
                     paddle::lite::kernels::host::TileCompute<int32_t>,
                     def_int32)
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
