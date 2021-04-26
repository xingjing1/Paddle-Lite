// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/select_input_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void SelectInputCompute::Run() {
  auto& param = this->Param<param_t>();

  // LOG(INFO) << "select input mask:" << *param.Mask->data<int>();
  //    for(int j  = 0; j < param.X[0]->numel(); j++) {
  // LOG(INFO) <<"select input input0:"<<param.X[0]->data<int>()[j];
  //  }
  // for(int j  = 0; j < param.X[1]->numel(); j++) {
  //        LOG(INFO) <<"select input input1:"<<param.X[1]->data<int64_t>()[j];
  // }

  param.Out->CopyDataFrom(*param.X[*param.Mask->data<int>()]);
  for (int i = 0; i < param.Out->numel(); i++) {
    // LOG(INFO) <<"select input out:" <<param.Out->data<int64_t>()[i];
    // LOG(INFO) <<"select input out precision: "
    // <<static_cast<int>(param.Out->precision());
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(select_input,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::SelectInputCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("Mask",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
