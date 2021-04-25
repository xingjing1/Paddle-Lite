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

#include "lite/kernels/host/write_to_array_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void WriteToArrayCompute::Run() {
  auto& param = this->template Param<operators::WriteToArrayParam>();
  CHECK_EQ(param.I->numel(), 1) << "input2 should have only one element";

  int id = param.I->data<int64_t>()[0];
  // LOG(INFO)<<"param.Out size:"<<param.Out->size() ;
  // int id = param.Out->size();
  if (param.Out->size() < id + 1) {
    param.Out->resize(id + 1);
    // LOG(INFO)<<"id+ 1:"<<id + 1 ;
  }
  param.Out->at(id).CopyDataFrom(*param.X);

  //    LOG(INFO)<<"param.X.numel:"<<param.X->numel();
  for (int i = 0; i < param.X->numel(); i++) {
    //  LOG(INFO)<<"param.X:"<<param.X->data<float>()[i];
  }

  // my_print(std::string name, T* ptr, const paddle::lite::DDim& dims,
  // paddle::lite::LoD lod = {{}}) {
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(write_to_array,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::WriteToArrayCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("I",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(TARGET(kHost),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();
