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

#pragma once
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
class TileCompute
    : public KernelLite<TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() {
    auto& param = Param<operators::TileParam>();
    auto rank = std::max(param.X->dims().size(), param.repeat_times.size());
    switch (rank) {
      case 1:
        Tile<1>(param);
        break;
      case 2:
        Tile<2>(param);
        break;
      case 3:
        Tile<3>(param);
        break;
      case 4:
        Tile<4>(param);
        break;
      case 5:
        Tile<5>(param);
        break;
      case 6:
        Tile<6>(param);
        break;
      default:
        paddle::lite::VoidifyFatal()
            << "rank must be greater than zero and less than 6";
    }
  }

  virtual ~TileCompute() = default;

 protected:
  template <int Rank>
  void Tile(operators::TileParam param) {
    auto repeat_times = param.repeat_times;
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
    fliud::Eigen::DSizes<int, Rank> bcast_dims;
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
    auto x = fluid::EigenTensor<T, Rank>::From(*in, new_in_dims);
    auto y = fluid::EigenTensor<T, Rank>::From(*out, out_dims);
    y = x.broadcast(bcast_dims);
  }
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
