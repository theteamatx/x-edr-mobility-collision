// Copyright 2023 Google LLC
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

#include "collision/hull_conversion.h"

#include <vector>

#include "absl/status/status.h"
#include "collision/hull.h"

namespace mobility::collision {

absl::Status FromProto(const HullProto& proto, Hull* data_out) {
  data_out->Clear();
  std::vector<eigenmath::Vector2d> pointcloud;
  for (int ii = 0; ii < proto.convex_hulls_size(); ++ii) {
    auto& pc_proto = proto.convex_hulls(ii).points();
    pointcloud.clear();
    pointcloud.reserve(pc_proto.size());
    for (auto& pt_proto : pc_proto) {
      pointcloud.emplace_back(pt_proto.vec(0), pt_proto.vec(1));
    }
    data_out->Add(pointcloud);
  }
  return absl::OkStatus();
}

absl::Status ToProto(const Hull& data_in, HullProto* proto_out) {
  auto& convex_hulls = data_in.GetConvexHulls();
  for (int ii = 0; ii < convex_hulls.size(); ++ii) {
    auto pc_proto = proto_out->mutable_convex_hulls()->Add()->mutable_points();
    for (const auto& pt : convex_hulls[ii].GetPoints()) {
      auto* pt_proto = pc_proto->Add();
      pt_proto->mutable_vec()->Add(pt.x());
      pt_proto->mutable_vec()->Add(pt.y());
    }
  }
  return absl::OkStatus();
}

}  // namespace mobility::collision
