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

#include "collision/convex_hull_conversions.h"

#include <vector>

#include "eigenmath/conversions.h"
#include "eigenmath/types.h"
#include "genit/transform_iterator.h"

namespace mobility::collision {
namespace {
eigenmath::Vector2d EigenVector2dFromProto(
    const eigenmath::Vector2dProto& proto) {
  return eigenmath::conversions::EigenVectorFromProto(proto);
}
}  // namespace

EagerConvexHull CreateEagerConvexHull(
    const std::vector<eigenmath::Vector2dProto>& pointcloud) {
  return EagerConvexHull(genit::CopyRange<std::vector<eigenmath::Vector2d>>(
      genit::TransformRange(pointcloud, EigenVector2dFromProto)));
}

std::vector<eigenmath::Vector2dProto> GetPoints(
    const EagerConvexHull& convex_hull) {
  return genit::CopyRange<std::vector<eigenmath::Vector2dProto>>(
      genit::TransformRange(convex_hull.GetPoints(),
                            eigenmath::conversions::ProtoFromVector2d));
}

eigenmath::Vector2dProto GetCentroid(const EagerConvexHull& convex_hull) {
  return eigenmath::conversions::ProtoFromVector2d(convex_hull.GetCentroid());
}

}  // namespace mobility::collision
