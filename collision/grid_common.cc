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

#include "collision/grid_common.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace mobility::collision {

namespace {
template <typename RangeContainer>
void RemoveEmptyRanges(RangeContainer *ranges) {
  ranges->resize(
      std::remove_if(ranges->begin(), ranges->end(),
                     [](const GridRange &range) { return range.Empty(); }) -
      ranges->begin());
}
template <typename RangeContainer>
void IntersectInplaceImpl(const GridRange &aa, RangeContainer *bb) {
  for (auto &result_range : *bb) {
    result_range.Intersect(aa);
  }
  RemoveEmptyRanges(bb);
}
template <typename RangeContainer>
GridRange SpanningUnionImpl(const RangeContainer &ranges) {
  GridRange result;
  for (auto &range : ranges) {
    result.SpanningUnion(range);
  }
  return result;  // NRVO
}
template <typename RangeContainer>
bool ContainsImpl(const RangeContainer &aa, const GridIndex &index) {
  for (auto &range : aa) {
    if (range.Contains(index)) {
      return true;
    }
  }
  return false;
}

// Returns the corner indices of the grid range.
std::array<GridIndex, 4> ExtremalIndicesOfGridRange(const GridRange &range) {
  return {range.lower,
          {range.lower.x(), range.upper.y() - 1},
          {range.upper.x() - 1, range.upper.y() - 1},
          {range.upper.x() - 1, range.lower.y()}};
}

}  // namespace

GridRange GridRange::GrowToInclude(const GridRange &range,
                                   const GridIndex &index) {
  if (range.Empty()) {
    return GridRange(index, index + GridIndex{1, 1});
  }
  return GridRange({std::min(range.lower.x(), index.x()),
                    std::min(range.lower.y(), index.y())},
                   {std::max(range.upper.x(), index.x() + 1),
                    std::max(range.upper.y(), index.y() + 1)});
}

GridRange GridRange::SpanningUnion(const GridRange &aa, const GridRange &bb) {
  if (aa.Empty()) {
    return bb;
  }
  if (bb.Empty()) {
    return aa;
  }
  return GrowToInclude(GrowToInclude(aa, bb.lower), bb.upper - GridIndex{1, 1});
}

GridRange GridRange::SpanningUnion(const std::vector<GridRange> &ranges) {
  return SpanningUnionImpl(ranges);
}

GridRange GridRange::SpanningUnion(const Quad &ranges) {
  return SpanningUnionImpl(ranges);
}

GridRange GridRange::Intersect(const GridRange &aa, const GridRange &bb) {
  if (aa.Empty() || bb.Empty()) {
    return GridRange();
  }
  return GridRange({std::max(aa.lower.x(), bb.lower.x()),
                    std::max(aa.lower.y(), bb.lower.y())},
                   {std::min(aa.upper.x(), bb.upper.x()),
                    std::min(aa.upper.y(), bb.upper.y())});
}

GridRange::Quad GridRange::Complement(const GridRange &aa) {
  constexpr int kIntMin = std::numeric_limits<int>::min();
  constexpr int kIntMax = std::numeric_limits<int>::max();
  Quad result;
  if (aa.Empty()) {
    result.resize(1);
    result[0] = GridRange({kIntMin, kIntMin}, {kIntMax, kIntMax});
  } else {
    result.resize(4);
    result[0] = GridRange({kIntMin, aa.lower.y()}, {aa.lower.x(), kIntMax});
    result[1] = GridRange({aa.lower.x(), aa.upper.y()}, {kIntMax, kIntMax});
    result[2] = GridRange({aa.upper.x(), kIntMin}, {kIntMax, aa.upper.y()});
    result[3] = GridRange({kIntMin, kIntMin}, {aa.upper.x(), aa.lower.y()});
  }
  return result;  // NRVO
}

GridRange::Quad GridRange::Difference(const GridRange &aa,
                                      const GridRange &bb) {
  Quad differences;
  if (aa.Empty()) {
    return differences;  // NRVO
  }
  if (bb.Empty()) {
    differences.resize(1);
    differences[0] = aa;
    return differences;  // NRVO
  }
  differences = Complement(bb);
  IntersectInplaceImpl(aa, &differences);
  return differences;  // NRVO
}

GridRange::Quad GridRange::NonSpanningUnion(const GridRange &aa,
                                            const GridRange &bb) {
  Quad result;
  if (aa.Contains(bb)) {
    result.resize(1);
    result[0] = aa;
    return result;  // NRVO
  }
  if (bb.Contains(aa)) {
    result.resize(1);
    result[0] = bb;
    return result;  // NRVO
  }
  result = Difference(aa, bb);
  if (!bb.Empty()) {
    const int i = result.size();
    result.resize(i + 1);
    result[i] = bb;
  }
  return result;  // NRVO
}

std::vector<GridRange> GridRange::Intersect(const GridRange &aa,
                                            const std::vector<GridRange> &bb) {
  std::vector<GridRange> result = bb;
  IntersectInplaceImpl(aa, &result);
  return result;  // NRVO
}

GridRange::Quad GridRange::Intersect(const GridRange &aa, const Quad &bb) {
  Quad result = bb;
  IntersectInplaceImpl(aa, &result);
  return result;  // NRVO
}

void GridRange::RemoveSelfIntersections(std::vector<GridRange> *ranges) {
  for (int head = 0, next_head = 1; head < ranges->size(); head = next_head++) {
    if ((*ranges)[head].Empty()) {
      ranges->erase(ranges->begin() + head);
      --head;
      --next_head;
      continue;
    }
    for (int tail = 0; tail < head; ++tail) {
      for (int current = head; current < next_head; ++current) {
        const GridRange current_range = (*ranges)[current];
        const GridRange inter = Intersect((*ranges)[tail], current_range);
        if (inter.Empty()) {
          continue;
        }
        ranges->erase(ranges->begin() + current);
        --current;
        --next_head;
        const Quad inter_complement = Complement(inter);
        for (const GridRange &comp : inter_complement) {
          const GridRange diff = Intersect(current_range, comp);
          if (!diff.Empty()) {
            ++current;
            ++next_head;
            ranges->insert(ranges->begin() + current, diff);
          }
        }
      }
    }
  }
}

bool GridRange::Contains(const std::vector<GridRange> &aa,
                         const GridIndex &index) {
  return ContainsImpl(aa, index);
}

bool GridRange::Contains(const Quad &aa, const GridIndex &index) {
  return ContainsImpl(aa, index);
}

GridRange GridFrame::GridToGridInclusive(const GridFrame &src_frame,
                                         const GridIndex &src_index,
                                         const GridFrame &dst_frame) {
  DCHECK_EQ(src_frame.frame_id, dst_frame.frame_id);
  const eigenmath::Vector2d src_center = src_frame.GridToFrame(src_index);
  const double shift_value = src_frame.resolution / 2.0 * M_SQRT2;
  return dst_frame.FrameCircleToGridRange(src_center, shift_value);
}

GridFrame::GridToGridFunctor GridFrame::GridToGrid(const GridFrame &src_frame,
                                                   const GridFrame &dst_frame) {
  DCHECK_EQ(src_frame.frame_id, dst_frame.frame_id);
  // Get a transform between the frame origins, using the Pose2d inverse for
  // better accuracy.
  const eigenmath::Pose2d dst_origin_pose_src_origin =
      dst_frame.origin.inverse() * src_frame.origin;
  const eigenmath::Matrix3d dst_affine_src =
      eigenmath::Vector3d{1 / dst_frame.resolution, 1 / dst_frame.resolution, 1}
          .asDiagonal() *
      dst_origin_pose_src_origin.matrix() *
      eigenmath::Vector3d{src_frame.resolution, src_frame.resolution, 1}
          .asDiagonal();
  return GridToGridFunctor{Eigen::Affine2d{dst_affine_src.block<2, 3>(0, 0)}};
}

GridRange SmallestTargetRangeCoveringSourceRange(
    const GridRange &source_range,
    const GridFrame::GridToGridFunctor &target_from_source) {
  GridRange target_range;
  if (source_range.Empty()) {
    return target_range;
  }
  // Transform corner indices to target frame.
  for (const GridIndex &corner : ExtremalIndicesOfGridRange(source_range)) {
    target_range.GrowToInclude(target_from_source(corner));
  }
  return target_range;
}

GridRange FullTargetRangeProjectingOntoSourceRange(
    const GridRange &source_range,
    const GridFrame::GridToGridFunctor &target_from_source) {
  return SmallestTargetRangeCoveringSourceRange(
      GridRange::GrowBy(source_range, 1), target_from_source);
}

GridLine::GridLine(const GridIndex &from, const GridIndex &to) : from_(from) {
  GridIndex difference = to - from;
  const int x_length = std::abs(difference.x());
  const int y_length = std::abs(difference.y());

  if (difference == GridIndex::Zero()) {
    // Avoid division by 0.
    length_ = 1;
    delta_ = eigenmath::Vector2d::Zero();
  } else if (x_length > y_length) {
    // Walk along x.
    length_ = x_length + 1;
    const double delta = (difference.x() > 0) ? 1.0 : -1.0;
    const double slope = delta * difference.y() / difference.x();
    delta_ = {delta, slope};
  } else {
    // Walk along y.
    length_ = y_length + 1;
    const double delta = (difference.y() > 0) ? 1.0 : -1.0;
    const double slope = delta * difference.x() / difference.y();
    delta_ = {slope, delta};
  }
}

GridLine::Iterator::Iterator(const GridLine &range, int index)
    : current_(range.from_.cast<double>() + index * range.delta_),
      delta_(range.delta_),
      index_(index) {}

}  // namespace mobility::collision
