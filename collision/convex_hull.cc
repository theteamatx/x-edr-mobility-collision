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

#include "collision/convex_hull.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "collision/collision_hull_augmentation.h"
#include "collision/oriented_box_2d.h"
#include "eigenmath/scalar_utils.h"
#include "eigenmath/types.h"
#include "eigenmath/utils.h"
#include "eigenmath/vector_utils.h"
#include "genit/adjacent_circular_iterator.h"
#include "genit/circular_iterator.h"
#include "genit/iterator_range.h"
#include "genit/zip_iterator.h"

namespace mobility::collision {

namespace {
constexpr double kEpsilon = 1e-12;

// Trims interior points, and returns an iterator past the last point.
template <typename Iterator>
Iterator RemoveInnerPoints(Iterator begin, Iterator end) {
  // Writes to last, reads from next.
  auto last = begin;
  for (auto next = std::next(last); next != end; ++next) {
    while (last != begin) {
      const auto prev = std::prev(last);
      const eigenmath::Vector2d OA = *last - *prev;
      const eigenmath::Vector2d OB = *next - *prev;
      if (eigenmath::CrossProduct(OA, OB) > 0) {
        break;
      }
      --last;
    }
    // Skip check for self-assignment.
    *++last = *next;
  }
  return std::next(last);
}

// Assumes that points describe a convex hull and are in counter clockwise
// order. Rhs can be empty.
void MergeCounterClockwisePoints(
    const std::vector<eigenmath::Vector2d> &ccw_lhs,
    const std::vector<eigenmath::Vector2d> &ccw_rhs,
    std::vector<eigenmath::Vector2d> *convex_hull_points) {
  auto &points = *convex_hull_points;
  CHECK_GT(ccw_lhs.size(), 2);
  if (ccw_rhs.empty()) {
    points = ccw_lhs;
    return;
  }
  CHECK_GT(ccw_rhs.size(), 2);

  // Keep buffer for all input points and repeated boundary points.
  points.resize(ccw_lhs.size() + ccw_rhs.size() + 3);

  // Compares two points, places points to the left first.  For ties places
  // lower point first.
  const auto compare_left_bottom = [](eigenmath::Vector2d p1,
                                      eigenmath::Vector2d p2) {
    return (p1.x() < p2.x()) || ((p1.x() == p2.x()) && (p1.y() < p2.y()));
  };

  // Find ranges for bottom and top halfs using a single sweep.
  const auto minmax_lhs =
      std::minmax_element(ccw_lhs.begin(), ccw_lhs.end(), compare_left_bottom);
  const auto minmax_rhs =
      std::minmax_element(ccw_rhs.begin(), ccw_rhs.end(), compare_left_bottom);
  // Get indices to convert to circular iterators, ensuring that min index comes
  // before max index.
  const int min_lhs = std::distance(ccw_lhs.begin(), minmax_lhs.first);
  const int max_lhs =
      min_lhs +
      ((std::distance(minmax_lhs.first, minmax_lhs.second) + ccw_lhs.size()) %
       ccw_lhs.size());
  const int min_rhs = std::distance(ccw_rhs.begin(), minmax_rhs.first);
  const int max_rhs =
      min_rhs +
      ((std::distance(minmax_rhs.first, minmax_rhs.second) + ccw_rhs.size()) %
       ccw_rhs.size());
  // Use circular ranges and add both boundary points to both bottom and top
  // half.
  using CircIt = genit::CircularIterator<decltype(ccw_lhs.begin())>;
  const CircIt circular_lhs = CircIt(ccw_lhs.begin(), ccw_lhs.end());
  const CircIt circular_rhs = CircIt(ccw_rhs.begin(), ccw_rhs.end());
  const auto bottom_lhs = genit::IteratorRange<CircIt>(
      circular_lhs + min_lhs, circular_lhs + max_lhs + 1);
  const auto bottom_rhs = genit::IteratorRange<CircIt>(
      circular_rhs + min_rhs, circular_rhs + max_rhs + 1);
  const auto top_lhs = genit::IteratorRange<CircIt>(
      circular_lhs + max_lhs, circular_lhs + min_lhs + 1 + ccw_lhs.size());
  const auto top_rhs = genit::IteratorRange<CircIt>(
      circular_rhs + max_rhs, circular_rhs + min_rhs + 1 + ccw_rhs.size());

  // Merge bottom half including boundary points to both sides.
  auto last = points.begin();
  last = std::merge(bottom_lhs.begin(), bottom_lhs.end(), bottom_rhs.begin(),
                    bottom_rhs.end(), last, compare_left_bottom);

  // Trim interior points from bottom half.
  last = RemoveInnerPoints(points.begin(), last);
  // Remove last point as it will be added again below.
  --last;
  const auto bottom_end = last;

  // Merge top half including boundary points to both sides.
  auto compare_right_top = [](eigenmath::Vector2d p1, eigenmath::Vector2d p2) {
    return (p1.x() > p2.x()) || ((p1.x() == p2.x()) && (p1.y() > p2.y()));
  };
  last = std::merge(top_lhs.begin(), top_lhs.end(), top_rhs.begin(),
                    top_rhs.end(), last, compare_right_top);

  // Trim interior points from upper half.
  last = RemoveInnerPoints(bottom_end, last);
  // Remove last point which equal first point and was added above.
  --last;
  points.erase(last, points.end());

  // Prune repeated or close points.
  constexpr double kSquaredEpsilon = kEpsilon * kEpsilon;
  auto unique = points.begin() + 1;
  while ((unique != points.end()) &&
         ((*unique - *std::prev(unique)).squaredNorm() >= kSquaredEpsilon)) {
    ++unique;
  }
  --unique;
  for (last = std::next(unique); last != points.end(); ++last) {
    const eigenmath::Vector2d dp = *last - *unique;
    if (dp.squaredNorm() >= kSquaredEpsilon) {
      ++unique;
      *unique = *last;
    }
  }
  points.erase(last, points.end());
}

}  // namespace

EagerConvexHull::EagerConvexHull(std::vector<eigenmath::Vector2d> pointcloud) {
  CHECK_GT(pointcloud.size(), 2);

  // Construct counter-clockwise (CCW) convex hull.
  // Using Monotone Chain Convex Hull algorithm.

  // Sort the points lexicographically by x, with tie-break on y.
  std::vector<eigenmath::Vector2d> sorted_points = std::move(pointcloud);
  std::sort(sorted_points.begin(), sorted_points.end(),
            [](const eigenmath::Vector2d &p1, const eigenmath::Vector2d &p2) {
              return (p1.x() < p2.x()) ||
                     ((p1.x() == p2.x()) && (p1.y() < p2.y()));
            });

  // Build the lower half-hull:
  for (auto it = sorted_points.begin(), it_end = sorted_points.end();
       it != it_end; ++it) {
    while (points_.size() >= 2) {
      const int sz = points_.size();
      const eigenmath::Vector2d OA = points_[sz - 1] - points_[sz - 2];
      const eigenmath::Vector2d OB = *it - points_[sz - 2];
      if (eigenmath::CrossProduct(OA, OB) > 0) {
        break;
      }
      points_.pop_back();
    }
    points_.push_back(*it);
  }

  // Build the upper half-hull:
  const int lower_sz = points_.size();
  for (auto it = sorted_points.rbegin(), it_end = sorted_points.rend();
       it != it_end; ++it) {
    while (points_.size() > lower_sz) {
      const int sz = points_.size();
      const eigenmath::Vector2d OA = points_[sz - 1] - points_[sz - 2];
      const eigenmath::Vector2d OB = *it - points_[sz - 2];
      if (eigenmath::CrossProduct(OA, OB) > 0) {
        break;
      }
      points_.pop_back();
    }
    points_.push_back(*it);
  }
  // Remove last point:
  points_.pop_back();

  for (int ii = 0; ii < points_.size(); ++ii) {
    const int jj = (ii + 1) % points_.size();
    const eigenmath::Vector2d dp = points_[jj] - points_[ii];
    const double dp_norm = dp.norm();
    if (dp_norm < kEpsilon) {
      points_.erase(points_.begin() + ii);
      --ii;
    } else {
      deltas_.push_back(dp / dp_norm);
    }
  }

  ComputeCircle();
}

void EagerConvexHull::ComputeCircle() {
  CHECK_GT(points_.size(), 2)
      << "Convex hull cannot possibly have a valid area with only two points!";

  // Fix first point.
  const eigenmath::Vector2d p0 = points_.front();

  // Compute the centroid as the weighted average of the centroid of every
  // triangle in the fan created by all vectors from p0.
  double sum_of_cross = 0.0;
  eigenmath::Vector2d sum_of_deltas = eigenmath::Vector2d::Zero();
  for (auto it = std::next(points_.begin()); it != points_.end(); ++it) {
    const eigenmath::Vector2d dp2 = *it - p0;
    const eigenmath::Vector2d dp1 = *std::prev(it) - p0;
    const double cross = eigenmath::CrossProduct(dp1, dp2);
    sum_of_cross += cross;
    sum_of_deltas += (dp1 + dp2) * cross;
  }
  CHECK_GT(sum_of_cross, std::numeric_limits<double>::epsilon())
      << "Convex hull has zero area!";
  centroid_ = p0 + (1.0 / (3.0 * sum_of_cross)) * sum_of_deltas;

  radius_ = 0.0;
  for (const eigenmath::Vector2d &point : points_) {
    const eigenmath::Vector2d d_center = point - centroid_;
    const double dist_sqr = d_center.squaredNorm();
    if (dist_sqr > radius_) {
      radius_ = dist_sqr;
    }
  }
  radius_ = std::sqrt(radius_);
}

bool EagerConvexHull::TransformAndCopy(const eigenmath::Pose2d &rel_pose,
                                       const EagerConvexHull &orig_hull) {
  if (points_.size() != orig_hull.points_.size()) {
    return false;
  }
  for (int i = 0; i < points_.size(); ++i) {
    points_[i] = rel_pose * orig_hull.points_[i];
    deltas_[i] = rel_pose.so2() * orig_hull.deltas_[i];
  }
  centroid_ = rel_pose * orig_hull.centroid_;
  return true;
}

EagerConvexHull EagerConvexHull::CreateBiggerHull(double padding) const {
  EagerConvexHull bigger;
  bigger.CreateBiggerHull(points_, padding);
  return bigger;  // NRVO
}

void EagerConvexHull::CreateBiggerHull(
    const std::vector<eigenmath::Vector2d> &orig_hull, double padding) {
  CreateBiggerHullImpl(
      orig_hull,
      [padding](eigenmath::Vector2d u, double u_norm) -> eigenmath::Vector2d {
        return (padding / u_norm) * u;
      });
}

EagerConvexHull EagerConvexHull::CreateBiggerHull(
    absl::FunctionRef<eigenmath::Vector2d(eigenmath::Vector2d, double)>
        directional_padding) const {
  EagerConvexHull bigger;
  bigger.CreateBiggerHullImpl(points_, std::move(directional_padding));
  return bigger;
}

template <typename Functor>
void EagerConvexHull::CreateBiggerHullImpl(
    const std::vector<eigenmath::Vector2d> &orig_hull,
    Functor directional_padding) {
  // Note that constructor assures we have at least 3 points.

  points_.clear();
  deltas_.clear();
  for (auto triad : genit::AdjacentElementsCircularRange<3>(orig_hull)) {
    const eigenmath::Vector2d v0 = triad[0] - triad[1];
    const eigenmath::Vector2d v1 = triad[2] - triad[1];
    const double l0 = v0.norm();
    const double l1 = v1.norm();
    const eigenmath::Vector2d uu = (1.0 / l0) * v0 + (1.0 / l1) * v1;
    const double uu_norm = uu.norm();
    // Add 3 points outside each original point placed at:
    //  - Padding distance outside ingoing edge from corner point.
    //  - Padding distance outside the corner point, in middle direction.
    //  - Padding distance outside outgoing edge from the corner point.
    // If corner is almost flat, only add one point outside the ingoing edge.
    // Otherwise, add the other two points too.
    points_.push_back(triad[1] + directional_padding(
                                     eigenmath::Vector2d(-v0.y(), v0.x()), l0));
    if (uu_norm > kEpsilon) {
      points_.push_back(triad[1] + directional_padding(-uu, uu_norm));
      points_.push_back(
          triad[1] +
          directional_padding(eigenmath::Vector2d(v1.y(), -v1.x()), l1));
    }
  }

  for (auto segment : genit::AdjacentElementsCircularRange<2>(points_)) {
    const eigenmath::Vector2d dp = segment[1] - segment[0];
    deltas_.push_back(dp.normalized());
  }

  ComputeCircle();
}

void EagerConvexHull::FromCounterClockwiseExtremePoints(
    std::vector<eigenmath::Vector2d> ccw_convex_hull) {
  points_ = std::move(ccw_convex_hull);

  deltas_.clear();
  for (auto segment : genit::AdjacentElementsCircularRange<2>(points_)) {
    const eigenmath::Vector2d dp = segment[1] - segment[0];
    deltas_.push_back(dp.normalized());
  }

  ComputeCircle();
}

template <typename Derived>
ConvexHullBase<Derived>::operator const EagerConvexHull &() const {
  CacheBoundingCircle();
  CachePointsAndDeltas();
  return GetCache();
}

template <typename Derived>
bool ConvexHullBase<Derived>::Contains(const eigenmath::Vector2d &point) const {
  return ContainsWithCircle(point) && ContainsWithPoints(point);
}

template <typename Derived>
template <typename Other>
bool ConvexHullBase<Derived>::Contains(
    const ConvexHullBase<Other> &other_hull) const {
  CacheBoundingCircle();
  other_hull.CacheBoundingCircle();
  const eigenmath::Vector2d d_center =
      other_hull.GetCache().centroid_ - GetCache().centroid_;
  const double r2_center = d_center.squaredNorm();
  if (r2_center - kEpsilon >
      eigenmath::Square(GetCache().radius_ + other_hull.GetCache().radius_)) {
    return false;
  }
  other_hull.CachePointsAndDeltas();
  for (const eigenmath::Vector2d &pt : other_hull.GetCache().points_) {
    if (!Contains(pt)) {
      return false;
    }
  }
  return true;
}

template <typename Derived>
template <typename Rhs>
bool ConvexHullBase<Derived>::IsApprox(const ConvexHullBase<Rhs> &rhs,
                                       double tolerance) const {
  CachePointsAndDeltas();
  rhs.CachePointsAndDeltas();
  const double tol_sqr = tolerance * tolerance;
  if ((GetCache().points_.size() != rhs.GetCache().points_.size()) ||
      ((GetCache().centroid_ - rhs.GetCache().centroid_).squaredNorm() >
       tol_sqr) ||
      (std::abs(GetCache().radius_ - rhs.GetCache().radius_) > tolerance)) {
    return false;
  }
  for (int offset = 0; offset < GetCache().points_.size(); ++offset) {
    bool found_mismatch = false;
    for (int i = 0; i < GetCache().points_.size(); ++i) {
      const int j = (i + offset) % GetCache().points_.size();
      if ((GetCache().points_[i] - rhs.GetCache().points_[j]).squaredNorm() >
          tol_sqr) {
        found_mismatch = true;
        break;
      }
    }
    if (!found_mismatch) {
      return true;
    }
  }
  return false;
}

template <typename Derived>
double ConvexHullBase<Derived>::Distance(
    const eigenmath::Vector2d &point) const {
  // Note that constructor assures we have at least 3 points.
  CachePointsAndDeltas();
  double min_positive_sqr = std::numeric_limits<double>::max();
  double max_negative = std::numeric_limits<double>::lowest();
  bool is_inside = true;
  for (int ii = 0; ii < GetCache().points_.size(); ++ii) {
    const eigenmath::Vector2d dq1 = point - GetCache().points_[ii];
    const double normal_dist =
        eigenmath::CrossProduct(dq1, GetCache().deltas_[ii]);
    is_inside = is_inside && (normal_dist <= 0.0);
    if (normal_dist > 0.0) {
      const double dist1 = eigenmath::DotProduct(GetCache().deltas_[ii], dq1);
      double real_dist_sqr = normal_dist * normal_dist;
      if (dist1 < 0.0) {
        // the point is 'behind' GetCache().points__[ii]
        real_dist_sqr += dist1 * dist1;
      } else {
        const int jj = (ii + 1) % GetCache().points_.size();
        const eigenmath::Vector2d dq2 = point - GetCache().points_[jj];
        const double dist2 = eigenmath::DotProduct(GetCache().deltas_[ii], dq2);
        if (dist2 > 0.0) {
          // the point is 'ahead' of GetCache().points_[jj]
          real_dist_sqr += dist2 * dist2;
        }
      }
      min_positive_sqr = std::min(real_dist_sqr, min_positive_sqr);
    } else {
      max_negative = std::max(normal_dist, max_negative);
    }
  }
  return (is_inside ? max_negative : std::sqrt(min_positive_sqr));
}

template <typename Derived>
double ConvexHullBase<Derived>::DistanceIfPenetrating(
    const eigenmath::Vector2d &point) const {
  if (!ContainsWithCircle(point)) {
    return 1.0;
  }
  return DistanceIfLessThanWithPoints(point);
}

template <typename Derived>
double ConvexHullBase<Derived>::DistanceIfLessThan(
    const eigenmath::Vector2d &point, double current_distance) const {
  if (!ContainsWithCircle(point, current_distance)) {
    return 1.0;
  }
  return DistanceIfLessThanWithPoints(point, current_distance);
}

template <typename Derived>
double ConvexHullBase<Derived>::ClosestPointIfPenetrating(
    const eigenmath::Vector2d &point,
    eigenmath::Vector2d *closest_point) const {
  if (!ContainsWithCircle(point)) {
    return 1.0;
  }
  return DistanceIfLessThanWithPoints(point, 0.0, closest_point);
}

template <typename Derived>
bool ConvexHullBase<Derived>::ContainsWithCircle(
    const eigenmath::Vector2d &point, double current_distance) const {
  CacheBoundingCircle();
  const double r2_center = (point - GetCache().centroid_).squaredNorm();
  return r2_center - kEpsilon <=
         eigenmath::Square(GetCache().radius_ + current_distance);
}

template <typename Derived>
template <typename Other>
bool ConvexHullBase<Derived>::ContainsWithCircle(
    const ConvexHullBase<Other> &other_hull, double current_distance) const {
  CacheBoundingCircle();
  other_hull.CacheBoundingCircle();
  const double r2_center =
      (other_hull.GetCache().centroid_ - other_hull.GetCache().centroid_)
          .squaredNorm();
  if (r2_center - kEpsilon >
      eigenmath::Square(GetCache().radius_ + current_distance +
                        other_hull.GetCache().radius_)) {
    return false;
  }
  other_hull.CachePointsAndDeltas();
  for (const eigenmath::Vector2d &pt : other_hull.GetCache().points_) {
    if (!ContainsWithCircle(pt, current_distance)) {
      return false;
    }
  }
  return true;
}

template <typename Derived>
bool ConvexHullBase<Derived>::ContainsWithPoints(
    const eigenmath::Vector2d &point) const {
  // Note that constructor assures we have at least 3 points.
  CachePointsAndDeltas();
  for (int ii = 0; ii < GetCache().points_.size(); ++ii) {
    const eigenmath::Vector2d dq = point - GetCache().points_[ii];
    if (eigenmath::CrossProduct(GetCache().deltas_[ii], dq) < -kEpsilon) {
      return false;
    }
  }
  return true;
}

template <typename Derived>
double ConvexHullBase<Derived>::DistanceIfLessThanWithPoints(
    const eigenmath::Vector2d &point, double current_distance,
    eigenmath::Vector2d *closest_point) const {
  // Note that constructor assures we have at least 3 points.
  CachePointsAndDeltas();
  double max_negative = std::numeric_limits<double>::lowest();
  for (int ii = 0; ii < GetCache().points_.size(); ++ii) {
    const eigenmath::Vector2d dq1 = point - GetCache().points_[ii];
    const double normal_dist =
        eigenmath::CrossProduct(dq1, GetCache().deltas_[ii]);
    if (normal_dist > 0.0) {
      return 1.0;  // It cannot be penetrating.
    }
    if (normal_dist > max_negative) {
      max_negative = normal_dist;
      if (closest_point) {
        *closest_point = point + normal_dist * eigenmath::RightOrthogonal(
                                                   GetCache().deltas_[ii]);
      }
    }
  }
  if (max_negative > current_distance) {
    return 1.0;  // It's not penetrating more than current distance.
  }
  return max_negative;
}

template <typename Derived>
OrientedBox2d ConvexHullBase<Derived>::GetMinAreaBoundingBox() const {
  CachePointsAndDeltas();
  const int n = GetCache().points_.size();
  double min_area = std::numeric_limits<double>::max();
  eigenmath::Vector2d leftmost{0.0, 0.0};
  eigenmath::Vector2d bottommost{0.0, 0.0};
  eigenmath::Vector2d major_basis{0.0, 0.0};
  eigenmath::Vector2d smallest_dims{0.0, 0.0};

  // Start with aligned extremal points.
  std::array<int, 4> corners = {0, 0, 0, 0};
  for (auto [i, pt] : genit::EnumerateRange(GetCache().points_)) {
    // Bottom.
    if (pt.y() < GetCache().points_[corners[0]].y()) {
      corners[0] = i;
    }
    // Right.
    if (pt.x() > GetCache().points_[corners[1]].x()) {
      corners[1] = i;
    }
    // Top.
    if (pt.y() > GetCache().points_[corners[2]].y()) {
      corners[2] = i;
    }
    // Left.
    if (pt.x() < GetCache().points_[corners[3]].x()) {
      corners[3] = i;
    }
  }

  eigenmath::Vector2d basis{1.0, 0.0};
  for (int i = 0; i < n; ++i) {
    // Find the dominant edge.
    double max_cosine = std::numeric_limits<double>::lowest();
    int dominant_edge = 0;
    for (int j = 0; j < 4; ++j) {
      double dp = eigenmath::DotProduct(GetCache().deltas_[corners[j]], basis);
      basis = eigenmath::RightOrthogonal(basis);
      if (dp > max_cosine) {
        dominant_edge = j;
        max_cosine = dp;
      }
    }

    basis = GetCache().deltas_[corners[dominant_edge]];
    for (int j = 0; j < dominant_edge; ++j) {
      basis = eigenmath::LeftOrthogonal(basis);
    }

    // Move dominant edge forward.
    corners[dominant_edge] = (corners[dominant_edge] + 1) % n;

    // Left to right edge.
    eigenmath::Vector2d edge =
        GetCache().points_[corners[1]] - GetCache().points_[corners[3]];
    const double width = eigenmath::DotProduct(edge, basis);

    // Bottom to top edge.
    edge = GetCache().points_[corners[2]] - GetCache().points_[corners[0]];
    const double height =
        eigenmath::DotProduct(edge, eigenmath::RightOrthogonal(basis));

    // Check area.
    const double area = width * height;
    if (area <= min_area) {
      min_area = area;
      leftmost = GetCache().points_[corners[3]];
      bottommost = GetCache().points_[corners[0]];
      major_basis = basis;
      smallest_dims = eigenmath::Vector2d{width, height};
    }
  }

  const eigenmath::Vector2d minor_basis =
      eigenmath::RightOrthogonal(major_basis);

  const double left_proj = eigenmath::DotProduct(major_basis, leftmost);
  const double bottom_proj = eigenmath::DotProduct(minor_basis, bottommost);

  const eigenmath::Vector2d corner =
      (bottom_proj * minor_basis + left_proj * major_basis) /
      eigenmath::CrossProduct(major_basis, minor_basis);

  return {eigenmath::Pose2d(eigenmath::Vector2d(
                                corner + major_basis * 0.5 * smallest_dims.x() +
                                minor_basis * 0.5 * smallest_dims.y()),
                            eigenmath::SO2d(major_basis.x(), major_basis.y())),
          smallest_dims};
}

template <typename Derived>
eigenmath::Vector2d ConvexHullBase<Derived>::GetFarthestPointInDirection(
    const eigenmath::Vector2d &direction) const {
  double max_proj = std::numeric_limits<double>::lowest();
  double prev_proj = std::numeric_limits<double>::lowest();
  eigenmath::Vector2d result_pt = eigenmath::Vector2d::Zero();
  int rising_count = 0;
  CachePointsAndDeltas();
  for (const eigenmath::Vector2d &pt : GetCache().points_) {
    const double proj = eigenmath::DotProduct(direction, pt);
    if (proj > max_proj || (proj == max_proj && proj > prev_proj)) {
      ++rising_count;
      max_proj = proj;
      result_pt = pt;
    } else if (rising_count > 1) {
      return result_pt;
    }
    prev_proj = proj;
  }
  return result_pt;
}

template <typename Derived>
template <typename Lhs, typename Rhs>
eigenmath::Vector2d ConvexHullBase<Derived>::GetMinkowskiDiffSupport(
    const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
    eigenmath::Vector2d axis) {
  const eigenmath::Vector2d p_lhs = lhs.GetFarthestPointInDirection(axis);
  const eigenmath::Vector2d p_rhs = rhs.GetFarthestPointInDirection(-axis);
  return eigenmath::Vector2d(p_lhs - p_rhs);
}

template <typename Derived>
template <typename Lhs, typename Rhs>
bool ConvexHullBase<Derived>::FindGJKSimplex(
    const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
    std::array<eigenmath::Vector2d, 3> *simplex) {
  // Initialize an empty simplex.
  const eigenmath::Vector2d kInfVector =
      eigenmath::Vector2d::Constant(std::numeric_limits<double>::infinity());
  (*simplex)[0] = kInfVector;
  (*simplex)[1] = kInfVector;
  (*simplex)[2] = kInfVector;

  // Run the GJK algorithm.
  lhs.CacheBoundingCircle();
  rhs.CacheBoundingCircle();
  lhs.CachePointsAndDeltas();
  rhs.CachePointsAndDeltas();
  eigenmath::Vector2d axis =
      rhs.GetCache().centroid_ - lhs.GetCache().centroid_;
  if (axis.squaredNorm() == 0) {
    axis = {1.0, 0.0};
  }
  int simplex_idx = 0;
  (*simplex)[simplex_idx] = GetMinkowskiDiffSupport(lhs, rhs, axis);
  axis = -axis;
  ++simplex_idx;
  while (true) {
    (*simplex)[simplex_idx] = GetMinkowskiDiffSupport(lhs, rhs, axis);
    if ((*simplex)[simplex_idx].dot(axis) <= 0.0) {
      return false;
    }
    const eigenmath::Vector2d ao = -(*simplex)[0];
    // Check for first iteration, where simplex is a line segment.
    if (!std::isfinite((*simplex)[2].x())) {
      const eigenmath::Vector2d ab = (*simplex)[1] - (*simplex)[0];
      axis = eigenmath::TripleProduct(ab, ao, ab);
      if (axis.squaredNorm() == 0) {
        axis = eigenmath::LeftOrthogonal(ab);
      }
      simplex_idx = 2;
      continue;
    }
    // Simplex is a triangle (main case), find the best point to "flip".
    const eigenmath::Vector2d ab = (*simplex)[1] - (*simplex)[0];
    const eigenmath::Vector2d ac = (*simplex)[2] - (*simplex)[0];
    const eigenmath::Vector2d ab_perp = eigenmath::TripleProduct(ac, ab, ab);
    if (ab_perp.dot(ao) > 0.0) {
      // Remove point c.
      simplex_idx = 2;
      axis = ab_perp;
      continue;
    }
    const eigenmath::Vector2d ac_perp = eigenmath::TripleProduct(ab, ac, ac);
    if (ac_perp.dot(ao) > 0.0) {
      // Remove point b.
      simplex_idx = 1;
      axis = ac_perp;
      continue;
    }
    const eigenmath::Vector2d bo = -(*simplex)[1];
    const eigenmath::Vector2d bc = (*simplex)[2] - (*simplex)[1];
    const eigenmath::Vector2d bc_perp = -eigenmath::TripleProduct(ab, bc, bc);
    if (bc_perp.dot(bo) > 0.0) {
      // Remove point a.
      simplex_idx = 0;
      axis = bc_perp;
      continue;
    }
    // Already contains the origin, return true.
    return true;
  }
}

template <typename Derived>
template <typename Lhs, typename Rhs>
double ConvexHullBase<Derived>::FindSeparationDistance(
    const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
    std::array<eigenmath::Vector2d, 3> simplex, double min_distance,
    double max_distance) {
  const eigenmath::Vector2d kOrigin = eigenmath::Vector2d::Zero();

  CHECK(std::isfinite(simplex[0].x()) && std::isfinite(simplex[0].y()));
  CHECK(std::isfinite(simplex[1].x()) && std::isfinite(simplex[1].y()));
  // Check if we are starting with a degenerate simplex, only a line-segment.
  if (!std::isfinite(simplex[2].x())) {
    // Generate a new point to complete the simplex.
    const eigenmath::Vector2d ao = -simplex[0];
    const eigenmath::Vector2d ab = simplex[1] - simplex[0];
    simplex[2] =
        GetMinkowskiDiffSupport(lhs, rhs, eigenmath::TripleProduct(ab, ao, ab));
  }

  // Continue the GJK algorithm until we are as close as can be to origin.
  double dists[3];
  double params[3];
  eigenmath::DistanceFromLineSegment(simplex[0], simplex[1], kOrigin, &dists[0],
                                     &params[0]);
  eigenmath::DistanceFromLineSegment(simplex[1], simplex[2], kOrigin, &dists[1],
                                     &params[1]);
  eigenmath::DistanceFromLineSegment(simplex[2], simplex[0], kOrigin, &dists[2],
                                     &params[2]);
  if (max_distance <= 0.0) {
    // Caller is not interested in separation distance.
    return *std::min_element(dists, dists + 3);
  }
  while (true) {
    if (std::abs(dists[0] - dists[1]) <
            std::numeric_limits<double>::epsilon() &&
        std::abs(dists[1] - dists[2]) <
            std::numeric_limits<double>::epsilon()) {
      // It looks like we've reached a cycle.
      return dists[0];
    }
    const int a_id = std::min_element(dists, dists + 3) - dists;
    if (dists[a_id] < min_distance) {
      return dists[a_id];
    }
    const int b_id = (a_id + 1) % 3;
    const int c_id = (b_id + 1) % 3;
    const eigenmath::Vector2d closest_pt =
        (simplex[b_id] - simplex[a_id]) * std::clamp(params[a_id], 0.0, 1.0) +
        simplex[a_id];
    // Calls CachePointsAndDeltas
    const eigenmath::Vector2d new_support =
        GetMinkowskiDiffSupport(lhs, rhs, -closest_pt);
    simplex[c_id] = new_support;
    const double c_proj = simplex[c_id].dot(-closest_pt);
    const double a_proj = simplex[a_id].dot(-closest_pt);
    if (c_proj - a_proj < kEpsilon) {
      // It looks like we can't make any more progress.
      return dists[a_id];
    }
    eigenmath::DistanceFromLineSegment(simplex[b_id], simplex[c_id], kOrigin,
                                       &dists[b_id], &params[b_id]);
    eigenmath::DistanceFromLineSegment(simplex[c_id], simplex[a_id], kOrigin,
                                       &dists[c_id], &params[c_id]);
  }
  return *std::min_element(dists, dists + 3);
}

template <typename Derived>
template <typename Lhs, typename Rhs>
double ConvexHullBase<Derived>::FindPenetrationDepth(
    const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
    const std::array<eigenmath::Vector2d, 3> &simplex, double min_distance) {
  // Use Expanding Polytope Algorithm to get the minimum distance.
  struct EPAEdge {
    eigenmath::Vector2d a, b;
    eigenmath::Vector2d normal;
    double normal_dist;
    EPAEdge(eigenmath::Vector2d a_, eigenmath::Vector2d b_) : a(a_), b(b_) {
      const eigenmath::Vector2d e = b - a;
      normal = eigenmath::TripleProduct(e, a, e).normalized();
      if (normal.squaredNorm() == 0.0) {
        // Oppose orthogonal in GJK simplex calculation.
        normal = eigenmath::RightOrthogonal(e).normalized();
      }
      normal_dist = normal.dot(a);
    }
    static bool PrioritizeClosest(const EPAEdge &lhs, const EPAEdge &rhs) {
      return lhs.normal_dist > rhs.normal_dist;
    }
  };

  lhs.CachePointsAndDeltas();
  rhs.CachePointsAndDeltas();
  std::vector<EPAEdge> edge_queue;
  const int minkowski_points_count =
      lhs.GetCache().points_.size() * rhs.GetCache().points_.size();
  edge_queue.reserve(minkowski_points_count);
  edge_queue.emplace_back(simplex[0], simplex[1]);
  edge_queue.emplace_back(simplex[1], simplex[2]);
  edge_queue.emplace_back(simplex[2], simplex[0]);
  std::make_heap(edge_queue.begin(), edge_queue.end(),
                 EPAEdge::PrioritizeClosest);
  auto pop_edge = [&]() {
    const EPAEdge closest_edge = edge_queue.front();
    std::pop_heap(edge_queue.begin(), edge_queue.end(),
                  EPAEdge::PrioritizeClosest);
    edge_queue.pop_back();
    return closest_edge;
  };
  auto maybe_push_edge = [&](eigenmath::Vector2d a, eigenmath::Vector2d b,
                             double closest_normal_dist) {
    const EPAEdge ab{a, b};
    // A new edge with smaller distance than its genitor is not convex.
    if (ab.normal_dist > closest_normal_dist) {
      edge_queue.push_back(ab);
      std::push_heap(edge_queue.begin(), edge_queue.end(),
                     EPAEdge::PrioritizeClosest);
    }
  };
  while (!edge_queue.empty() && edge_queue.size() <= minkowski_points_count) {
    const EPAEdge closest_edge = pop_edge();
    if (-closest_edge.normal_dist < min_distance) {
      // Caller is not interested in a better penetration depth estimate.
      return -closest_edge.normal_dist;
    }

    const eigenmath::Vector2d new_point =
        GetMinkowskiDiffSupport(lhs, rhs, closest_edge.normal);
    if (new_point.dot(closest_edge.normal) - closest_edge.normal_dist <
            kEpsilon ||
        (new_point - closest_edge.a).template lpNorm<Eigen::Infinity>() <
            kEpsilon ||
        (new_point - closest_edge.b).template lpNorm<Eigen::Infinity>() <
            kEpsilon) {
      // It looks like we can't make any more progress.
      return -closest_edge.normal_dist;
    }

    maybe_push_edge(closest_edge.a, new_point, closest_edge.normal_dist);
    maybe_push_edge(new_point, closest_edge.b, closest_edge.normal_dist);
  }
  CHECK(false) << absl::StrFormat(
      "This point should never be reached! Data:\n  lhs = \n%s\n  rhs = \n%s\n "
      " simplex = \n{%s, %s, %s}\n  min_distance = %f",
      absl::StrCat(lhs), absl::StrCat(rhs),
      absl::StrCat(eigenmath::AbslStringified(simplex[0])),
      absl::StrCat(eigenmath::AbslStringified(simplex[1])),
      absl::StrCat(eigenmath::AbslStringified(simplex[2])), min_distance);
  return 0.0;
}

template <typename Derived>
double ConvexHullBase<Derived>::GetMaxSquaredRadiusAround(
    const eigenmath::Vector2d &point) const {
  double max_radius_sqr = 0.0;
  CachePointsAndDeltas();
  for (const eigenmath::Vector2d &pt : GetCache().points_) {
    const double pt_radius_sqr = (pt - point).squaredNorm();
    if (pt_radius_sqr > max_radius_sqr) {
      max_radius_sqr = pt_radius_sqr;
    }
  }
  return max_radius_sqr;
}

template <typename Derived>
double ConvexHullBase<Derived>::GetMaxRadiusAround(
    const eigenmath::Vector2d &point) const {
  return std::sqrt(GetMaxSquaredRadiusAround(point));
}

eigenmath::Vector2d AlignedPaddingBox::operator()(eigenmath::Vector2d u,
                                                  double u_norm) const {
  // Divisions near zero in code below is fine, just goes near infinity.
  double factor = std::numeric_limits<double>::max();
  if (u.x() > 0.0) {
    factor = std::min(factor, padding_x_pos / u.x());
  } else if (u.x() < 0.0) {
    factor = std::min(factor, -padding_x_neg / u.x());
  }
  if (u.y() > 0.0) {
    factor = std::min(factor, padding_y_pos / u.y());
  } else if (u.y() < 0.0) {
    factor = std::min(factor, -padding_y_neg / u.y());
  }
  return factor * u;
}

template <typename Original>
LazyTransformedConvexHullBase<Original>::LazyTransformedConvexHullBase(
    const ConvexHullBase<Original> *orig_hull)
    : transform_() {
  ResetHullBase(orig_hull);
}

template <typename Original>
void LazyTransformedConvexHullBase<Original>::ResetHullBase(
    const ConvexHullBase<Original> *orig_hull) {
  orig_hull_ = orig_hull;
  has_transformed_centroid_ = false;
  has_transformed_points_ = false;
  // Reserve memory for maximum number of points.
  const int n_points_bound = orig_hull->UpperBoundNumberOfPoints();
  transformed_hull_.points_.reserve(n_points_bound);
  transformed_hull_.deltas_.reserve(n_points_bound);
}

template <typename Original>
bool LazyTransformedConvexHullBase<Original>::ApplyTransform(
    const eigenmath::Pose2d &transformed_pose_orig) {
  transform_ = transformed_pose_orig;
  has_transformed_centroid_ = false;
  has_transformed_points_ = false;
  return true;
}

template <typename Original>
void LazyTransformedConvexHullBase<Original>::CachePointsAndDeltas() const {
  if (!has_transformed_points_) {
    TransformPoints();
  }
}

template <typename Original>
void LazyTransformedConvexHullBase<Original>::CacheBoundingCircle() const {
  if (!has_transformed_centroid_) {
    TransformCentroid();
  }
}

template <typename Original>
void LazyTransformedConvexHullBase<Original>::TransformCentroid() const {
  orig_hull_->CacheBoundingCircle();
  transformed_hull_.centroid_ = transform_ * orig_hull_->GetCache().centroid_;
  transformed_hull_.radius_ = orig_hull_->GetCache().radius_;
  has_transformed_centroid_ = true;
}

template <typename Original>
void LazyTransformedConvexHullBase<Original>::TransformPoints() const {
  orig_hull_->CachePointsAndDeltas();
  transformed_hull_.points_.resize(orig_hull_->GetCache().points_.size());
  transformed_hull_.deltas_.resize(orig_hull_->GetCache().deltas_.size());
  for (int i = 0; i < orig_hull_->GetCache().points_.size(); ++i) {
    transformed_hull_.points_[i] =
        transform_ * orig_hull_->GetCache().points_[i];
    transformed_hull_.deltas_[i] =
        transform_.so2() * orig_hull_->GetCache().deltas_[i];
  }
  has_transformed_points_ = true;

  // If a lazy augmented convex hull is used, the centroid and radius can
  // change.
  if (has_transformed_centroid_) {
    TransformCentroid();
  }
}

template <typename Original>
LazyAugmentedConvexHullBase<Original>::LazyAugmentedConvexHullBase(
    const diff_drive::DynamicLimits *limits,
    const ConvexHullBase<Original> *orig_hull)
    : danger_margin_(0) {
  ResetHullBase(limits, orig_hull);
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::ResetHullBase(
    const diff_drive::DynamicLimits *limits,
    const ConvexHullBase<Original> *orig_hull) {
  orig_hull_ = orig_hull;
  limits_ = limits;
  has_augmented_bounding_circle_ = false;
  has_augmented_hull_ = false;
  // Reserve memory for maximum number of points.  Adding transformed points
  // can double the number of points.  Adding the padding can triple the
  // number of points.
  const int n_points_bound = orig_hull->UpperBoundNumberOfPoints();
  augmented_hull_.points_.reserve(6 * n_points_bound);
  augmented_hull_.deltas_.reserve(6 * n_points_bound);
  point_buffer_.reserve(2 * n_points_bound);
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::CachePointsAndDeltas() const {
  if (!has_augmented_hull_) {
    AugmentConvexHull();
  }
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::CacheBoundingCircle() const {
  if (!has_augmented_bounding_circle_) {
    AugmentBoundingCircle();
  }
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::ApplyMotion(
    const diff_drive::State &state) {
  // Invalidate cached data.
  has_augmented_bounding_circle_ = false;
  has_augmented_hull_ = false;

  state_ = state;
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::SetDangerMargin(
    double danger_margin) {
  CHECK_GE(danger_margin, 0);

  danger_margin_ = danger_margin;
  has_augmented_bounding_circle_ = false;
  has_augmented_hull_ = false;
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::AugmentBoundingCircle() const {
  CHECK_NE(limits_, nullptr);
  CHECK_NE(orig_hull_, nullptr);

  // Perform motion based augmentation.
  orig_hull_->CacheBoundingCircle();
  mobility::collision::AugmentBoundingCircle(
      *limits_, state_, orig_hull_->GetCache().centroid_,
      orig_hull_->GetCache().radius_, &augmented_hull_.centroid_,
      &augmented_hull_.radius_);

  // Add reaction zone.
  augmented_hull_.radius_ += danger_margin_;

  has_augmented_bounding_circle_ = true;
}

template <typename Original>
void LazyAugmentedConvexHullBase<Original>::AugmentConvexHull() const {
  CHECK_NE(limits_, nullptr);
  CHECK_NE(orig_hull_, nullptr);

  // Perform motion based augmentation, avoiding the convex hull constructor.
  orig_hull_->CachePointsAndDeltas();
  orig_hull_->CacheBoundingCircle();
  // Use augmented hull as unmerged point buffer and merge into point_buffer_.
  auto &unmerged_buffer = augmented_hull_.GetCache().points_;
  unmerged_buffer.clear();
  mobility::collision::AppendMomentaryStoppingPoints(
      *limits_, state_, orig_hull_->GetCache().points_, &unmerged_buffer);
  MergeCounterClockwisePoints(orig_hull_->GetCache().points_, unmerged_buffer,
                              &point_buffer_);

  // Add reaction zone.
  if (danger_margin_ > 0) {
    augmented_hull_.CreateBiggerHull(point_buffer_, danger_margin_);
  } else {
    augmented_hull_.FromCounterClockwiseExtremePoints(point_buffer_);
  }

  // Overwrites the bounding circle as well.
  has_augmented_bounding_circle_ = true;
  has_augmented_hull_ = true;
}

template <typename Lhs, typename Rhs>
double DistanceBetween(const ConvexHullBase<Lhs> &lhs,
                       const ConvexHullBase<Rhs> &rhs, double min_distance,
                       double max_distance) {
  lhs.CacheBoundingCircle();
  rhs.CacheBoundingCircle();
  // First check circle-to-circle distance against max_distance.
  const eigenmath::Vector2d c2c =
      rhs.GetCache().centroid_ - lhs.GetCache().centroid_;
  const double c2c_dist_sqr = c2c.squaredNorm();
  const double lhs_radius = lhs.GetCache().radius_;
  const double rhs_radius = rhs.GetCache().radius_;
  if (c2c_dist_sqr >
      eigenmath::Square(lhs_radius + max_distance + rhs_radius)) {
    return std::sqrt(c2c_dist_sqr) - lhs_radius - rhs_radius;
  }

  std::array<eigenmath::Vector2d, 3> simplex;
  const bool gjk_found_overlap = ConvexHull::FindGJKSimplex(lhs, rhs, &simplex);

  if (gjk_found_overlap) {
    return ConvexHull::FindPenetrationDepth(lhs, rhs, simplex, min_distance);
  } else {
    return ConvexHull::FindSeparationDistance(lhs, rhs, simplex, min_distance,
                                              max_distance);
  }
}

template <typename Lhs, typename Rhs>
bool AreOverlapping(const ConvexHullBase<Lhs> &lhs,
                    const ConvexHullBase<Rhs> &rhs) {
  lhs.CacheBoundingCircle();
  rhs.CacheBoundingCircle();
  // First check circle-to-circle overlap.
  const eigenmath::Vector2d c2c =
      rhs.GetCache().centroid_ - lhs.GetCache().centroid_;
  if (c2c.squaredNorm() >
      eigenmath::Square(lhs.GetCache().radius_ + rhs.GetCache().radius_)) {
    return false;
  }
  std::array<eigenmath::Vector2d, 3> simplex;
  return ConvexHull::FindGJKSimplex(lhs, rhs, &simplex);
}

template class ConvexHullBase<EagerConvexHull>;
template class ConvexHullBase<LazyAugmentedConvexHull>;
template class ConvexHullBase<LazyTransformedConvexHull>;
template class ConvexHullBase<LazyTransformedAugmentedConvexHull>;

template class LazyAugmentedConvexHullBase<EagerConvexHull>;
template class LazyTransformedConvexHullBase<EagerConvexHull>;
template class LazyTransformedConvexHullBase<LazyAugmentedConvexHull>;

// Function template instantiations.
#define INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(lhs, rhs)     \
  template double DistanceBetween(const ConvexHullBase<lhs> &,             \
                                  const ConvexHullBase<rhs> &, double,     \
                                  double);                                 \
  template bool AreOverlapping(const ConvexHullBase<lhs> &,                \
                               const ConvexHullBase<rhs> &);               \
  template bool ConvexHullBase<lhs>::IsApprox(const ConvexHullBase<rhs> &, \
                                              double) const;               \
  template bool ConvexHullBase<lhs>::Contains(const ConvexHullBase<rhs> &) \
      const;                                                               \
  template bool ConvexHullBase<lhs>::ContainsWithCircle(                   \
      const ConvexHullBase<rhs> &, double) const;

INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(EagerConvexHull,
                                                     EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(EagerConvexHull,
                                                     LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(EagerConvexHull,
                                                     LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    EagerConvexHull, LazyTransformedAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyAugmentedConvexHull,
                                                     EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyAugmentedConvexHull,
                                                     LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyAugmentedConvexHull,
                                                     LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyAugmentedConvexHull, LazyTransformedAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyTransformedConvexHull,
                                                     EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyTransformedConvexHull,
                                                     LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyTransformedConvexHull,
                                                     LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedConvexHull, LazyTransformedAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, LazyTransformedAugmentedConvexHull);

#undef INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES

}  // namespace mobility::collision
