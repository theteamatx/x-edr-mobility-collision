/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MOBILITY_COLLISION_COLLISION_HULL_H_
#define MOBILITY_COLLISION_COLLISION_HULL_H_

#include <limits>
#include <utility>
#include <vector>

#include "collision/convex_hull.h"
#include "diff_drive/dynamic_limits.h"
#include "diff_drive/state.h"

namespace mobility::collision {

// forward declaration
template <typename ConvexHull>
class HullBase;
class Hull;
class LazyTransformedHull;
class LazyAugmentedHull;

// Determines the minimum distance from this hull to another.
// In case of separation (non-overlapping), the distance returned will be
// positive and reflects the shortest distance between the hulls.
// In case of penetration (overlapping), the distance returned will be
// negative and its magnitude reflects the depth of penetration.
// The min_distance and max_distance parameters can be used to limit the
// algorithm to a distance range. For example, with a min_distance of zero,
// only the separation case will be accurately computed. And conversely,
// with a max distance of zero, only the penetration case will be accurately
// computed. In other words, the algorithm will stop as soon as it can be
// sure that the actual distance is outside of the given bounds, and the
// distance returned is the lower or upper bound for the actual distance.
// For a simple overlap test, see AreOverlapping function (~4x faster, run
// benchmarks in convex_hull_test.cc for accurate number).
template <typename Lhs, typename Rhs>
double DistanceBetween(
    const HullBase<Lhs> &lhs, const HullBase<Rhs> &rhs,
    double min_distance = std::numeric_limits<double>::lowest(),
    double max_distance = std::numeric_limits<double>::max());

// Determines if this hull overlaps with the given one. This function uses
// the GJK method to make a simple overlap test.
// For distance computations, see DistanceBetween function (~4x slower,
// run benchmarks in convex_hull_test.cc for accurate number).
template <typename Lhs, typename Rhs>
bool AreOverlapping(const HullBase<Lhs> &lhs, const HullBase<Rhs> &rhs);

// Determines if two hulls overlap. This function is a convenient wrapper
// for mixing hulls and convex-hulls.
template <typename Lhs, typename Rhs>
bool AreOverlapping(const HullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs) {
  for (const auto &lhs_chull : lhs.GetConvexHulls()) {
    if (AreOverlapping(lhs_chull, rhs)) {
      return true;
    }
  }
  return false;
}
template <typename Lhs, typename Rhs>
bool AreOverlapping(const ConvexHullBase<Lhs> &lhs, const HullBase<Rhs> &rhs) {
  return AreOverlapping(rhs, lhs);
}

// Combine all points of all convex hulls of a given hull to form a single
// convex-hull to cover it all. Note that if the given hull is very non-convex,
// with gaps and concave areas, the resulting convex-hull will be very
// different from the original (encompassing all gaps).
template <typename CHull>
ConvexHull FlattenToConvexHull(const HullBase<CHull> &hull) {
  std::vector<eigenmath::Vector2d> all_points;
  for (const auto &chull : hull.GetConvexHulls()) {
    const auto &chull_pts = chull.GetPoints();
    all_points.insert(all_points.end(), chull_pts.begin(), chull_pts.end());
  }
  return ConvexHull(all_points);
}

// Straightforward implementation as a set of convex hulls, allowing different
// specialized convex hull types to be used.  The algorithms allow to combine
// various convex hull types.  Do not use this base class directly, but prefer
// to use the derived variants.
template <typename ConvexHullType>
class HullBase {
 public:
  void Clear() { convex_hulls_.clear(); }

  bool Contains(const eigenmath::Vector2d &point) const;

  bool Contains(double px, double py) const {
    return Contains(eigenmath::Vector2d{px, py});
  }

  // Returns true if the two hulls are almost the same, with a tolerance.
  // NOTE: The check requires that the two hulls have the same number of points
  // which means that hulls could be approximately the same, but with different
  // points, and that would fail this similarity test.
  template <typename OtherConvexHullType>
  bool IsApprox(const HullBase<OtherConvexHullType> &rhs,
                double tolerance = 1e-6) const;

  // Returns a signed distance from the hull to the point. When the point is
  // inside the hull, the distance is negative.
  double Distance(const eigenmath::Vector2d &point) const;

  bool Distance(double px, double py) const {
    return Distance(eigenmath::Vector2d{px, py});
  }

  // Returns a signed distance from the hull to the point. When the point is
  // inside the hull, the distance is negative.
  // This version returns 1.0 if not penetrating.
  double DistanceIfPenetrating(const eigenmath::Vector2d &point) const;

  double DistanceIfPenetrating(double px, double py) const {
    return DistanceIfPenetrating(eigenmath::Vector2d{px, py});
  }

  // Returns a signed distance from the hull to the point. When the point is
  // inside the hull, the distance is negative.
  // This version returns 1.0 if the penetration distance is above the given
  // current_distance value.
  double DistanceIfLessThan(const eigenmath::Vector2d &point,
                            double current_distance) const;

  double DistanceIfLessThan(double px, double py,
                            double current_distance) const {
    return DistanceIfLessThan(eigenmath::Vector2d{px, py}, current_distance);
  }

  // Returns a signed distance from the hull to the point and the closest point
  // on the hull to the given point. When the point is inside the hull,
  // the distance is negative.
  // This version returns 1.0 if not penetrating.
  double ClosestPointIfPenetrating(const eigenmath::Vector2d &point,
                                   eigenmath::Vector2d *closest_point) const;

  double ClosestPointIfPenetrating(double px, double py,
                                   eigenmath::Vector2d *closest_point) const {
    return ClosestPointIfPenetrating(eigenmath::Vector2d{px, py},
                                     closest_point);
  }

  bool Empty() const { return convex_hulls_.empty(); }

  // Computes the maximum distance of any point in the hull from
  // the given point. In other words, if one were to sweep-rotate the hull
  // around the given point, then the maximum radius (outer circle) obtained
  // is what this function computes.
  double GetMaxRadiusAround(const eigenmath::Vector2d &point) const;

  // Same as above, but avoids taking the squared root.
  double GetMaxSquaredRadiusAround(const eigenmath::Vector2d &point) const;

  const std::vector<ConvexHullType> &GetConvexHulls() const {
    return convex_hulls_;
  }

  template <typename Other>
  friend class HullBase;

 protected:
  HullBase() = default;

  std::vector<ConvexHullType> convex_hulls_;
};

class Hull : public HullBase<EagerConvexHull> {
 public:
  Hull() = default;

  explicit Hull(const std::vector<eigenmath::Vector2d> &points) { Add(points); }

  // Constructor to turn any type of (lazy) hull into a Hull (eager).
  template <typename OtherConvexHullType>
  explicit Hull(const HullBase<OtherConvexHullType> &rhs) {
    for (auto &rhs_chull : rhs.GetConvexHulls()) {
      Add(EagerConvexHull{rhs_chull});
    }
  }

  // Explicitly default the move-/copy- constructors because of above.
  Hull(const Hull &) = default;
  Hull(Hull &&) = default;
  Hull &operator=(const Hull &) = default;
  Hull &operator=(Hull &&) = default;

  void Add(ConvexHull convex_hull) {
    convex_hulls_.emplace_back(std::move(convex_hull));
  }

  void Add(const std::vector<eigenmath::Vector2d> &points);

  // Go through each hull and remove any convex hull that is entirely
  // contained in another.
  void RemoveInternalHulls();

  // Removes any convex hull that is almost contained in another.
  // Warning: This function can grow the overall area of the hull by creating
  // bigger convex hulls to encompass multiple original convex hulls that are
  // not entirely contained within each other. Use RemoveInternalHulls() for
  // a more conservative reduction that only prunes internal hulls.
  void SimplifyHulls();

  // Transforms the convex hulls of a given original hull with the given pose
  // and copies the result into this hull object.
  // Returns false if there is a mismatch in the number of convex hulls or
  // the numbers of points in them from the original hull versus what this
  // hull can contain, so as to not reallocate memory.
  // This function is RT-safe, will not reallocate any memory.
  bool TransformAndCopy(const eigenmath::Pose2d &rel_pose,
                        const Hull &orig_hull);

  // Transforms the convex hulls with the given pose.
  bool ApplyTransform(const eigenmath::Pose2d &rel_pose) {
    return TransformAndCopy(rel_pose, *this);
  }

  friend class LazyTransformedHull;
  friend class LazyAugmentedHull;
};

// This class wraps a reference to a given untransformed hull and
// uses it to generate a transformed hull in a way that is most
// economical w.r.t. answering collision-checking queries.
// For instance, this avoids transforming all the points of its convex hulls if
// there is never a need to use them during collision-checking because no
// obstacle point falls within the bounding circle.
class LazyTransformedHull : public HullBase<LazyTransformedConvexHull> {
 public:
  LazyTransformedHull() = default;

  // Constructs a lazy transformed hull that refers to a specific original
  // (untransformed) hull.
  // Note: this object will always refer to this particular hull as containing
  // the original points to be transformed. It is the responsibility of the
  // user to guarantee that orig_hull remains valid.
  explicit LazyTransformedHull(const Hull &orig_hull);

  // Resets a lazy transformed hull that refers to a specific original
  // (untransformed) hull.
  // Note: this object will refer to this particular hull as containing
  // the original points to be transformed. It is the responsibility of the
  // user to guarantee that orig_hull remains valid.
  void ResetHullBase(const Hull &orig_hull);

  // Explicitly document that it is OK to copy / move objects of this class.
  // However, all copies will always refer back to the same original convex
  // hull, but will have their own local copy for the transformed hull.
  LazyTransformedHull(const LazyTransformedHull &) = default;
  LazyTransformedHull(LazyTransformedHull &&) = default;
  LazyTransformedHull &operator=(const LazyTransformedHull &) = default;
  LazyTransformedHull &operator=(LazyTransformedHull &&) = default;

  // Resets the transform to be applied to the original hull.
  // This function is RT-safe, will not reallocate any memory.
  void ApplyTransform(const eigenmath::Pose2d &transformed_pose_orig);
};

// This class serves as a container to forward the augmentation queries
// 'ApplyMotion' and to hold the augmented hull used within LazyTransformedHull.
// It is not intended to be queried directly as a Hull, but mainly as a
// mechanism to improve collision checking queries.
// For instance, this avoid augmenting each convex hull which is part of this
// hull, if using bounding circles suffices to perform the queries.
class LazyAugmentedHull : public HullBase<LazyAugmentedConvexHull> {
 public:
  LazyAugmentedHull() = default;

  // Constructs a lazy augmented hull that refers to a specific original hull.
  // Note: this object will always refer to this particular hull as
  // containing the original points to be transformed. It is the responsibility
  // of the user to guarantee that orig_hull and limits remain valid.
  LazyAugmentedHull(const diff_drive::DynamicLimits *limits,
                    const Hull &orig_hull);

  // Resets a lazy augmented hull to refer to a new original hull.
  // Note: this object will refer to this particular hull as
  // containing the original points to be transformed. It is the responsibility
  // of the user to guarantee that orig_hull and limits remain valid.
  void ResetHullBase(const diff_drive::DynamicLimits *limits,
                     const Hull &orig_hull);

  // Explicitly document that it is OK to copy / move objects of this class.
  // However, all copies will always refer back to the same original convex
  // hull, but will have their own local copy for the augmented hull.
  LazyAugmentedHull(const LazyAugmentedHull &) = default;
  LazyAugmentedHull(LazyAugmentedHull &&) = default;
  LazyAugmentedHull &operator=(const LazyAugmentedHull &) = default;
  LazyAugmentedHull &operator=(LazyAugmentedHull &&) = default;

  // Resets the motion which defines the augmentation.  If no motion is
  // provided, forwards the original hull.
  void ApplyMotion(const diff_drive::State &state);

  // Apply an additional danger margin around the augmented hull.
  void SetDangerMargin(double danger_margin);

  friend class LazyTransformedAugmentedHull;
};

// This class wraps a reference to a given untransformed, augmented hull and
// uses it to generate a transformed hull in a way that is most
// economical w.r.t. answering collision-checking queries.
// For instance, this avoids transforming all the points of its convex hulls if
// there is never a need to use them during collision-checking because no
// obstacle point falls within the bounding circle.
class LazyTransformedAugmentedHull
    : public HullBase<LazyTransformedAugmentedConvexHull> {
 public:
  LazyTransformedAugmentedHull() = default;

  // Constructs a lazy transformed hull that refers to a specific original
  // (untransformed) augmented hull.
  // Note: this object will always refer to this particular hull as containing
  // the original points to be transformed. It is the responsibility of the
  // user to guarantee that orig_hull remains valid.
  explicit LazyTransformedAugmentedHull(const LazyAugmentedHull &orig_hull);

  // Resets a lazy transformed hull that refers to a specific original
  // (untransformed) augmented hull.
  // Note: this object will refer to this particular hull as containing
  // the original points to be transformed. It is the responsibility of the
  // user to guarantee that orig_hull remains valid.
  void ResetHullBase(const LazyAugmentedHull &orig_hull);

  // Explicitly document that it is OK to copy / move objects of this class.
  // However, all copies will always refer back to the same original convex
  // hull, but will have their own local copy for the transformed hull.
  LazyTransformedAugmentedHull(const LazyTransformedAugmentedHull &) = default;
  LazyTransformedAugmentedHull(LazyTransformedAugmentedHull &&) = default;
  LazyTransformedAugmentedHull &operator=(
      const LazyTransformedAugmentedHull &) = default;
  LazyTransformedAugmentedHull &operator=(LazyTransformedAugmentedHull &&) =
      default;

  // Resets the transform to be applied to the original hull.
  // This function is RT-safe, will not reallocate any memory.
  void ApplyTransform(const eigenmath::Pose2d &transformed_pose_orig);
};

extern template class HullBase<EagerConvexHull>;
extern template class HullBase<LazyTransformedConvexHull>;
extern template class HullBase<LazyAugmentedConvexHull>;
extern template class HullBase<LazyTransformedAugmentedConvexHull>;

// Function template instantiations.
#define INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(lhs, rhs)        \
  extern template double DistanceBetween(                                     \
      const HullBase<lhs> &, const HullBase<rhs> &, double, double);          \
  extern template bool AreOverlapping(const HullBase<lhs> &,                  \
                                      const HullBase<rhs> &);                 \
  extern template bool HullBase<lhs>::IsApprox(const HullBase<rhs> &, double) \
      const;

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

#endif  // MOBILITY_COLLISION_COLLISION_HULL_H_
