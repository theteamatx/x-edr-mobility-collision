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

#ifndef MOBILITY_COLLISION_COLLISION_CONVEX_HULL_H_
#define MOBILITY_COLLISION_COLLISION_CONVEX_HULL_H_

#include <array>
#include <limits>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/str_format.h"
#include "collision/oriented_box_2d.h"
#include "diff_drive/dynamic_limits.h"
#include "diff_drive/state.h"

namespace mobility::collision {

// forward declaration
template <typename Derived>
class ConvexHullBase;
class EagerConvexHull;
template <typename Original>
class LazyTransformedConvexHullBase;
template <typename Original>
class LazyAugmentedConvexHullBase;
class Hull;

using ConvexHull = EagerConvexHull;
using LazyAugmentedConvexHull = LazyAugmentedConvexHullBase<EagerConvexHull>;
using LazyTransformedConvexHull =
    LazyTransformedConvexHullBase<EagerConvexHull>;
using LazyTransformedAugmentedConvexHull =
    LazyTransformedConvexHullBase<LazyAugmentedConvexHullBase<EagerConvexHull>>;

// Determines the minimum distance from this convex hull to another.
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
    const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
    double min_distance = std::numeric_limits<double>::lowest(),
    double max_distance = std::numeric_limits<double>::max());

// Determines if two hulls overlap. This function uses the GJK method to make
// a simple overlap test.
// For distance computations, see DistanceBetween function (~4x slower,
// run benchmarks in convex_hull_test.cc for accurate number).
template <typename Lhs, typename Rhs>
bool AreOverlapping(const ConvexHullBase<Lhs> &lhs,
                    const ConvexHullBase<Rhs> &rhs);

// CRTP base class which defines convex hull algorithms in terms of its ccw
// extreme points, their edge vectors, and optimized checks using a bounding
// circle around the centroid.  The data is accessed using cache evaluating
// wrappers to allow for on-demand calculation if a shape is modified using a
// transformation, for example.
template <typename Derived>
class ConvexHullBase {
 public:
  explicit operator const EagerConvexHull &() const;

  bool Contains(const eigenmath::Vector2d &point) const;

  bool Contains(double px, double py) const {
    return Contains(eigenmath::Vector2d{px, py});
  }

  template <typename Other>
  bool Contains(const ConvexHullBase<Other> &other_hull) const;

  // Returns true if the two hulls are almost the same, with a tolerance.
  // NOTE: The check requires that the two hulls have the same number of points
  // which means that hulls could be approximately the same, but with different
  // points, and that would fail this similarity test.
  template <typename Other>
  bool IsApprox(const ConvexHullBase<Other> &rhs,
                double tolerance = 1e-6) const;

  // Returns a signed distance from the hull to the point. When the point is
  // inside the hull, the distance is negative.
  double Distance(const eigenmath::Vector2d &point) const;

  double Distance(double px, double py) const {
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

  // Find the first counterclockwise point of this convex hull that reaches the
  // farthest along a given direction.
  // In essence, this function projects every point of the convex hull onto the
  // given axis (dot-product) and keeps track of the max.
  eigenmath::Vector2d GetFarthestPointInDirection(
      const eigenmath::Vector2d &direction) const;

  // Computes the maximum distance of any point in the hull from
  // the given point. In other words, if one were to sweep-rotate the hull
  // around the given point, then the maximum radius (outer circle) obtained
  // is what this function computes.
  double GetMaxRadiusAround(const eigenmath::Vector2d &point) const;

  // Same as above, but avoids taking the squared root.
  double GetMaxSquaredRadiusAround(const eigenmath::Vector2d &point) const;

  // Find the bounding box with the minimum area.
  // Returns the center pose (incl. orientation) of the box and the dimensions
  // of the box along x and y.
  OrientedBox2d GetMinAreaBoundingBox() const;

  // Provides access for tests.  Requires derived class to implement caching
  // functionality.
  const std::vector<eigenmath::Vector2d> &GetPoints() const {
    CachePointsAndDeltas();
    return GetCache().points_;
  }
  eigenmath::Vector2d GetCentroid() const {
    CacheBoundingCircle();
    return GetCache().centroid_;
  }
  double GetRadius() const {
    CacheBoundingCircle();
    return GetCache().radius_;
  }

  void CacheBoundingCircle() const {
    static_cast<const Derived *>(this)->CacheBoundingCircle();
  }
  void CachePointsAndDeltas() const {
    static_cast<const Derived *>(this)->CachePointsAndDeltas();
  }
  EagerConvexHull &GetCache() {
    return static_cast<Derived *>(this)->GetCache();
  }
  const EagerConvexHull &GetCache() const {
    return static_cast<const Derived *>(this)->GetCache();
  }

  // Allows to use helper functions when performing pairwise operations with
  // convex hull objects of different derived types.
  friend class Hull;
  template <typename Original>
  friend class LazyAugmentedConvexHullBase;
  template <typename Original>
  friend class LazyTransformedConvexHullBase;

 protected:
  ConvexHullBase() = default;
  ~ConvexHullBase() = default;

  // Gets an upper bound to the number of points in the convex hull.
  int UpperBoundNumberOfPoints() const {
    return static_cast<const Derived *>(this)->UpperBoundNumberOfPoints();
  }

 private:
  bool ContainsWithCircle(const eigenmath::Vector2d &point,
                          double current_distance = 0.0) const;
  template <typename Other>
  bool ContainsWithCircle(const ConvexHullBase<Other> &other_hull,
                          double current_distance = 0.0) const;
  bool ContainsWithPoints(const eigenmath::Vector2d &point) const;

  double DistanceIfLessThanWithPoints(
      const eigenmath::Vector2d &point, double current_distance = 0.0,
      eigenmath::Vector2d *closest_point = nullptr) const;

  // Find a Minkowski difference support vector, for GJK implementation.
  template <typename Lhs, typename Rhs>
  static eigenmath::Vector2d GetMinkowskiDiffSupport(
      const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
      eigenmath::Vector2d axis);

  // Basic GJK algorithm that stops when overlap / no-overlap is detected and
  // produces a simplex of three points that it stopped with.
  template <typename Lhs, typename Rhs>
  static bool FindGJKSimplex(const ConvexHullBase<Lhs> &lhs,
                             const ConvexHullBase<Rhs> &rhs,
                             std::array<eigenmath::Vector2d, 3> *simplex);

  // Enhanced GJK algorithm that keeps moving the GJK simplex until it can no
  // longer get closer to the origin, and returns the separation distance.
  template <typename Lhs, typename Rhs>
  static double FindSeparationDistance(
      const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
      std::array<eigenmath::Vector2d, 3> simplex, double min_distance,
      double max_distance);

  // Expanding Polytope algorithm (EPA) that expands a given simplex until it
  // cannot push the closest edge further away from the origin, and returns the
  // penetration depth.
  template <typename Lhs, typename Rhs>
  static double FindPenetrationDepth(
      const ConvexHullBase<Lhs> &lhs, const ConvexHullBase<Rhs> &rhs,
      const std::array<eigenmath::Vector2d, 3> &simplex, double min_distance);
};

// Print to ostream for testing and debugging.
template <typename Sink, typename Derived>
void AbslStringify(Sink &sink, const ConvexHullBase<Derived> &hull) {
  for (const eigenmath::Vector2d &point : hull.GetPoints()) {
    absl::Format(&sink, "%v, ", eigenmath::AbslStringified(point));
  }
}

template <typename Derived>
inline std::ostream &operator<<(std::ostream &stream,
                                const ConvexHullBase<Derived> &hull) {
  return stream << absl::StrCat(hull);
}

// A class serving as a storage type for convex hulls.  All modifications are
// eagerly evaluated.
class EagerConvexHull : public ConvexHullBase<EagerConvexHull> {
 public:
  // May fail: we need at least 3 points to create a convex hull.
  explicit EagerConvexHull(std::vector<eigenmath::Vector2d> pointcloud);

  explicit EagerConvexHull(const OrientedBox2d &box) : EagerConvexHull() {
    FromCounterClockwiseExtremePoints(box.GetPoints());
  }

  // Creates a convex hull by growing an existing one by a given
  // amount of padding. It simply offsets each line by the padding
  // amount toward the outside of the original hull. It thus is a
  // conservative estimate of the set of points that are within
  // padding distance of the original hull (the corners would need to
  // be converted into sections of disks to model this
  // properly).
  EagerConvexHull CreateBiggerHull(double padding) const;

  // Creates a convex hull by growing an existing one by a given
  // amount of directional padding. The padding function should take a normal
  // vector and its magnitude, and output a displacement vector in the same
  // direction as this normal vector but with the magnitude of the desired
  // padding in that direction
  EagerConvexHull CreateBiggerHull(
      absl::FunctionRef<eigenmath::Vector2d(eigenmath::Vector2d, double)>
          directional_padding) const;

  // Transforms the points of a given original hull with the given pose
  // and copies the result into this hull object.
  // Returns false if there is a mismatch in the number of points in
  // the original hull versus the number of points that this hull can
  // contain, so as to not reallocate memory.
  // This function is RT-safe, will not reallocate any memory.
  bool TransformAndCopy(const eigenmath::Pose2d &rel_pose,
                        const EagerConvexHull &orig_hull);

  // Transforms the convex hull with the given pose.
  bool ApplyTransform(const eigenmath::Pose2d &rel_pose) {
    return TransformAndCopy(rel_pose, *this);
  }

  template <typename Lhs, typename Rhs>
  friend bool AreOverlapping(const ConvexHullBase<Lhs> &lhs,
                             const ConvexHullBase<Rhs> &rhs);

  template <typename Lhs, typename Rhs>
  friend double DistanceBetween(const ConvexHullBase<Lhs> &lhs,
                                const ConvexHullBase<Rhs> &rhs,
                                double min_distance, double max_distance);

  int UpperBoundNumberOfPoints() const { return points_.size(); }

  // Allows lazy wrappers to call default constructor.
  template <typename Original>
  friend class LazyAugmentedConvexHullBase;
  template <typename Original>
  friend class LazyTransformedConvexHullBase;
  template <typename Derived>
  friend class ConvexHullBase;
  friend class Hull;

  // Test-only section.
  enum TestOnlyFlag {
    kTestOnlyClockwisePoints = 0,
  };
  // Test-only constructor without clockwise-reordering points, this is
  // to create a specific ordering of points for a test.
  EagerConvexHull(std::vector<eigenmath::Vector2d> pointcloud, TestOnlyFlag) {
    FromCounterClockwiseExtremePoints(std::move(pointcloud));
  }

 private:
  EagerConvexHull() = default;

  // Same as public versions, but modifying the object.
  void CreateBiggerHull(const std::vector<eigenmath::Vector2d> &orig_hull,
                        double padding);

  template <typename Functor>
  void CreateBiggerHullImpl(const std::vector<eigenmath::Vector2d> &orig_hull,
                            Functor directional_padding);

  // Constructs a convex hull object from a counter clockwise sequence of
  // extreme points.
  void FromCounterClockwiseExtremePoints(
      std::vector<eigenmath::Vector2d> ccw_convex_hull);

  void CacheBoundingCircle() const {}
  void CachePointsAndDeltas() const {}
  EagerConvexHull &GetCache() { return *this; }
  const EagerConvexHull &GetCache() const { return *this; }

  int NumberOfPoints() { return points_.size(); }

  void ComputeCircle();

  // Possibly invalid data members.  Call Cache... before using.
  std::vector<eigenmath::Vector2d> points_, deltas_;
  eigenmath::Vector2d centroid_;
  double radius_;
};

// This class creates an asymmetric padding box for use in the CreateBiggerHull
// function of ConvexHull to enlarge the convex hull differently in different
// directions (aligned with x-y axes).
//
// The padding values are separate for each axis and each direction in those
// axes, between positive and negative.
//
// Typical use-case:
// If the convex hull is expressed in the robot frame, then the padding in the
// positive x-axis might be the forward collision distance, and the padding in
// other directions would be related to side collisions.
struct AlignedPaddingBox {
  double padding_x_pos = 0.0;
  double padding_x_neg = 0.0;
  double padding_y_pos = 0.0;
  double padding_y_neg = 0.0;

  AlignedPaddingBox(double padding_x_pos_, double padding_x_neg_,
                    double padding_y_pos_, double padding_y_neg_)
      : padding_x_pos(padding_x_pos_),
        padding_x_neg(padding_x_neg_),
        padding_y_pos(padding_y_pos_),
        padding_y_neg(padding_y_neg_) {}

  // Allows this padding object to be used in CreateBiggerHull.
  // This function returns the scaled normal vector that gets to the boundary
  // of the padding region from a given normal vector.
  eigenmath::Vector2d operator()(eigenmath::Vector2d u, double u_norm) const;
};

// This class wraps a reference to a given untransformed convex hull and
// uses it to generate a transformed convex hull in a way that is most
// economical w.r.t. answering collision-checking queries.
// For instance, this avoids transforming all the points of the convex hull if
// there is never a need to use them during collision-checking because no
// obstacle point falls within the bounding circle.
//
// The representation of the original convex hull can be a lazy shape as well,
// and evaluations are minimized where possible.
template <typename Original>
class LazyTransformedConvexHullBase
    : public ConvexHullBase<LazyTransformedConvexHullBase<Original>> {
 public:
  // Construct a lazy transformed convex hull that refers to a specific
  // original (untransformed) convex hull.
  // Note: this object will always refer to this particular convex hull as
  // containing the original points to be transformed. It is the responsibility
  // of the user to guarantee that orig_hull remains valid.
  explicit LazyTransformedConvexHullBase(
      const ConvexHullBase<Original> *orig_hull);

  // Reset a lazy transformed convex hull that refers to a specific
  // original (untransformed) convex hull.
  // Note: this object will refer to this particular convex hull as
  // containing the original points to be transformed. It is the responsibility
  // of the user to guarantee that orig_hull remains valid.
  void ResetHullBase(const ConvexHullBase<Original> *orig_hull);

  // Explicitly document that it is OK to copy / move objects of this class.
  // However, all copies will always refer back to the same original convex
  // hull, but will have their own local copy for the transformed hull.
  LazyTransformedConvexHullBase(const LazyTransformedConvexHullBase &) =
      default;
  LazyTransformedConvexHullBase(LazyTransformedConvexHullBase &&) = default;
  LazyTransformedConvexHullBase &operator=(
      const LazyTransformedConvexHullBase &) = default;
  LazyTransformedConvexHullBase &operator=(LazyTransformedConvexHullBase &&) =
      default;

  int UpperBoundNumberOfPoints() const {
    return orig_hull_->UpperBoundNumberOfPoints();
  }

  // Resets the transform to be applied to the original hull.
  // This function is RT-safe, will not reallocate any memory.
  bool ApplyTransform(const eigenmath::Pose2d &transformed_pose_orig);

  // For debugging / testing purposes:
  bool HasTransformedCentroid() const { return has_transformed_centroid_; }
  bool HasTransformedPoints() const { return has_transformed_points_; }

 private:
  void CachePointsAndDeltas() const;
  void CacheBoundingCircle() const;

  EagerConvexHull &GetCache() { return transformed_hull_; }
  const EagerConvexHull &GetCache() const { return transformed_hull_; }

  friend class ConvexHullBase<LazyTransformedConvexHullBase<Original>>;

  const ConvexHullBase<Original> *orig_hull_;
  eigenmath::Pose2d transform_;
  mutable EagerConvexHull transformed_hull_;
  mutable bool has_transformed_centroid_ = false;
  mutable bool has_transformed_points_ = false;

  void TransformCentroid() const;
  void TransformPoints() const;
};

// This class wraps a reference to a fixed convex hull and uses it to create an
// augmented convex hull for use in collision checking queries.  The augmented
// hull is calculated for the last provided state on demand, and the result is
// cached for repeated queries with the same state information.  If no state
// information is provided, no motion is assumed.
template <typename Original>
class LazyAugmentedConvexHullBase
    : public ConvexHullBase<LazyAugmentedConvexHullBase<Original>> {
 public:
  // Constructs a lazy augmented convex hull that refers to a specific original
  // convex hull, and to fixed limits.
  // Note: this object will always refer to this particular convex hull as
  // containing the original points to be transformed, and to these particular
  // limits. It is the responsibility of the user to guarantee that orig_hull
  // and limits remain valid.
  // Note: The underlying convex hull is not checked for changes.  Recomputation
  // of the augmentation only happens after ApplyMotion() is called.
  LazyAugmentedConvexHullBase(const diff_drive::DynamicLimits *limits,
                              const ConvexHullBase<Original> *orig_hull);

  // Reset a lazy augmented convex hull that refers to a specific
  // original convex hull, and to fixed limits.
  // Note: this object will refer to this particular convex hull as
  // containing the original points to be transformed. It is the responsibility
  // of the user to guarantee that orig_hull remains valid.
  void ResetHullBase(const diff_drive::DynamicLimits *limits,
                     const ConvexHullBase<Original> *orig_hull);

  LazyAugmentedConvexHullBase(const LazyAugmentedConvexHullBase &) = default;
  LazyAugmentedConvexHullBase(LazyAugmentedConvexHullBase &&) = default;
  LazyAugmentedConvexHullBase &operator=(const LazyAugmentedConvexHullBase &) =
      default;
  LazyAugmentedConvexHullBase &operator=(LazyAugmentedConvexHullBase &&) =
      default;

  int UpperBoundNumberOfPoints() const {
    // The augmentation can double the number of points when adding the stopping
    // points, and triple the number of points when adding the danger margin
    // padding.  So in total there can be up to six times the number of points
    // as contained in the original hull.
    return 6 * orig_hull_->UpperBoundNumberOfPoints();
  }

  // Applies a delayed augmentation on the convex hull, based on the motion
  // description in the state.  If no motion is provided, applies only the
  // danger margin to the original hull.
  //
  // Note: Multiple calls do not result in accumulated motion.  The latest call
  // defines the motion used for the augmentation.  If no call is made, the
  // original hull is forwarded.
  void ApplyMotion(const diff_drive::State &state);

  // Set a danger margin to apply to additionally apply to the augmented hull.
  // Note: The danger margin will be applied _after_ the augmentation, even when
  // this is called before ApplyMotion().
  void SetDangerMargin(double danger_margin);

  // For testing purposes
  bool HasAugmentedBoundingCircle() const {
    return has_augmented_bounding_circle_;
  }
  bool HasAugmentedPoints() const { return has_augmented_hull_; }

  // Get a reference to the configured dynamic limits.
  const diff_drive::DynamicLimits &GetLimits() const { return *limits_; }

 private:
  void CachePointsAndDeltas() const;
  void CacheBoundingCircle() const;

  EagerConvexHull &GetCache() { return augmented_hull_; }
  const EagerConvexHull &GetCache() const { return augmented_hull_; }

  // Enables CRTP using private member functions.
  friend class ConvexHullBase<LazyAugmentedConvexHullBase<Original>>;

  const ConvexHullBase<Original> *orig_hull_;
  const diff_drive::DynamicLimits *limits_;
  diff_drive::State state_;
  double danger_margin_;
  mutable EagerConvexHull augmented_hull_;
  mutable std::vector<eigenmath::Vector2d> point_buffer_;
  mutable bool has_augmented_bounding_circle_ = false;
  mutable bool has_augmented_hull_ = false;

  void AugmentBoundingCircle() const;
  void AugmentConvexHull() const;
};

// Explicitly instantiate used types.
extern template class ConvexHullBase<EagerConvexHull>;
extern template class ConvexHullBase<LazyAugmentedConvexHull>;
extern template class ConvexHullBase<LazyTransformedConvexHull>;
extern template class ConvexHullBase<LazyTransformedAugmentedConvexHull>;

extern template class LazyAugmentedConvexHullBase<EagerConvexHull>;
extern template class LazyTransformedConvexHullBase<EagerConvexHull>;
extern template class LazyTransformedConvexHullBase<LazyAugmentedConvexHull>;

// Function template instantiations.
#define INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(lhs, rhs)        \
  extern template double DistanceBetween(const ConvexHullBase<lhs> &,         \
                                         const ConvexHullBase<rhs> &, double, \
                                         double);                             \
  extern template bool AreOverlapping(const ConvexHullBase<lhs> &,            \
                                      const ConvexHullBase<rhs> &);           \
  extern template bool ConvexHullBase<lhs>::IsApprox(                         \
      const ConvexHullBase<rhs> &, double) const;                             \
  extern template bool ConvexHullBase<lhs>::Contains(                         \
      const ConvexHullBase<rhs> &) const;                                     \
  extern template bool ConvexHullBase<lhs>::ContainsWithCircle(               \
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

#endif  // MOBILITY_COLLISION_COLLISION_CONVEX_HULL_H_
