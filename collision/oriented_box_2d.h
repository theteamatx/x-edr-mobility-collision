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

#ifndef MOBILITY_COLLISION_COLLISION_ORIENTED_BOX_2D_H_
#define MOBILITY_COLLISION_COLLISION_ORIENTED_BOX_2D_H_

#include <vector>

#include "eigenmath/pose2.h"
#include "eigenmath/types.h"

namespace mobility::collision {

// This class represents a simple oriented box in 2D. The box is parametrized
// as a central pose and two dimensions (width, height).
class OrientedBox2d {
 public:
  // Default box at origin with a zero size.
  OrientedBox2d() : center_pose_(), size_(0.0, 0.0) {}

  // Construct box from a given center pose and size vector.
  // Size values must be non-negative.
  OrientedBox2d(const eigenmath::Pose2d& center_pose,
                const eigenmath::Vector2d& size)
      : center_pose_(center_pose), size_(size) {
    CHECK_GE(size_.x(), 0.0);
    CHECK_GE(size_.y(), 0.0);
  }

  // Returns the area of the box.
  double Area() const { return size_.x() * size_.y(); }

  // Creates a vector of points representing the four corners of the box.
  // Points are ordered in counter-clockwise order.
  std::vector<eigenmath::Vector2d> GetPoints() const {
    return {
        center_pose_ * eigenmath::Vector2d{-0.5 * size_.x(), -0.5 * size_.y()},
        center_pose_ * eigenmath::Vector2d{0.5 * size_.x(), -0.5 * size_.y()},
        center_pose_ * eigenmath::Vector2d{0.5 * size_.x(), 0.5 * size_.y()},
        center_pose_ * eigenmath::Vector2d{-0.5 * size_.x(), 0.5 * size_.y()}};
  }

  // Returns the central pose of the box.
  const eigenmath::Pose2d& CenterPose() const { return center_pose_; }
  // Returns the size of the box in the local x-axis.
  const double& Width() const { return size_.x(); }
  // Returns the size of the box in the local y-axis.
  const double& Height() const { return size_.y(); }
  // Returns the centroid of the box.
  const eigenmath::Vector2d& Centroid() const {
    return center_pose_.translation();
  }
  // Returns the radius of the box.
  double Radius() const { return size_.norm() * 0.5; }

  // Checks if a point is contained within the box.
  bool Contains(const eigenmath::Vector2d& point) const {
    const eigenmath::Vector2d local_pt = center_pose_.inverse() * point;
    return (local_pt.cwiseAbs() - size_ * 0.5).maxCoeff() <= 0.0;
  }

 private:
  eigenmath::Pose2d center_pose_;
  eigenmath::Vector2d size_;
};

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_ORIENTED_BOX_2D_H_
