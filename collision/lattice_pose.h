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

#ifndef MOBILITY_COLLISION_COLLISION_LATTICE_POSE_H_
#define MOBILITY_COLLISION_COLLISION_LATTICE_POSE_H_

#include <array>
#include <cmath>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/hash/hash.h"
#include "collision/grid_common.h"

namespace mobility::collision {

// This describes a vertex of a lattice grid.
struct LatticePose {
  GridIndex position;
  int angle;

  LatticePose() = default;
  LatticePose(const GridIndex& position_, int angle_)
      : position(position_), angle(angle_) {}

  bool operator==(const LatticePose& rhs) const {
    return (position == rhs.position) && (angle == rhs.angle);
  }
  bool operator!=(const LatticePose& rhs) const {
    return (position != rhs.position) || (angle != rhs.angle);
  }
};
// This is a hash function for looking up lattice cells.
struct LatticePoseHash {
  size_t operator()(const LatticePose& u) const {
    return absl::HashOf(0, u.position.x(), u.position.y(), u.angle);
  }
};

using LatticePoseSet = ::absl::flat_hash_set<LatticePose, LatticePoseHash>;

template <typename T>
using LatticePoseMap = ::absl::flat_hash_map<LatticePose, T, LatticePoseHash>;

template <typename T>
using LatticePoseNodeMap =
    ::absl::node_hash_map<LatticePose, T, LatticePoseHash>;

// This struct is used to specify the destination of a lattice connection,
// in terms of a relative (x,y) move and an absolute angle.
struct LatticeMove {
  GridIndex relative_xy_index;
  int angle_index;

  LatticeMove() = default;
  LatticeMove(const GridIndex& relative_xy_index_, int angle_index_)
      : relative_xy_index(relative_xy_index_), angle_index(angle_index_) {}

  bool operator==(const LatticeMove& rhs) const {
    return (relative_xy_index == rhs.relative_xy_index) &&
           (angle_index == rhs.angle_index);
  }
  bool operator!=(const LatticeMove& rhs) const {
    return (relative_xy_index != rhs.relative_xy_index) ||
           (angle_index != rhs.angle_index);
  }
};
// This is a hash function for looking up connections.
struct LatticeMoveHash {
  size_t operator()(const LatticeMove& spec) const {
    return absl::HashOf(0, spec.relative_xy_index.x(),
                        spec.relative_xy_index.y(), spec.angle_index);
  }
};

using LatticeMoveSet = ::absl::flat_hash_set<LatticeMove, LatticeMoveHash>;

template <typename T>
using LatticeMoveMap = ::absl::flat_hash_map<LatticeMove, T, LatticeMoveHash>;

template <typename T>
using LatticeMoveNodeMap =
    ::absl::node_hash_map<LatticeMove, T, LatticeMoveHash>;

// Specifies the reference frame and resolution of a grid so that conversions
// between continuous world coordinate points and the discrete grid cell on
// which those points fall can be done.
// This class also stores a string to identify the frame of reference.
struct LatticeFrame : GridFrame {
  static constexpr int MakeEvenDivisions(int i) {
    return (i > 4 ? 2 * ((i + 1) / 2) : 4);
  }

  LatticeFrame() = default;
  LatticeFrame(const GridFrame& grid_frame_, int num_angle_divisions_)
      : GridFrame(grid_frame_),
        num_angle_divisions(MakeEvenDivisions(num_angle_divisions_)) {}
  LatticeFrame(GridFrame&& grid_frame_, int num_angle_divisions_)
      : GridFrame(std::move(grid_frame_)),
        num_angle_divisions(MakeEvenDivisions(num_angle_divisions_)) {}
  LatticeFrame(std::string_view frame_id_, const eigenmath::Pose2d& origin_,
               double resolution_, int num_angle_divisions_)
      : GridFrame(frame_id_, origin_, resolution_),
        num_angle_divisions(MakeEvenDivisions(num_angle_divisions_)) {}

  void SetGridFrame(const GridFrame& grid_frame) {
    static_cast<GridFrame*>(this)->operator=(grid_frame);
  }

  // Transform world orientation (real-valued) into lattice angle index.
  int FrameSO2ToLatticeAngleIndex(const eigenmath::SO2d& rotation) const {
    double relative_angle = (origin.so2().inverse() * rotation).angle();
    if (relative_angle < 0.0) {
      // Bring in [0 .. 2*PI) range.
      relative_angle += 2.0 * M_PI;
    }
    return std::lrint(relative_angle * num_angle_divisions / (2.0 * M_PI)) %
           num_angle_divisions;
  }

  // Transform world orientation (real-valued) into lattice angle indices that
  // represent the two closest discrete angles.
  std::array<int, 2> FrameSO2ToLatticeAngleIndices(
      const eigenmath::SO2d& rotation) const {
    double relative_angle = (origin.so2().inverse() * rotation).angle();
    if (relative_angle < 0.0) {
      // Bring in [0 .. 2*PI) range.
      relative_angle += 2.0 * M_PI;
    }
    std::array<int, 2> result;
    result[0] = std::lrint(std::floor(relative_angle * num_angle_divisions /
                                      (2.0 * M_PI))) %
                num_angle_divisions;
    result[1] = (result[0] + 1) % num_angle_divisions;
    return result;
  }

  // Transform lattice angle index into world orientation (real-valued).
  eigenmath::SO2d LatticeAngleIndexToFrameSO2(int angle_id) const {
    return eigenmath::SO2d(angle_id * 2.0 * M_PI / num_angle_divisions);
  }

  // Transform world pose (real-valued 2d pose) into lattice cell pose.
  LatticePose FrameToLattice(const eigenmath::Pose2d& world_pose) const {
    // Compute the lattice and angle indices:
    LatticePose result;
    result.position = FrameToGrid(world_pose.translation());
    result.angle = FrameSO2ToLatticeAngleIndex(world_pose.so2());
    return result;  // NRVO
  }

  // Transform world pose (real-valued 2d pose) into lattice cell poses that
  // contain all the cells within grid resolution distance of the world point
  // and at the two discrete angles closest to the world orientation.
  std::array<LatticePose, 8> ClosestLatticePoses(
      const eigenmath::Pose2d& world_pose) const {
    // Compute the lattice and angle indices:
    GridRange grid_range = FrameToGridRange(world_pose.translation());
    std::array<int, 2> angle_indices =
        FrameSO2ToLatticeAngleIndices(world_pose.so2());
    std::array<LatticePose, 8> results;
    auto next_result = results.begin();
    grid_range.ForEachGridCoord([&](const GridIndex& index) {
      for (int angle : angle_indices) {
        next_result->position = index;
        next_result->angle = angle;
        ++next_result;
      }
    });
    return results;  // NRVO
  }

  // Transform lattice cell pose into world pose (real-valued 2d pose).
  eigenmath::Pose2d LatticeToFrame(const LatticePose& lattice_pose) const {
    return origin * eigenmath::Pose2d(
                        eigenmath::Vector2d(
                            lattice_pose.position.cast<double>() * resolution),
                        LatticeAngleIndexToFrameSO2(lattice_pose.angle));
  }

  // Get the angular resolution, in radians.
  double AngularResolution() const { return 2.0 * M_PI / num_angle_divisions; }

  // Get the angle index that is directly opposed to the given angle index.
  int GetOpposingAngle(int lattice_angle) const {
    return (num_angle_divisions / 2 + lattice_angle) % num_angle_divisions;
  }

  int num_angle_divisions;
};

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_LATTICE_POSE_H_
