# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_proto_library")

licenses(["notice"])

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "base_collision_config_proto",
    srcs = [
        "base_collision_config.proto",
    ],
    deps = [
        ":hull_proto",
        "@eigenmath//eigenmath:eigenmath_proto",
    ],
)

cc_proto_library(
    name = "base_collision_config_cc_proto",
    deps = [":base_collision_config_proto"],
)

cc_library(
    name = "collision_checking",
    srcs = [
        "collision_hull_augmentation.cc",
        "convex_hull.cc",
        "danger_hull.cc",
        "hull.cc",
    ],
    hdrs = [
        "collision_hull_augmentation.h",
        "convex_hull.h",
        "danger_hull.h",
        "hull.h",
        "oriented_box_2d.h",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings:str_format",
        "@eigenmath//eigenmath",
        "@genit//genit:iterators",
        "@mobility_diff_drive//diff_drive:differential_drive",
    ],
)

cc_test(
    name = "collision_hull_augmentation_test",
    srcs = ["collision_hull_augmentation_test.cc"],
    deps = [
        ":collision_checking",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
        "@mobility_diff_drive//diff_drive:differential_drive",
        "@mobility_diff_drive//diff_drive:test_fixtures",
    ],
)

cc_library(
    name = "collision_utils",
    srcs = [
        "collision_utils.cc",
    ],
    hdrs = [
        "collision_utils.h",
    ],
    deps = [
        ":collision_checking",
        ":grid_common",
        ":occupancy_grid",
        "@com_google_absl//absl/algorithm:container",
        "@eigenmath//eigenmath",
        "@genit//genit:iterators",
        "@mobility_diff_drive//diff_drive:differential_drive",
    ],
)

cc_test(
    name = "collision_utils_test",
    srcs = [
        "collision_utils_test.cc",
    ],
    deps = [
        ":collision_checking",
        ":collision_utils",
        ":grid_test_utils",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
        "@mobility_diff_drive//diff_drive:test_fixtures",
    ],
)

cc_test(
    name = "convex_hull_test",
    srcs = [
        "convex_hull_test.cc",
    ],
    deps = [
        ":collision_checking",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath:matchers",
        "@mobility_diff_drive//diff_drive:test_fixtures",
    ],
)

cc_test(
    name = "danger_hull_test",
    srcs = [
        "danger_hull_test.cc",
    ],
    deps = [
        ":collision_checking",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath:matchers",
        "@mobility_diff_drive//diff_drive:test_fixtures",
    ],
)

cc_library(
    name = "gaussian_mixture_model_2d",
    srcs = [
        "gaussian_mixture_model_2d.cc",
    ],
    hdrs = [
        "gaussian_mixture_model_2d.h",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/functional:any_invocable",
        "@eigenmath//eigenmath",
        "@genit//genit:iterators",
    ],
)

cc_test(
    name = "gaussian_mixture_model_2d_test",
    srcs = [
        "gaussian_mixture_model_2d_test.cc",
    ],
    deps = [
        ":gaussian_mixture_model_2d",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_library(
    name = "goal_geometry",
    srcs = [
        "goal_geometry.cc",
    ],
    hdrs = [
        "goal_geometry.h",
    ],
    deps = [
        ":collision_checking",
        ":grid_common",
        ":lattice_pose",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:variant",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "goal_geometry_test",
    srcs = [
        "goal_geometry_test.cc",
    ],
    deps = [
        ":goal_geometry",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath:matchers",
    ],
)

proto_library(
    name = "goal_geometry_proto",
    srcs = [
        "goal_geometry.proto",
    ],
    deps = [
        ":hull_proto",
        "@eigenmath//eigenmath:eigenmath_proto",
    ],
)

cc_proto_library(
    name = "goal_geometry_cc_proto",
    deps = [":goal_geometry_proto"],
)

cc_library(
    name = "goal_geometry_conversion",
    srcs = [
        "goal_geometry_conversion.cc",
    ],
    hdrs = [
        "goal_geometry_conversion.h",
    ],
    deps = [
        ":goal_geometry",
        ":goal_geometry_cc_proto",
        ":hull_conversion",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@eigenmath//eigenmath:eigenmath_cc_proto",
        "@eigenmath//eigenmath:conversions",
    ],
)

cc_test(
    name = "goal_geometry_conversion_test",
    srcs = [
        "goal_geometry_conversion_test.cc",
    ],
    deps = [
        ":goal_geometry",
        ":goal_geometry_conversion",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "hull_proto",
    srcs = [
        "hull.proto",
    ],
    deps = [
        "@eigenmath//eigenmath:eigenmath_proto",
    ],
)

cc_proto_library(
    name = "hull_cc_proto",
    deps = [":hull_proto"],
)

cc_library(
    name = "hull_conversion",
    srcs = [
        "hull_conversion.cc",
    ],
    hdrs = [
        "hull_conversion.h",
    ],
    deps = [
        ":collision_checking",
        ":hull_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "collision_probabilities",
    srcs = ["collision_probabilities.cc"],
    hdrs = ["collision_probabilities.h"],
    deps = [
        ":collision_checking",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "hull_conversion_test",
    srcs = [
        "hull_conversion_test.cc",
    ],
    deps = [
        ":collision_checking",
        ":hull_conversion",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "convex_hull_conversions",
    srcs = ["convex_hull_conversions.cc"],
    hdrs = ["convex_hull_conversions.h"],
    deps = [
        ":collision_checking",
        "@eigenmath//eigenmath:eigenmath_cc_proto",
        "@eigenmath//eigenmath:conversions",
        "@eigenmath//eigenmath",
    ],
)

cc_library(
    name = "hull_matchers",
    hdrs = ["hull_matchers.h"],
    deps = [
        ":collision_checking",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "cost_grid_utils",
    srcs = [
        "cost_grid_utils.cc",
    ],
    hdrs = [
        "cost_grid_utils.h",
    ],
    deps = [
        ":grid_common",
        ":occupancy_grid",
        ":collision_utils",
    ],
)

cc_test(
    name = "cost_grid_utils_test",
    srcs = [
        "cost_grid_utils_test.cc",
    ],
    deps = [
        ":cost_grid_utils",
        ":grid_test_utils",
        ":collision_checking",
        ":collision_utils",
        "@mobility_diff_drive//diff_drive:test_fixtures",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "grid_common",
    srcs = ["grid_common.cc"],
    hdrs = ["grid_common.h"],
    deps = [
        "@eigenmath//eigenmath",
        "@genit//genit:iterators",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "grid_common_test",
    srcs = ["grid_common_test.cc"],
    deps = [
        ":grid_common",
        ":occupancy_grid",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:sampling",
        "@eigenmath//eigenmath:matchers",
        "@genit//genit:iterators",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "grid_common_proto",
    srcs = [
        "grid_common.proto",
    ],
    deps = ["@eigenmath//eigenmath:eigenmath_proto"],
)

cc_proto_library(
    name = "grid_common_cc_proto",
    deps = [":grid_common_proto"],
)

cc_library(
    name = "grid_common_conversion",
    srcs = [
        "grid_common_conversion.cc",
    ],
    hdrs = [
        "grid_common_conversion.h",
    ],
    deps = [
        ":grid_common",
        ":grid_common_cc_proto",
        "@eigenmath//eigenmath:eigenmath_cc_proto",
        "@eigenmath//eigenmath:conversions",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "grid_common_conversion_test",
    srcs = [
        "grid_common_conversion_test.cc",
    ],
    deps = [
        ":grid_common",
        ":grid_common_cc_proto",
        ":grid_common_conversion",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
        "@eigenmath//eigenmath:eigenmath_cc_proto",
        "@eigenmath//eigenmath:conversions",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "grid_test_utils",
    testonly = True,
    srcs = [
        "grid_test_utils.cc",
    ],
    hdrs = [
        "grid_test_utils.h",
    ],
    deps = [
        ":cost_grid_utils",
        ":grid_common",
        ":occupancy_grid",
        "@eigenmath//eigenmath",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "grid_test_utils_test",
    srcs = ["grid_test_utils_test.cc"],
    deps = [
        ":grid_test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "lattice_pose",
    srcs = [],
    hdrs = [
        "lattice_pose.h",
    ],
    deps = [
        ":grid_common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/hash",
    ],
)

cc_test(
    name = "lattice_pose_test",
    srcs = ["lattice_pose_test.cc"],
    deps = [
        ":lattice_pose",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:sampling",
        "@eigenmath//eigenmath:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "occupancy_grid",
    hdrs = [
        "occupancy_grid.h",
    ],
    deps = [
        ":grid_common",
        "@eigenmath//eigenmath",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "occupancy_grid_test",
    srcs = [
        "occupancy_grid_test.cc",
    ],
    deps = [
        ":occupancy_grid",
        "@eigenmath//eigenmath:sampling",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_library(
    name = "occupancy_grid_utils",
    srcs = ["occupancy_grid_utils.cc"],
    hdrs = ["occupancy_grid_utils.h"],
    deps = [
        ":grid_common",
        ":occupancy_grid",
        "@eigenmath//eigenmath",
        "@genit//genit:iterators",
        ":collision_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "occupancy_grid_utils_test",
    srcs = ["occupancy_grid_utils_test.cc"],
    deps = [
        ":grid_common",
        ":grid_test_utils",
        ":occupancy_grid",
        ":occupancy_grid_utils",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
        "@genit//genit:iterators",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "occupancy_grid_proto",
    srcs = [
        "occupancy_grid.proto",
    ],
    deps = [
        ":grid_common_proto",
    ],
)

cc_proto_library(
    name = "occupancy_grid_cc_proto",
    deps = [":occupancy_grid_proto"],
)

cc_library(
    name = "occupancy_grid_conversion",
    srcs = ["occupancy_grid_conversion.cc"],
    hdrs = ["occupancy_grid_conversion.h"],
    deps = [
        ":grid_common",
        ":grid_common_cc_proto",
        ":grid_common_conversion",
        ":occupancy_grid",
        ":occupancy_grid_cc_proto",
        "@eigenmath//eigenmath:eigenmath_cc_proto",
        "@eigenmath//eigenmath:conversions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "occupancy_grid_conversion_test",
    srcs = [
        "occupancy_grid_conversion_test.cc",
    ],
    deps = [
        ":grid_common",
        ":occupancy_grid_conversion",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest_main",
    ],
)

