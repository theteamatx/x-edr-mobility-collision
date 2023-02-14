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

#ifndef MOBILITY_COLLISION_COLLISION_GAUSSIAN_MIXTURE_MODEL_2D_H_
#define MOBILITY_COLLISION_COLLISION_GAUSSIAN_MIXTURE_MODEL_2D_H_

#include <vector>

#include "eigenmath/normal_distribution.h"
#include "eigenmath/types.h"

namespace mobility::collision {

// Representation of a 2D Gaussian mixture model probability distribution.
class GaussianMixtureModel2d {
 public:
  // Constructs a Gaussian Mixture model with specified number of components.
  explicit GaussianMixtureModel2d(int num_components);

  // Sets the mean of the component of mixture model.
  void SetMean(const eigenmath::Vector2d& mean, int num_component);

  // Sets the covariance of the component of mixture model.
  void SetCovariance(const eigenmath::Matrix2d& covariance, int num_component);

  // Sets the weight of the component of mixture model.
  void SetWeight(const double weight, int num_component);

  // Returns the mean of the component of mixture model.
  const eigenmath::Vector2d& Mean(int num_component) const;

  // Returns the covariance of the component of mixture model.
  const eigenmath::Matrix2d& Covariance(int num_component) const;

  // Returns the weight of the component of mixture model.
  const double& Weight(int num_component) const;

  // Returns the probability of a point belonging to a component based on
  // the parameters of the mixture model
  double Probability(const eigenmath::Vector2d& point, int num_component) const;

  // Returns the number of components in the mixture model.
  int num_components() const { return components_.size(); }

  // Returns the means of all components of mixture model as a vector.
  std::vector<eigenmath::Vector2d> Means() const;

  // Returns the covariances of all the components of mixture model as a vector.
  std::vector<eigenmath::Matrix2d> Covariances() const;

  // Returns the weights of all the components of mixture model as a vector.
  std::vector<double> Weights() const;

 private:
  struct Component {
    eigenmath::EuclideanNormalDistribution<double, 2> dist;
    double weight;
  };

  std::vector<Component> components_;
};

// Estimates the parameters of 2d gaussian distribution using EM algorithm
// for the input `samples`. The `num_components` parameter determines the number
// of Gaussian distributions in the mixture model. The `max_iterations`
// parameter limits the number of EM update steps. The `log_prob_change_thresh`
// parameter can be used for early stopping if the change in the mean of log
// probabilities for each sample is below the threshold between two consecutive
// updates. The `weights` parameter can be used to specify the contribution of
// each point in the mixture model.
GaussianMixtureModel2d FitGaussianMixture(
    const std::vector<eigenmath::Vector2d>& samples, const int num_components,
    const int max_iterations, const double log_prob_change_thresh = 0.0,
    const std::vector<double>& weights = {});

// Assigns the input `points` to the mixture model component with highest
// probability for the point. Returns a vector of size model.num_components
// where each element is a vector of the indices of points belonging to the
// component.
std::vector<std::vector<int>> Cluster(
    const GaussianMixtureModel2d& model,
    const std::vector<eigenmath::Vector2d>& points);

// Normalize elliptical distance to equal 1.0 for the farthest points and then
// add `margin` to the principal axes.
// Returns the expanded covariance matrix, no longer representing a covariance
// but an ellipse that covers the given cluster of points.
eigenmath::Matrix2d ScaleAndExpandEllipse(
    const eigenmath::Vector2d& mean, const eigenmath::Matrix2d& covariance,
    const std::vector<int>& assignments,
    const std::vector<eigenmath::Vector2d>& points, double margin);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_GAUSSIAN_MIXTURE_MODEL_2D_H_
