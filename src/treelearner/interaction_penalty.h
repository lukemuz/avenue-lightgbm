/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_INTERACTION_PENALTY_H_
#define LIGHTGBM_TREELEARNER_INTERACTION_PENALTY_H_

#include <vector>
#include <set>
#include <algorithm>
#include <mutex>

namespace LightGBM {

class InteractionPenalty {
public:
    InteractionPenalty(double penalty, double complexity) : penalty_(penalty), complexity_(complexity) {}


    // Initialize method to clear used features
    void Init() {
        std::lock_guard<std::mutex> lock(mutex_);
        used_features_.clear();
        current_tree_features_.clear();
    }

    // Method to calculate the penalty for a given feature combination
    double CalculatePenalty(int feature) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::set<int> tree_features = current_tree_features_;

        bool is_feature_new = tree_features.insert(feature).second;
        if (!is_feature_new) {
            // LightGBM::Log::Warning("Feature %d is already in the tree", feature);
            return 0.0;  // No penalty if the feature is already in the tree
        }

        // Check against previously used combinations (feature already inserted above)
        for (const auto& set : used_features_) {
            if (std::includes(set.begin(), set.end(), tree_features.begin(), tree_features.end())) {
                // LightGBM::Log::Warning("Feature combination has been used before");
                return 0.0;  // No penalty if the combination has been used before
            }
        }
        // LightGBM::Log::Info("Feature combination is new");
        // Calculate a transformed penalty based on the number of features in the tree
        double transformed_penalty =  penalty_ * tree_features.size();
        return transformed_penalty;
    }

    double CalculateComplexityPenalty(int feature) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::set<int> tree_features = current_tree_features_;
        bool is_feature_new = tree_features.insert(feature).second;
        if (!is_feature_new) {
            // LightGBM::Log::Warning("Feature %d is already in the tree", feature);
            return 0.0;  // No penalty if the feature is already in the tree
        }

        return complexity_ * tree_features.size();


    }


    // Method to update the used features after a tree is constructed
    void UpdateUsedFeatures() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!current_tree_features_.empty()) {  // Only add non-empty sets
            used_features_.push_back(current_tree_features_);
        }
    }

    void ClearCurrentTreeFeatures() {
        std::lock_guard<std::mutex> lock(mutex_);
        current_tree_features_.clear();
    }

    void AddCurrentTreeFeature(int feature) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_tree_features_.insert(feature);
    }

private:
    mutable std::mutex mutex_;  // Mutable to allow locking in const methods
    double penalty_;
    double complexity_;
    std::vector<std::set<int>> used_features_;
    std::set<int> current_tree_features_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_INTERACTION_PENALTY_H_
