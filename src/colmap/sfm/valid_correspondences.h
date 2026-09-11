#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

// Visit each retained match once in observation order. Pair-by-pair extraction
// rescans an image's entire adjacency list for each of its neighbors.
// Image indices follow image_ids; only valid pose-graph edges are visited.
template <typename Callback>
void ForEachValidCorrespondence(
    const CorrespondenceGraph& graph,
    const PoseGraph& pose_graph,
    const Reconstruction& reconstruction,
    const std::vector<image_t>& image_ids,
    const FlatHashMap<image_t, size_t>& image_indices,
    Callback callback) {
  std::vector<FlatHashMap<image_t, size_t>> neighbors(image_ids.size());
  for (const auto& pair : pose_graph.ValidEdges()) {
    const auto [first, second] = PairIdToImagePair(pair.first);
    neighbors[image_indices.at(first)].emplace(second,
                                               image_indices.at(second));
  }
  for (size_t first = 0; first < image_ids.size(); ++first) {
    if (neighbors[first].empty()) continue;
    const image_t image_id = image_ids[first];
    const auto num_points = reconstruction.Image(image_id).NumPoints2D();
    for (point2D_t point = 0; point < num_points; ++point) {
      const auto range = graph.FindCorrespondences(image_id, point);
      for (auto corr = range.beg; corr != range.end; ++corr) {
        const auto neighbor = neighbors[first].find(corr->image_id);
        if (neighbor != neighbors[first].end()) {
          callback(first, point, neighbor->second, corr->point2D_idx);
        }
      }
    }
  }
}

}  // namespace colmap
