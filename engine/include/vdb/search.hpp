#pragma once

#include <cstdint>
#include <vector>

#include "vdb/distance.hpp"
#include "vdb/vector_store.hpp"

namespace vdb {

struct SearchResult {
    std::uint64_t id;
    float score;
};

// Exhaustive nearest-neighbor search over every vector in `store`. Always
// exact, since it never skips a candidate. Runs in O(n * dim) time and
// O(top_k) extra memory: a bounded min-heap tracks the current top-k
// candidates instead of sorting the entire result set, which matters once
// `store` holds millions of rows and top_k is small (the common case).
//
// Results are returned best-first, regardless of metric.
std::vector<SearchResult> brute_force_search(
    const FlatVectorStore& store,
    const float* query,
    std::size_t top_k,
    Metric metric);

}  // namespace vdb
