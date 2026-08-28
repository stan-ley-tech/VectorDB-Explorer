#include "vdb/search.hpp"

#include <algorithm>
#include <queue>

namespace vdb {

namespace {

struct ScoredRow {
    float score;
    std::size_t row;
};

// Min-heap ordered by score, so the weakest of the current top-k candidates
// is always at the top and cheap to evict.
struct MinHeapCompare {
    bool operator()(const ScoredRow& a, const ScoredRow& b) const {
        return a.score > b.score;
    }
};

}  // namespace

std::vector<SearchResult> brute_force_search(
    const FlatVectorStore& store,
    const float* query,
    std::size_t top_k,
    Metric metric) {
    if (top_k == 0 || store.size() == 0) {
        return {};
    }

    const std::size_t dim = store.dim();
    const std::size_t n = store.size();
    const std::size_t k = std::min(top_k, n);

    std::priority_queue<ScoredRow, std::vector<ScoredRow>, MinHeapCompare> heap;

    for (std::size_t row = 0; row < n; ++row) {
        const float score = ranking_score(metric, query, store.row(row), dim);
        if (heap.size() < k) {
            heap.push({score, row});
        } else if (score > heap.top().score) {
            heap.pop();
            heap.push({score, row});
        }
    }

    std::vector<ScoredRow> ordered;
    ordered.reserve(heap.size());
    while (!heap.empty()) {
        ordered.push_back(heap.top());
        heap.pop();
    }
    std::sort(ordered.begin(), ordered.end(), [](const ScoredRow& a, const ScoredRow& b) {
        return a.score > b.score;
    });

    std::vector<SearchResult> results;
    results.reserve(ordered.size());
    const auto& ids = store.ids();
    for (const auto& entry : ordered) {
        results.push_back({ids[entry.row], entry.score});
    }
    return results;
}

}  // namespace vdb
