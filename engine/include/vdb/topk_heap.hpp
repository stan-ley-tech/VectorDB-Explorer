#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#include "vdb/search.hpp"

namespace vdb {

namespace detail {

struct ScoredRow {
    float score;
    std::size_t row;
};

struct MinHeapCompare {
    bool operator()(const ScoredRow& a, const ScoredRow& b) const {
        return a.score > b.score;
    }
};

}  // namespace detail

// Shared by brute-force and IVF search: scans `candidate_rows`, scoring each
// with `score_fn(row)`, and keeps only the best `top_k` using a bounded
// min-heap (O(candidates * log top_k) instead of a full sort). Both search
// strategies differ only in which rows they consider, so this is the one
// place that owns "keep the best k and return them best-first".
template <typename ScoreFn>
std::vector<SearchResult> select_top_k(
    const std::vector<std::size_t>& candidate_rows,
    const std::vector<std::uint64_t>& ids,
    std::size_t top_k,
    ScoreFn score_fn) {
    if (top_k == 0 || candidate_rows.empty()) {
        return {};
    }

    const std::size_t k = std::min(top_k, candidate_rows.size());
    std::priority_queue<detail::ScoredRow, std::vector<detail::ScoredRow>, detail::MinHeapCompare> heap;

    for (const std::size_t row : candidate_rows) {
        const float score = score_fn(row);
        if (heap.size() < k) {
            heap.push({score, row});
        } else if (score > heap.top().score) {
            heap.pop();
            heap.push({score, row});
        }
    }

    std::vector<detail::ScoredRow> ordered;
    ordered.reserve(heap.size());
    while (!heap.empty()) {
        ordered.push_back(heap.top());
        heap.pop();
    }
    std::sort(ordered.begin(), ordered.end(), [](const detail::ScoredRow& a, const detail::ScoredRow& b) {
        return a.score > b.score;
    });

    std::vector<SearchResult> results;
    results.reserve(ordered.size());
    for (const auto& entry : ordered) {
        results.push_back({ids[entry.row], entry.score});
    }
    return results;
}

}  // namespace vdb
