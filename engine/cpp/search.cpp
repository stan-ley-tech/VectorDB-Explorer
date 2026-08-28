#include "vdb/search.hpp"

#include <numeric>

#include "vdb/topk_heap.hpp"

namespace vdb {

std::vector<SearchResult> brute_force_search(
    const FlatVectorStore& store,
    const float* query,
    std::size_t top_k,
    Metric metric) {
    const std::size_t n = store.size();
    std::vector<std::size_t> all_rows(n);
    std::iota(all_rows.begin(), all_rows.end(), 0);

    const std::size_t dim = store.dim();
    return select_top_k(all_rows, store.ids(), top_k, [&](std::size_t row) {
        return ranking_score(metric, query, store.row(row), dim);
    });
}

}  // namespace vdb
