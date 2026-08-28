#include "vdb/ivf_index.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "vdb/topk_heap.hpp"

namespace vdb {

void IVFIndex::initialize_centroids_kmeans_pp(const FlatVectorStore& store, std::mt19937& rng) {
    const std::size_t n = store.size();
    centroids_.assign(nlist_ * dim_, 0.0f);

    std::uniform_int_distribution<std::size_t> uniform(0, n - 1);
    const std::size_t first = uniform(rng);
    std::copy(store.row(first), store.row(first) + dim_, centroids_.begin());

    std::vector<float> nearest_sq_dist(n, std::numeric_limits<float>::max());

    for (std::size_t c = 1; c < nlist_; ++c) {
        const float* prev_centroid = centroids_.data() + (c - 1) * dim_;
        for (std::size_t i = 0; i < n; ++i) {
            const float d = squared_euclidean_distance(store.row(i), prev_centroid, dim_);
            nearest_sq_dist[i] = std::min(nearest_sq_dist[i], d);
        }

        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total += nearest_sq_dist[i];
        }

        std::size_t chosen = n - 1;
        if (total > 0.0) {
            std::uniform_real_distribution<double> dist(0.0, total);
            const double target = dist(rng);
            double cumulative = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                cumulative += nearest_sq_dist[i];
                if (cumulative >= target) {
                    chosen = i;
                    break;
                }
            }
        } else {
            // All remaining points coincide with an existing centroid; any
            // point is as good a seed as any other.
            chosen = uniform(rng);
        }

        std::copy(store.row(chosen), store.row(chosen) + dim_, centroids_.begin() + static_cast<std::ptrdiff_t>(c * dim_));
    }
}

void IVFIndex::assign_clusters(const FlatVectorStore& store, std::vector<std::size_t>& assignment) const {
    const std::size_t n = store.size();
    assignment.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        float best_dist = std::numeric_limits<float>::max();
        std::size_t best_cluster = 0;
        for (std::size_t c = 0; c < nlist_; ++c) {
            const float d = squared_euclidean_distance(store.row(i), centroids_.data() + c * dim_, dim_);
            if (d < best_dist) {
                best_dist = d;
                best_cluster = c;
            }
        }
        assignment[i] = best_cluster;
    }
}

bool IVFIndex::recompute_centroids(const FlatVectorStore& store, const std::vector<std::size_t>& assignment, std::mt19937& rng) {
    const std::size_t n = store.size();
    std::vector<double> sums(nlist_ * dim_, 0.0);
    std::vector<std::size_t> counts(nlist_, 0);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = assignment[i];
        counts[c] += 1;
        const float* v = store.row(i);
        double* sum_row = sums.data() + c * dim_;
        for (std::size_t d = 0; d < dim_; ++d) {
            sum_row[d] += v[d];
        }
    }

    std::uniform_int_distribution<std::size_t> uniform(0, n - 1);
    float max_shift = 0.0f;

    for (std::size_t c = 0; c < nlist_; ++c) {
        float* centroid_row = centroids_.data() + c * dim_;

        if (counts[c] == 0) {
            // Empty cluster: reseed from a random data point so it has a
            // chance to pick up members on the next iteration instead of
            // sitting permanently empty.
            const std::size_t replacement = uniform(rng);
            std::copy(store.row(replacement), store.row(replacement) + dim_, centroid_row);
            max_shift = std::numeric_limits<float>::max();
            continue;
        }

        std::vector<float> new_centroid(dim_);
        const double* sum_row = sums.data() + c * dim_;
        for (std::size_t d = 0; d < dim_; ++d) {
            new_centroid[d] = static_cast<float>(sum_row[d] / static_cast<double>(counts[c]));
        }

        const float shift = euclidean_distance(centroid_row, new_centroid.data(), dim_);
        max_shift = std::max(max_shift, shift);
        std::copy(new_centroid.begin(), new_centroid.end(), centroid_row);
    }

    constexpr float kConvergenceEpsilon = 1e-4f;
    return max_shift < kConvergenceEpsilon;
}

void IVFIndex::build(const FlatVectorStore& store, std::size_t kmeans_iterations, unsigned seed) {
    if (store.dim() != dim_) {
        throw std::invalid_argument("store dimension does not match index dimension");
    }
    if (store.size() < nlist_) {
        throw std::invalid_argument("store must contain at least nlist vectors to build an IVF index");
    }

    std::mt19937 rng(seed);
    initialize_centroids_kmeans_pp(store, rng);

    std::vector<std::size_t> assignment;
    for (std::size_t iter = 0; iter < kmeans_iterations; ++iter) {
        assign_clusters(store, assignment);
        if (recompute_centroids(store, assignment, rng)) {
            break;
        }
    }
    assign_clusters(store, assignment);

    inverted_lists_.assign(nlist_, {});
    for (std::size_t i = 0; i < assignment.size(); ++i) {
        inverted_lists_[assignment[i]].push_back(i);
    }

    built_store_version_ = store.version();
    built_ = true;
}

std::vector<SearchResult> IVFIndex::search(
    const FlatVectorStore& store,
    const float* query,
    std::size_t top_k,
    std::size_t nprobe) const {
    if (!built_) {
        throw std::logic_error("IVFIndex::search called before build()");
    }
    if (store.version() != built_store_version_) {
        throw std::logic_error("vector store has changed since the index was built; call build() again");
    }

    nprobe = std::min(nprobe, nlist_);
    if (nprobe == 0) {
        return {};
    }

    std::vector<std::pair<float, std::size_t>> centroid_scores;
    centroid_scores.reserve(nlist_);
    for (std::size_t c = 0; c < nlist_; ++c) {
        const float score = ranking_score(metric_, query, centroids_.data() + c * dim_, dim_);
        centroid_scores.emplace_back(score, c);
    }
    std::partial_sort(
        centroid_scores.begin(),
        centroid_scores.begin() + static_cast<std::ptrdiff_t>(nprobe),
        centroid_scores.end(),
        [](const std::pair<float, std::size_t>& a, const std::pair<float, std::size_t>& b) {
            return a.first > b.first;
        });

    std::vector<std::size_t> candidates;
    for (std::size_t p = 0; p < nprobe; ++p) {
        const std::size_t cluster = centroid_scores[p].second;
        const auto& list = inverted_lists_[cluster];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    return select_top_k(candidates, store.ids(), top_k, [&](std::size_t row) {
        return ranking_score(metric_, query, store.row(row), dim_);
    });
}

std::vector<std::size_t> IVFIndex::cluster_sizes() const {
    std::vector<std::size_t> sizes;
    sizes.reserve(inverted_lists_.size());
    for (const auto& list : inverted_lists_) {
        sizes.push_back(list.size());
    }
    return sizes;
}

}  // namespace vdb
