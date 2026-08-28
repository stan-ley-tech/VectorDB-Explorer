#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "vdb/distance.hpp"
#include "vdb/search.hpp"
#include "vdb/vector_store.hpp"

namespace vdb {

// A minimal inverted-file (IVF) approximate nearest-neighbor index.
//
// Build partitions the vector space into `nlist` clusters with Lloyd's
// k-means (k-means++ initialization, squared Euclidean assignment — the
// objective Lloyd's algorithm actually minimizes, independent of the
// metric used at query time). Each cluster keeps the row indices of the
// vectors assigned to it.
//
// Search probes only the `nprobe` clusters whose centroid is closest to the
// query, then runs an exact top-k scan restricted to those clusters. This
// trades recall for speed: nprobe == nlist is equivalent to brute force;
// nprobe == 1 is fastest and least accurate.
//
// The index is a point-in-time snapshot of the store it was built from. It
// does not track subsequent inserts/removes — call build() again after the
// store changes. search() checks the store's version() and throws if it has
// drifted from the version at build time, rather than silently scanning
// stale row indices.
class IVFIndex {
public:
    IVFIndex(std::size_t dim, Metric metric, std::size_t nlist)
        : dim_(dim), metric_(metric), nlist_(nlist) {
        if (dim_ == 0) {
            throw std::invalid_argument("vector dimension must be positive");
        }
        if (nlist_ == 0) {
            throw std::invalid_argument("nlist must be positive");
        }
    }

    void build(const FlatVectorStore& store, std::size_t kmeans_iterations = 15, unsigned seed = 42);

    std::vector<SearchResult> search(
        const FlatVectorStore& store,
        const float* query,
        std::size_t top_k,
        std::size_t nprobe) const;

    bool is_built() const { return built_; }
    std::size_t nlist() const { return nlist_; }
    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }

    // Number of vectors assigned to each cluster, in cluster order. Useful
    // for diagnosing skewed partitions (e.g. one giant cluster that makes
    // nprobe scans no better than brute force).
    std::vector<std::size_t> cluster_sizes() const;

private:
    void initialize_centroids_kmeans_pp(const FlatVectorStore& store, std::mt19937& rng);
    void assign_clusters(const FlatVectorStore& store, std::vector<std::size_t>& assignment) const;
    bool recompute_centroids(const FlatVectorStore& store, const std::vector<std::size_t>& assignment, std::mt19937& rng);

    std::size_t dim_;
    Metric metric_;
    std::size_t nlist_;
    std::vector<float> centroids_;  // nlist_ rows of dim_ floats, row-major
    std::vector<std::vector<std::size_t>> inverted_lists_;
    std::uint64_t built_store_version_ = 0;
    bool built_ = false;
};

}  // namespace vdb
