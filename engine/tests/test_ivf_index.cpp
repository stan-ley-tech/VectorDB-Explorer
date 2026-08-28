#include <algorithm>
#include <doctest/doctest.h>

#include "vdb/ivf_index.hpp"
#include "vdb/search.hpp"

using namespace vdb;

namespace {

// Three well-separated 2D clusters, five points each, so an IVF index with
// nlist=3 should recover exactly this partition and nprobe=1 should still
// find the true nearest neighbor for a query planted inside one cluster.
FlatVectorStore make_clustered_store() {
    FlatVectorStore store(2);

    const float cluster_a[5][2] = {{0.0f, 0.0f}, {0.1f, 0.0f}, {0.0f, 0.1f}, {0.1f, 0.1f}, {-0.1f, 0.1f}};
    const float cluster_b[5][2] = {{100.0f, 0.0f}, {100.1f, 0.0f}, {100.0f, 0.1f}, {100.1f, 0.1f}, {99.9f, 0.1f}};
    const float cluster_c[5][2] = {{0.0f, 100.0f}, {0.1f, 100.0f}, {0.0f, 100.1f}, {0.1f, 100.1f}, {-0.1f, 100.1f}};

    for (int i = 0; i < 5; ++i) store.upsert(static_cast<std::uint64_t>(1 + i), cluster_a[i]);
    for (int i = 0; i < 5; ++i) store.upsert(static_cast<std::uint64_t>(6 + i), cluster_b[i]);
    for (int i = 0; i < 5; ++i) store.upsert(static_cast<std::uint64_t>(11 + i), cluster_c[i]);

    return store;
}

}  // namespace

TEST_CASE("build rejects a store with fewer vectors than nlist") {
    FlatVectorStore store(2);
    const float v[] = {1.0f, 1.0f};
    store.upsert(1, v);

    IVFIndex index(2, Metric::Euclidean, 5);
    CHECK_THROWS_AS(index.build(store), std::invalid_argument);
}

TEST_CASE("build rejects a store with mismatched dimension") {
    FlatVectorStore store = make_clustered_store();
    IVFIndex index(3, Metric::Euclidean, 3);
    CHECK_THROWS_AS(index.build(store), std::invalid_argument);
}

TEST_CASE("search before build throws") {
    FlatVectorStore store = make_clustered_store();
    IVFIndex index(2, Metric::Euclidean, 3);
    const float query[] = {0.0f, 0.0f};
    CHECK_THROWS_AS(index.search(store, query, 1, 1), std::logic_error);
}

TEST_CASE("search after the store mutates past the built version throws") {
    FlatVectorStore store = make_clustered_store();
    IVFIndex index(2, Metric::Euclidean, 3);
    index.build(store);

    const float extra[] = {50.0f, 50.0f};
    store.upsert(999, extra);

    const float query[] = {0.0f, 0.0f};
    CHECK_THROWS_AS(index.search(store, query, 1, 1), std::logic_error);
}

TEST_CASE("cluster_sizes sums to the number of vectors in the store") {
    FlatVectorStore store = make_clustered_store();
    IVFIndex index(2, Metric::Euclidean, 3);
    index.build(store);

    auto sizes = index.cluster_sizes();
    REQUIRE(sizes.size() == 3);
    CHECK(sizes[0] + sizes[1] + sizes[2] == store.size());
}

TEST_CASE("nprobe == nlist matches brute-force search exactly") {
    FlatVectorStore store = make_clustered_store();
    IVFIndex index(2, Metric::Cosine, 3);
    index.build(store);

    const float query[] = {0.05f, 0.05f};
    auto ivf_results = index.search(store, query, 4, /*nprobe=*/3);
    auto brute_results = brute_force_search(store, query, 4, Metric::Cosine);

    REQUIRE(ivf_results.size() == brute_results.size());
    for (std::size_t i = 0; i < ivf_results.size(); ++i) {
        CHECK(ivf_results[i].id == brute_results[i].id);
        CHECK(ivf_results[i].score == doctest::Approx(brute_results[i].score));
    }
}

TEST_CASE("nprobe of 1 still finds the true nearest neighbor inside a well-separated cluster") {
    FlatVectorStore store = make_clustered_store();
    IVFIndex index(2, Metric::Euclidean, 3);
    index.build(store);

    const float query[] = {0.05f, 0.05f};
    auto results = index.search(store, query, 1, /*nprobe=*/1);

    REQUIRE(results.size() == 1);
    CHECK(results[0].id >= 1);
    CHECK(results[0].id <= 5);
}
