#include <doctest/doctest.h>

#include "vdb/search.hpp"

using namespace vdb;

namespace {

FlatVectorStore make_fixture_store() {
    FlatVectorStore store(2);
    const float id1[] = {1.0f, 0.0f};    // identical to the query below
    const float id2[] = {0.0f, 1.0f};    // orthogonal
    const float id3[] = {-1.0f, 0.0f};   // opposite
    const float id4[] = {0.9f, 0.1f};    // near-identical
    store.upsert(1, id1);
    store.upsert(2, id2);
    store.upsert(3, id3);
    store.upsert(4, id4);
    return store;
}

}  // namespace

TEST_CASE("brute_force_search on an empty store returns no results") {
    FlatVectorStore store(2);
    const float query[] = {1.0f, 0.0f};
    auto results = brute_force_search(store, query, 5, Metric::Cosine);
    CHECK(results.empty());
}

TEST_CASE("brute_force_search with top_k of zero returns no results") {
    FlatVectorStore store = make_fixture_store();
    const float query[] = {1.0f, 0.0f};
    auto results = brute_force_search(store, query, 0, Metric::Cosine);
    CHECK(results.empty());
}

TEST_CASE("brute_force_search ranks by cosine similarity, best first") {
    FlatVectorStore store = make_fixture_store();
    const float query[] = {1.0f, 0.0f};

    auto results = brute_force_search(store, query, 2, Metric::Cosine);

    REQUIRE(results.size() == 2);
    CHECK(results[0].id == 1);
    CHECK(results[0].score == doctest::Approx(1.0f));
    CHECK(results[1].id == 4);
}

TEST_CASE("brute_force_search ranks by euclidean distance, closest first") {
    FlatVectorStore store = make_fixture_store();
    const float query[] = {1.0f, 0.0f};

    auto results = brute_force_search(store, query, 2, Metric::Euclidean);

    REQUIRE(results.size() == 2);
    CHECK(results[0].id == 1);
    CHECK(results[1].id == 4);
}

TEST_CASE("brute_force_search ranks by dot product, best first") {
    FlatVectorStore store = make_fixture_store();
    const float query[] = {1.0f, 0.0f};

    auto results = brute_force_search(store, query, 2, Metric::DotProduct);

    REQUIRE(results.size() == 2);
    CHECK(results[0].id == 1);
    CHECK(results[1].id == 4);
}

TEST_CASE("brute_force_search clamps top_k to the store size") {
    FlatVectorStore store = make_fixture_store();
    const float query[] = {1.0f, 0.0f};

    auto results = brute_force_search(store, query, 100, Metric::Cosine);

    CHECK(results.size() == 4);
}
