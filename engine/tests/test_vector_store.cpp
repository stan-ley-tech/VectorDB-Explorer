#include <doctest/doctest.h>

#include "vdb/vector_store.hpp"

using namespace vdb;

TEST_CASE("zero dimension is rejected at construction") {
    CHECK_THROWS_AS(FlatVectorStore(0), std::invalid_argument);
}

TEST_CASE("upsert inserts new rows and grows size") {
    FlatVectorStore store(3);
    const float v1[] = {1.0f, 2.0f, 3.0f};
    const float v2[] = {4.0f, 5.0f, 6.0f};

    store.upsert(10, v1);
    store.upsert(20, v2);

    CHECK(store.size() == 2);
    CHECK(store.contains(10));
    CHECK(store.contains(20));
    CHECK_FALSE(store.contains(30));
}

TEST_CASE("upsert on an existing id overwrites in place without growing") {
    FlatVectorStore store(2);
    const float original[] = {1.0f, 1.0f};
    const float updated[] = {9.0f, 9.0f};

    store.upsert(1, original);
    store.upsert(1, updated);

    CHECK(store.size() == 1);
    const float* stored = store.vector_for_id(1);
    CHECK(stored[0] == doctest::Approx(9.0f));
    CHECK(stored[1] == doctest::Approx(9.0f));
}

TEST_CASE("vector_for_id throws for a missing id") {
    FlatVectorStore store(2);
    const float v[] = {1.0f, 1.0f};
    store.upsert(1, v);
    CHECK_THROWS_AS(store.vector_for_id(999), std::out_of_range);
}

TEST_CASE("remove swaps the last row into the removed slot") {
    FlatVectorStore store(1);
    const float a[] = {1.0f};
    const float b[] = {2.0f};
    const float c[] = {3.0f};

    store.upsert(1, a);
    store.upsert(2, b);
    store.upsert(3, c);

    CHECK(store.remove(2));
    CHECK(store.size() == 2);
    CHECK_FALSE(store.contains(2));

    // The remaining ids must still resolve to their own vectors after the
    // swap-remove reshuffled row indices internally.
    CHECK(store.vector_for_id(1)[0] == doctest::Approx(1.0f));
    CHECK(store.vector_for_id(3)[0] == doctest::Approx(3.0f));
}

TEST_CASE("removing an id that was never inserted returns false") {
    FlatVectorStore store(1);
    CHECK_FALSE(store.remove(42));
}

TEST_CASE("version increments on every mutation, not on reads") {
    FlatVectorStore store(1);
    const float v[] = {1.0f};

    CHECK(store.version() == 0);
    store.upsert(1, v);
    CHECK(store.version() == 1);
    store.upsert(1, v);  // overwrite counts as a mutation too
    CHECK(store.version() == 2);

    const auto ignored_size = store.size();
    (void)ignored_size;
    CHECK(store.version() == 2);

    store.remove(1);
    CHECK(store.version() == 3);
}
