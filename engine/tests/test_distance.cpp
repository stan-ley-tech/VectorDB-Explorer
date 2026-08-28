#include <doctest/doctest.h>

#include "vdb/distance.hpp"

using namespace vdb;

TEST_CASE("dot product of orthogonal vectors is zero") {
    const float a[] = {1.0f, 0.0f};
    const float b[] = {0.0f, 1.0f};
    CHECK(dot_product(a, b, 2) == doctest::Approx(0.0f));
}

TEST_CASE("cosine similarity of identical vectors is one") {
    const float a[] = {1.0f, 2.0f, 3.0f};
    CHECK(cosine_similarity(a, a, 3) == doctest::Approx(1.0f));
}

TEST_CASE("cosine similarity of opposite vectors is minus one") {
    const float a[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {-1.0f, -2.0f, -3.0f};
    CHECK(cosine_similarity(a, b, 3) == doctest::Approx(-1.0f));
}

TEST_CASE("cosine similarity of orthogonal vectors is zero") {
    const float a[] = {1.0f, 0.0f};
    const float b[] = {0.0f, 1.0f};
    CHECK(cosine_similarity(a, b, 2) == doctest::Approx(0.0f));
}

TEST_CASE("cosine similarity handles a zero-norm vector without dividing by zero") {
    const float a[] = {0.0f, 0.0f};
    const float b[] = {1.0f, 1.0f};
    CHECK(cosine_similarity(a, b, 2) == doctest::Approx(0.0f));
}

TEST_CASE("euclidean distance matches the classic 3-4-5 triangle") {
    const float a[] = {0.0f, 0.0f};
    const float b[] = {3.0f, 4.0f};
    CHECK(euclidean_distance(a, b, 2) == doctest::Approx(5.0f));
}

TEST_CASE("euclidean distance of a vector to itself is zero") {
    const float a[] = {5.0f, -2.0f, 7.5f};
    CHECK(euclidean_distance(a, a, 3) == doctest::Approx(0.0f));
}

TEST_CASE("ranking_score is descending-comparable across metrics") {
    const float query[] = {1.0f, 0.0f};
    const float close[] = {0.9f, 0.1f};
    const float far[] = {-1.0f, 0.0f};

    SUBCASE("cosine: closer vector scores higher") {
        CHECK(ranking_score(Metric::Cosine, query, close, 2) > ranking_score(Metric::Cosine, query, far, 2));
    }
    SUBCASE("dot product: closer vector scores higher") {
        CHECK(ranking_score(Metric::DotProduct, query, close, 2) > ranking_score(Metric::DotProduct, query, far, 2));
    }
    SUBCASE("euclidean: closer vector still scores higher (negated distance)") {
        CHECK(ranking_score(Metric::Euclidean, query, close, 2) > ranking_score(Metric::Euclidean, query, far, 2));
    }
}

TEST_CASE("higher_is_better reflects each metric's natural direction") {
    CHECK(higher_is_better(Metric::Cosine) == true);
    CHECK(higher_is_better(Metric::DotProduct) == true);
    CHECK(higher_is_better(Metric::Euclidean) == false);
}

TEST_CASE("metric name round-trips through string conversion") {
    CHECK(metric_from_string("cosine") == Metric::Cosine);
    CHECK(metric_from_string("euclidean") == Metric::Euclidean);
    CHECK(metric_from_string("dot") == Metric::DotProduct);
    CHECK(std::string(metric_to_string(Metric::Cosine)) == "cosine");
}

TEST_CASE("metric_from_string rejects unknown names") {
    CHECK_THROWS_AS(metric_from_string("manhattan"), std::invalid_argument);
}
