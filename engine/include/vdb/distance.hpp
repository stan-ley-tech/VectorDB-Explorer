#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace vdb {

// Supported similarity/distance metrics for vector comparison.
enum class Metric {
    Cosine,
    Euclidean,
    DotProduct
};

inline Metric metric_from_string(const std::string& name) {
    if (name == "cosine") return Metric::Cosine;
    if (name == "euclidean") return Metric::Euclidean;
    if (name == "dot") return Metric::DotProduct;
    throw std::invalid_argument("unknown metric: " + name);
}

inline const char* metric_to_string(Metric metric) {
    switch (metric) {
        case Metric::Cosine: return "cosine";
        case Metric::Euclidean: return "euclidean";
        case Metric::DotProduct: return "dot";
    }
    return "unknown";
}

// True when a higher score means a better (closer) match for this metric.
// Cosine and dot product are similarities (higher is better); Euclidean is
// a raw distance (lower is better), so callers need to know which way to
// sort a result set.
inline bool higher_is_better(Metric metric) {
    return metric != Metric::Euclidean;
}

inline float dot_product(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float squared_euclidean_distance(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline float euclidean_distance(const float* a, const float* b, std::size_t dim) {
    return std::sqrt(squared_euclidean_distance(a, b, dim));
}

inline float vector_norm(const float* a, std::size_t dim) {
    return std::sqrt(dot_product(a, a, dim));
}

// Cosine similarity in [-1, 1]. Vectors with (near) zero norm are treated as
// having zero similarity to anything rather than dividing by zero.
inline float cosine_similarity(const float* a, const float* b, std::size_t dim) {
    const float denom = vector_norm(a, dim) * vector_norm(b, dim);
    if (denom < 1e-12f) {
        return 0.0f;
    }
    return dot_product(a, b, dim) / denom;
}

// Computes the comparison score for a metric such that, for every metric,
// a larger value from higher_is_better() combined with this score always
// means "closer neighbor". Euclidean is returned as a negated distance so
// every metric can be sorted the same way (descending) by callers that want
// a single code path; callers that need the true distance/similarity value
// should call the specific function above instead.
inline float ranking_score(Metric metric, const float* a, const float* b, std::size_t dim) {
    switch (metric) {
        case Metric::Cosine:
            return cosine_similarity(a, b, dim);
        case Metric::DotProduct:
            return dot_product(a, b, dim);
        case Metric::Euclidean:
            return -euclidean_distance(a, b, dim);
    }
    throw std::invalid_argument("unhandled metric");
}

}  // namespace vdb
