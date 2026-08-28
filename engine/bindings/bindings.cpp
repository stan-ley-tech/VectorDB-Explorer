#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vdb/distance.hpp"
#include "vdb/ivf_index.hpp"
#include "vdb/search.hpp"
#include "vdb/vector_store.hpp"

namespace py = pybind11;
using namespace vdb;

namespace {

void check_dimension(const std::vector<float>& vec, std::size_t expected, const char* what) {
    if (vec.size() != expected) {
        throw py::value_error(
            std::string(what) + " has dimension " + std::to_string(vec.size()) +
            ", expected " + std::to_string(expected));
    }
}

std::vector<float> vector_for_id_py(const FlatVectorStore& store, std::uint64_t id) {
    const float* row = store.vector_for_id(id);
    return std::vector<float>(row, row + store.dim());
}

void upsert_py(FlatVectorStore& store, std::uint64_t id, const std::vector<float>& vec) {
    check_dimension(vec, store.dim(), "vector");
    store.upsert(id, vec.data());
}

std::vector<SearchResult> brute_force_search_py(
    const FlatVectorStore& store, const std::vector<float>& query, std::size_t top_k, Metric metric) {
    check_dimension(query, store.dim(), "query");
    return brute_force_search(store, query.data(), top_k, metric);
}

std::vector<SearchResult> ivf_search_py(
    const IVFIndex& index,
    const FlatVectorStore& store,
    const std::vector<float>& query,
    std::size_t top_k,
    std::size_t nprobe) {
    check_dimension(query, index.dim(), "query");
    return index.search(store, query.data(), top_k, nprobe);
}

}  // namespace

PYBIND11_MODULE(vdb_native, m) {
    m.doc() = "Native C++ vector search engine bindings for VectorDB Explorer";

    py::enum_<Metric>(m, "Metric")
        .value("COSINE", Metric::Cosine)
        .value("EUCLIDEAN", Metric::Euclidean)
        .value("DOT_PRODUCT", Metric::DotProduct)
        .export_values();

    m.def("metric_from_string", &metric_from_string, py::arg("name"));
    m.def("metric_to_string", &metric_to_string, py::arg("metric"));

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("id", &SearchResult::id)
        .def_readonly("score", &SearchResult::score)
        .def("__repr__", [](const SearchResult& r) {
            return "SearchResult(id=" + std::to_string(r.id) + ", score=" + std::to_string(r.score) + ")";
        });

    py::class_<FlatVectorStore>(m, "VectorStore")
        .def(py::init<std::size_t>(), py::arg("dim"))
        .def("upsert", &upsert_py, py::arg("id"), py::arg("vector"))
        .def("remove", &FlatVectorStore::remove, py::arg("id"))
        .def("contains", &FlatVectorStore::contains, py::arg("id"))
        .def("vector_for_id", &vector_for_id_py, py::arg("id"))
        .def("size", &FlatVectorStore::size)
        .def("dim", &FlatVectorStore::dim)
        .def("version", &FlatVectorStore::version)
        .def("memory_usage_bytes", &FlatVectorStore::memory_usage_bytes)
        .def("__len__", &FlatVectorStore::size);

    m.def(
        "brute_force_search",
        &brute_force_search_py,
        py::arg("store"),
        py::arg("query"),
        py::arg("top_k"),
        py::arg("metric"),
        "Exhaustive nearest-neighbor search over every vector in the store.");

    py::class_<IVFIndex>(m, "IVFIndex")
        .def(py::init<std::size_t, Metric, std::size_t>(), py::arg("dim"), py::arg("metric"), py::arg("nlist"))
        .def("build", &IVFIndex::build, py::arg("store"), py::arg("kmeans_iterations") = 15, py::arg("seed") = 42)
        .def("search", &ivf_search_py, py::arg("store"), py::arg("query"), py::arg("top_k"), py::arg("nprobe"))
        .def("is_built", &IVFIndex::is_built)
        .def("nlist", &IVFIndex::nlist)
        .def("dim", &IVFIndex::dim)
        .def("metric", &IVFIndex::metric)
        .def("cluster_sizes", &IVFIndex::cluster_sizes);
}
