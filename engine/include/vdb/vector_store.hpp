#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vdb {

// Row-major, contiguous storage for fixed-dimension float vectors, keyed by
// a caller-supplied 64-bit id. Contiguous storage keeps a full brute-force
// scan cache-friendly instead of chasing pointers through per-vector heap
// allocations.
//
// Removal is O(1): the last row is swapped into the removed row's slot
// rather than leaving a tombstone, so the store never grows sparse and a
// full scan never wastes time skipping deleted rows.
class FlatVectorStore {
public:
    explicit FlatVectorStore(std::size_t dim) : dim_(dim) {
        if (dim_ == 0) {
            throw std::invalid_argument("vector dimension must be positive");
        }
    }

    std::size_t dim() const { return dim_; }
    std::size_t size() const { return ids_.size(); }

    // Bumped on every upsert/remove. Approximate indexes built over a
    // snapshot of this store (see IVFIndex) record the version at build
    // time so a search against a mutated store can be rejected instead of
    // silently scanning stale row indices.
    std::uint64_t version() const { return version_; }

    bool contains(std::uint64_t id) const {
        return id_to_row_.find(id) != id_to_row_.end();
    }

    // Inserts a new vector, or overwrites the existing one for `id`.
    void upsert(std::uint64_t id, const float* vec) {
        auto it = id_to_row_.find(id);
        if (it != id_to_row_.end()) {
            std::copy(vec, vec + dim_, data_.begin() + static_cast<std::ptrdiff_t>(it->second * dim_));
            ++version_;
            return;
        }
        const std::size_t row = ids_.size();
        id_to_row_.emplace(id, row);
        ids_.push_back(id);
        data_.insert(data_.end(), vec, vec + dim_);
        ++version_;
    }

    // Removes a vector by id. Returns false if the id was not present.
    bool remove(std::uint64_t id) {
        auto it = id_to_row_.find(id);
        if (it == id_to_row_.end()) {
            return false;
        }
        const std::size_t row = it->second;
        const std::size_t last_row = ids_.size() - 1;

        if (row != last_row) {
            std::copy(
                data_.begin() + static_cast<std::ptrdiff_t>(last_row * dim_),
                data_.begin() + static_cast<std::ptrdiff_t>((last_row + 1) * dim_),
                data_.begin() + static_cast<std::ptrdiff_t>(row * dim_));
            ids_[row] = ids_[last_row];
            id_to_row_[ids_[row]] = row;
        }

        ids_.pop_back();
        data_.resize(last_row * dim_);
        id_to_row_.erase(it);
        ++version_;
        return true;
    }

    const float* row(std::size_t row_index) const {
        return data_.data() + row_index * dim_;
    }

    const float* vector_for_id(std::uint64_t id) const {
        auto it = id_to_row_.find(id);
        if (it == id_to_row_.end()) {
            throw std::out_of_range("id not found in vector store");
        }
        return row(it->second);
    }

    const std::vector<std::uint64_t>& ids() const { return ids_; }
    const std::vector<float>& raw_data() const { return data_; }

private:
    std::size_t dim_;
    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
    std::unordered_map<std::uint64_t, std::size_t> id_to_row_;
    std::uint64_t version_ = 0;
};

}  // namespace vdb
