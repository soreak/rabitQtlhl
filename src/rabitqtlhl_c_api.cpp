#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "rabitqlib/index/tree_rabitq_graph/tree_rabitq_graph.hpp"

#if defined(_WIN32)
#define RABITQTLHL_API __declspec(dllexport)
#else
#define RABITQTLHL_API __attribute__((visibility("default")))
#endif

extern "C" {

struct RabitqTlhlParams {
    std::size_t n_centers;
    std::size_t center_leaf_min_size;
    std::size_t center_scan_keep;
    std::size_t exact_center_keep;
    std::size_t center_refine_neighbor_scan;
    int center_entry_mode;

    std::size_t center_real_pool_size;
    std::size_t center_real_pool_take;
    std::size_t center_real_pool_trigger_topk;

    std::size_t center_topn_scan;
    std::size_t center_topn_probe;
    std::size_t center_topn_coarse_keep;
    std::size_t init_keep;

    int use_center_anchor_seeds;
    std::size_t center_anchor_pool_size;
    std::size_t center_anchor_take;
    std::size_t center_anchor_neighbor_centers;
    std::size_t center_anchor_center_keep;

    int use_micro_entry_seeds;
    std::size_t micro_entry_bits;
    std::size_t micro_entry_bucket_take;
    std::size_t micro_entry_probe;
    std::size_t micro_entry_center_keep;

    std::size_t graph_degree;
    std::size_t ef_search;
    std::size_t graph_build_intra_candidates;
    std::size_t graph_build_cross_candidates;
    std::size_t graph_build_projection_dims;
    std::size_t graph_build_center_neighbors;
    int graph_search_use_rabitq;
    int graph_search_full_rabitq;
    std::size_t graph_rerank_candidates;
    int graph_early_stop;
    std::size_t graph_search_neighbor_cap;
    int graph_lazy_center_distance;
    int graph_distance_use_norm_dot;

    int graph_build_bridge_edges;
    std::size_t graph_bridge_center_neighbors;
    std::size_t graph_bridge_points_per_center;
    std::size_t graph_bridge_candidate_scan;
    int graph_query_adjacency_order;
    int graph_reorder_by_center;

    std::size_t rabitq_total_bits;
    std::size_t random_seed;
};

struct RabitqTlhlHandle;

RABITQTLHL_API RabitqTlhlParams rabitqtlhl_default_params();
RABITQTLHL_API RabitqTlhlHandle* rabitqtlhl_create(const RabitqTlhlParams* params);
RABITQTLHL_API void rabitqtlhl_free(RabitqTlhlHandle* handle);
RABITQTLHL_API int rabitqtlhl_build(
    RabitqTlhlHandle* handle,
    const float* data,
    std::size_t rows,
    std::size_t dim
);
RABITQTLHL_API int rabitqtlhl_set_query_params(
    RabitqTlhlHandle* handle,
    std::size_t ef_search,
    std::size_t graph_search_neighbor_cap,
    std::size_t graph_rerank_candidates
);
RABITQTLHL_API int rabitqtlhl_search(
    RabitqTlhlHandle* handle,
    const float* query,
    std::size_t k,
    std::uint32_t* out_ids
);
RABITQTLHL_API int rabitqtlhl_search_many(
    RabitqTlhlHandle* handle,
    const float* queries,
    std::size_t query_count,
    std::size_t k,
    std::uint32_t* out_ids
);
RABITQTLHL_API std::size_t rabitqtlhl_graph_edges(const RabitqTlhlHandle* handle);
RABITQTLHL_API double rabitqtlhl_avg_graph_degree(const RabitqTlhlHandle* handle);
RABITQTLHL_API const char* rabitqtlhl_last_error(const RabitqTlhlHandle* handle);

}  // extern "C"

namespace {

namespace tg = rabitqlib::treegraph;

tg::TreeRaBitQGraphConfig to_config(const RabitqTlhlParams& p) {
    tg::TreeRaBitQGraphConfig c;
    c.n_centers = p.n_centers;
    c.center_leaf_min_size = p.center_leaf_min_size;
    c.center_scan_keep = p.center_scan_keep;
    c.exact_center_keep = p.exact_center_keep;
    c.center_refine_neighbor_scan = p.center_refine_neighbor_scan;
    c.center_entry_mode = static_cast<tg::CenterEntryMode>(
        std::max(0, std::min(2, p.center_entry_mode))
    );

    c.center_real_pool_size = p.center_real_pool_size;
    c.center_real_pool_take = p.center_real_pool_take;
    c.center_real_pool_trigger_topk = p.center_real_pool_trigger_topk;

    c.center_topn_scan = p.center_topn_scan;
    c.center_topn_probe = p.center_topn_probe;
    c.center_topn_coarse_keep = p.center_topn_coarse_keep;
    c.init_keep = p.init_keep;

    c.use_center_anchor_seeds = p.use_center_anchor_seeds != 0;
    c.center_anchor_pool_size = p.center_anchor_pool_size;
    c.center_anchor_take = p.center_anchor_take;
    c.center_anchor_neighbor_centers = p.center_anchor_neighbor_centers;
    c.center_anchor_center_keep = p.center_anchor_center_keep;

    c.use_micro_entry_seeds = p.use_micro_entry_seeds != 0;
    c.micro_entry_bits = p.micro_entry_bits;
    c.micro_entry_bucket_take = p.micro_entry_bucket_take;
    c.micro_entry_probe = p.micro_entry_probe;
    c.micro_entry_center_keep = p.micro_entry_center_keep;

    c.graph_degree = p.graph_degree;
    c.ef_search = p.ef_search;
    c.graph_build_intra_candidates = p.graph_build_intra_candidates;
    c.graph_build_cross_candidates = p.graph_build_cross_candidates;
    c.graph_build_projection_dims = p.graph_build_projection_dims;
    c.graph_build_center_neighbors = p.graph_build_center_neighbors;
    c.graph_search_use_rabitq = p.graph_search_use_rabitq != 0;
    c.graph_search_full_rabitq = p.graph_search_full_rabitq != 0;
    c.graph_rerank_candidates = p.graph_rerank_candidates;
    c.graph_early_stop = p.graph_early_stop != 0;
    c.graph_search_neighbor_cap = p.graph_search_neighbor_cap;
    c.graph_lazy_center_distance = p.graph_lazy_center_distance != 0;
    c.graph_distance_use_norm_dot = p.graph_distance_use_norm_dot != 0;

    c.graph_build_bridge_edges = p.graph_build_bridge_edges != 0;
    c.graph_bridge_center_neighbors = p.graph_bridge_center_neighbors;
    c.graph_bridge_points_per_center = p.graph_bridge_points_per_center;
    c.graph_bridge_candidate_scan = p.graph_bridge_candidate_scan;
    c.graph_query_adjacency_order = p.graph_query_adjacency_order != 0;
    c.graph_reorder_by_center = p.graph_reorder_by_center != 0;

    c.rabitq_total_bits = p.rabitq_total_bits;
    c.random_seed = p.random_seed;
    return c;
}

int fail(RabitqTlhlHandle* handle, const char* message);
int fail(RabitqTlhlHandle* handle, const std::exception& error);

}  // namespace

struct RabitqTlhlHandle {
    RabitqTlhlParams params{};
    std::unique_ptr<tg::TreeRaBitQGraphIndex> index;
    std::size_t dim = 0;
    std::string last_error;
};

namespace {

int fail(RabitqTlhlHandle* handle, const char* message) {
    if (handle != nullptr) {
        handle->last_error = message;
    }
    return -1;
}

int fail(RabitqTlhlHandle* handle, const std::exception& error) {
    return fail(handle, error.what());
}

}  // namespace

extern "C" {

RABITQTLHL_API RabitqTlhlParams rabitqtlhl_default_params() {
    RabitqTlhlParams p{};
    p.n_centers = 2048;
    p.center_leaf_min_size = 16;
    p.center_scan_keep = 32;
    p.exact_center_keep = 6;
    p.center_refine_neighbor_scan = 768;
    p.center_entry_mode = 2;

    p.center_real_pool_size = 256;
    p.center_real_pool_take = 48;
    p.center_real_pool_trigger_topk = 100;

    p.center_topn_scan = 8000;
    p.center_topn_probe = 0;
    p.center_topn_coarse_keep = 64;
    p.init_keep = 32;

    p.use_center_anchor_seeds = 0;
    p.center_anchor_pool_size = 256;
    p.center_anchor_take = 16;
    p.center_anchor_neighbor_centers = 0;
    p.center_anchor_center_keep = 1;

    p.use_micro_entry_seeds = 1;
    p.micro_entry_bits = 6;
    p.micro_entry_bucket_take = 8;
    p.micro_entry_probe = 8;
    p.micro_entry_center_keep = 1;

    p.graph_degree = 48;
    p.ef_search = 56;
    p.graph_build_intra_candidates = 384;
    p.graph_build_cross_candidates = 256;
    p.graph_build_projection_dims = 6;
    p.graph_build_center_neighbors = 8;
    p.graph_search_use_rabitq = 0;
    p.graph_search_full_rabitq = 0;
    p.graph_rerank_candidates = 64;
    p.graph_early_stop = 0;
    p.graph_search_neighbor_cap = 24;
    p.graph_lazy_center_distance = 0;
    p.graph_distance_use_norm_dot = 0;

    p.graph_build_bridge_edges = 1;
    p.graph_bridge_center_neighbors = 2;
    p.graph_bridge_points_per_center = 2;
    p.graph_bridge_candidate_scan = 64;
    p.graph_query_adjacency_order = 1;
    p.graph_reorder_by_center = 1;

    p.rabitq_total_bits = 4;
    p.random_seed = 42;
    return p;
}

RABITQTLHL_API RabitqTlhlHandle* rabitqtlhl_create(const RabitqTlhlParams* params) {
    try {
        auto* handle = new RabitqTlhlHandle();
        handle->params = params == nullptr ? rabitqtlhl_default_params() : *params;
        return handle;
    } catch (...) {
        return nullptr;
    }
}

RABITQTLHL_API void rabitqtlhl_free(RabitqTlhlHandle* handle) {
    delete handle;
}

RABITQTLHL_API int rabitqtlhl_build(
    RabitqTlhlHandle* handle,
    const float* data,
    std::size_t rows,
    std::size_t dim
) {
    if (handle == nullptr) {
        return -1;
    }
    if (data == nullptr || rows == 0 || dim == 0) {
        return fail(handle, "rabitqtlhl_build received empty data");
    }
    try {
        auto config = to_config(handle->params);
        handle->index = std::make_unique<tg::TreeRaBitQGraphIndex>(config);
        handle->index->construct(data, rows, dim);
        handle->dim = dim;
        handle->last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        handle->index.reset();
        return fail(handle, error);
    }
}

RABITQTLHL_API int rabitqtlhl_set_query_params(
    RabitqTlhlHandle* handle,
    std::size_t ef_search,
    std::size_t graph_search_neighbor_cap,
    std::size_t graph_rerank_candidates
) {
    if (handle == nullptr) {
        return -1;
    }
    handle->params.ef_search = ef_search;
    handle->params.graph_search_neighbor_cap = graph_search_neighbor_cap;
    handle->params.graph_rerank_candidates = graph_rerank_candidates;
    if (handle->index != nullptr) {
        handle->index->set_search_params(
            ef_search, graph_rerank_candidates, graph_search_neighbor_cap
        );
    }
    return 0;
}

RABITQTLHL_API int rabitqtlhl_search(
    RabitqTlhlHandle* handle,
    const float* query,
    std::size_t k,
    std::uint32_t* out_ids
) {
    if (handle == nullptr || handle->index == nullptr) {
        return fail(handle, "index is not built");
    }
    if (query == nullptr || out_ids == nullptr || k == 0) {
        return fail(handle, "rabitqtlhl_search received invalid arguments");
    }
    try {
        std::vector<rabitqlib::PID> ids = handle->index->search(query, k);
        if (ids.size() < k) {
            return fail(handle, "search returned fewer ids than requested");
        }
        for (std::size_t i = 0; i < k; ++i) {
            out_ids[i] = ids[i];
        }
        return 0;
    } catch (const std::exception& error) {
        return fail(handle, error);
    }
}

RABITQTLHL_API int rabitqtlhl_search_many(
    RabitqTlhlHandle* handle,
    const float* queries,
    std::size_t query_count,
    std::size_t k,
    std::uint32_t* out_ids
) {
    if (handle == nullptr || handle->index == nullptr) {
        return fail(handle, "index is not built");
    }
    if (queries == nullptr || out_ids == nullptr || query_count == 0 || k == 0) {
        return fail(handle, "rabitqtlhl_search_many received invalid arguments");
    }
    try {
        for (std::size_t qi = 0; qi < query_count; ++qi) {
            const float* query = queries + (qi * handle->dim);
            int code = rabitqtlhl_search(handle, query, k, out_ids + (qi * k));
            if (code != 0) {
                return code;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        return fail(handle, error);
    }
}

RABITQTLHL_API std::size_t rabitqtlhl_graph_edges(const RabitqTlhlHandle* handle) {
    if (handle == nullptr || handle->index == nullptr) {
        return 0;
    }
    return handle->index->graph_edges();
}

RABITQTLHL_API double rabitqtlhl_avg_graph_degree(const RabitqTlhlHandle* handle) {
    if (handle == nullptr || handle->index == nullptr) {
        return 0.0;
    }
    return handle->index->avg_graph_degree();
}

RABITQTLHL_API const char* rabitqtlhl_last_error(const RabitqTlhlHandle* handle) {
    if (handle == nullptr) {
        return "null handle";
    }
    return handle->last_error.c_str();
}

}  // extern "C"
