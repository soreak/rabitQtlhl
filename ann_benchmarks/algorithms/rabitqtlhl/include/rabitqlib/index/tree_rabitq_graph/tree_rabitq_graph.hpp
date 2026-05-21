#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::treegraph {

enum class CenterEntryMode : uint8_t {
    TreeThenRabitq = 0,
    TreeOnly = 1,
    RabitqOnly = 2
};

struct TreeRaBitQGraphConfig {
    size_t n_centers = 128;
    size_t center_leaf_min_size = 16;

    size_t center_scan_keep = 16;
    size_t exact_center_keep = 6;
    size_t center_refine_neighbor_scan = 0;
    CenterEntryMode center_entry_mode = CenterEntryMode::TreeThenRabitq;

    size_t center_real_pool_size = 256;
    size_t center_real_pool_take = 48;
    size_t center_real_pool_trigger_topk = 100;

    size_t center_topn_scan = 2000;
    size_t center_topn_probe = 0;
    size_t center_topn_coarse_keep = 96;
    size_t init_keep = 32;
    bool use_center_anchor_seeds = false;
    size_t center_anchor_pool_size = 256;
    size_t center_anchor_take = 16;
    size_t center_anchor_neighbor_centers = 0;
    size_t center_anchor_center_keep = 1;
    bool use_micro_entry_seeds = false;
    size_t micro_entry_bits = 6;
    size_t micro_entry_bucket_take = 8;
    size_t micro_entry_probe = 8;
    size_t micro_entry_center_keep = 1;

    size_t graph_degree = 48;
    size_t ef_search = 64;
    size_t graph_build_intra_candidates = 384;
    size_t graph_build_cross_candidates = 256;
    size_t graph_build_projection_dims = 6;
    size_t graph_build_center_neighbors = 8;
    bool graph_search_use_rabitq = true;
    bool graph_search_full_rabitq = false;
    size_t graph_rerank_candidates = 128;
    bool graph_early_stop = false;
    size_t graph_search_neighbor_cap = 0;
    bool graph_lazy_center_distance = false;
    bool graph_distance_use_norm_dot = false;
    bool graph_build_bridge_edges = false;
    size_t graph_bridge_center_neighbors = 2;
    size_t graph_bridge_points_per_center = 2;
    size_t graph_bridge_candidate_scan = 64;
    bool graph_query_adjacency_order = false;
    bool graph_reorder_by_center = false;

    size_t rabitq_total_bits = 4;
    size_t random_seed = 42;
};

enum class SeedMode : uint8_t {
    CenterRealPool,
    RabitqTopN,
    CenterAnchor,
    MicroEntry,
    Fallback
};

struct QueryStats {
    size_t entry_points_raw = 0;
    size_t entry_points_before_cap = 0;
    size_t entry_points_after_cap = 0;
    size_t init_size = 0;
    size_t seed_pushes = 0;
    size_t heap_pushes = 0;
    size_t heap_pops = 0;
    size_t visited_nodes = 0;
    size_t node_expansions = 0;
    size_t edges_scanned = 0;
    size_t final_candidates = 0;
    size_t topk = 0;
    size_t jumps = 0;
    size_t rabitq_center_est_evals = 0;
    size_t rabitq_topn_coarse_evals = 0;
    size_t rabitq_topn_refine_evals = 0;
    size_t graph_est_evals = 0;
    size_t exact_distance_evals = 0;
    SeedMode init_mode = SeedMode::Fallback;
    bool trigger_pass = false;
    bool fallback = false;

    // Legacy fields kept for older sample/debug output while the prototype evolves.
    size_t visited = 0;
    size_t center_estimates = 0;
    size_t center_exact_scores = 0;
    size_t topn_coarse_estimates = 0;
    size_t topn_full_estimates = 0;
    size_t graph_center_exact_scores = 0;
    size_t graph_distance_evals = 0;
    size_t exact_rerank_scores = 0;
    size_t graph_early_stop_count = 0;
    size_t fallback_count = 0;
    size_t trigger_pass_count = 0;
    SeedMode mode = SeedMode::Fallback;
    PID routed_center = 0;
    PID best_center = 0;

    double query_total_ms = 0.0;
    double query_prepare_ms = 0.0;
    double route_ms = 0.0;
    double center_refine_ms = 0.0;
    double seed_select_ms = 0.0;
    double graph_prepare_ms = 0.0;
    double graph_search_ms = 0.0;
    double final_select_ms = 0.0;
};

using TreeRaBitQGraphQueryStats = QueryStats;

struct StatSummary {
    size_t sum = 0;
    size_t max = 0;
    size_t count = 0;

    void add(size_t value) {
        sum += value;
        max = std::max(max, value);
        ++count;
    }

    [[nodiscard]] double avg() const {
        return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count);
    }
};

struct TimeSummary {
    double sum = 0.0;
    double max = 0.0;
    size_t count = 0;

    void add(double value) {
        sum += value;
        max = std::max(max, value);
        ++count;
    }

    [[nodiscard]] double avg() const {
        return count == 0 ? 0.0 : sum / static_cast<double>(count);
    }
};

struct BatchStats {
    StatSummary entry_points_raw;
    StatSummary entry_points_before_cap;
    StatSummary entry_points_after_cap;
    StatSummary init_sizes;
    StatSummary seed_pushes;
    StatSummary heap_pushes;
    StatSummary heap_pops;
    StatSummary visited_nodes;
    StatSummary node_expansions;
    StatSummary edges_scanned;
    StatSummary final_candidates;
    StatSummary topk;
    StatSummary jumps;
    StatSummary rabitq_center_est_evals;
    StatSummary rabitq_topn_coarse_evals;
    StatSummary rabitq_topn_refine_evals;
    StatSummary graph_est_evals;
    StatSummary exact_distance_evals;
    StatSummary distance_evals;
    StatSummary graph_center_exact_scores;
    StatSummary graph_distance_evals;
    StatSummary exact_rerank_scores;
    StatSummary graph_early_stop_count;
    StatSummary trigger_pass;
    StatSummary fallback;
    TimeSummary query_total_ms;
    TimeSummary query_prepare_ms;
    TimeSummary route_ms;
    TimeSummary center_refine_ms;
    TimeSummary seed_select_ms;
    TimeSummary graph_prepare_ms;
    TimeSummary graph_search_ms;
    TimeSummary final_select_ms;
    size_t fallback_count = 0;
    size_t center_real_pool = 0;
    size_t rabitq_topn = 0;
    size_t center_anchor = 0;
    size_t micro_entry = 0;
    size_t trigger_pass_count = 0;

    void add(const QueryStats& stats) {
        entry_points_raw.add(stats.entry_points_raw);
        entry_points_before_cap.add(stats.entry_points_before_cap);
        entry_points_after_cap.add(stats.entry_points_after_cap);
        init_sizes.add(stats.init_size);
        seed_pushes.add(stats.seed_pushes);
        heap_pushes.add(stats.heap_pushes);
        heap_pops.add(stats.heap_pops);
        visited_nodes.add(stats.visited_nodes);
        node_expansions.add(stats.node_expansions);
        edges_scanned.add(stats.edges_scanned);
        final_candidates.add(stats.final_candidates);
        topk.add(stats.topk);
        jumps.add(stats.jumps);
        rabitq_center_est_evals.add(stats.rabitq_center_est_evals);
        rabitq_topn_coarse_evals.add(stats.rabitq_topn_coarse_evals);
        rabitq_topn_refine_evals.add(stats.rabitq_topn_refine_evals);
        graph_est_evals.add(stats.graph_est_evals);
        exact_distance_evals.add(stats.exact_distance_evals);
        distance_evals.add(stats.exact_distance_evals);
        graph_center_exact_scores.add(stats.graph_center_exact_scores);
        graph_distance_evals.add(stats.graph_distance_evals);
        exact_rerank_scores.add(stats.exact_rerank_scores);
        graph_early_stop_count.add(stats.graph_early_stop_count);
        trigger_pass.add(stats.trigger_pass ? 1 : 0);
        fallback.add(stats.fallback ? 1 : 0);
        query_total_ms.add(stats.query_total_ms);
        query_prepare_ms.add(stats.query_prepare_ms);
        route_ms.add(stats.route_ms);
        center_refine_ms.add(stats.center_refine_ms);
        seed_select_ms.add(stats.seed_select_ms);
        graph_prepare_ms.add(stats.graph_prepare_ms);
        graph_search_ms.add(stats.graph_search_ms);
        final_select_ms.add(stats.final_select_ms);
        fallback_count += stats.fallback ? 1 : stats.fallback_count;
        trigger_pass_count += stats.trigger_pass ? 1 : stats.trigger_pass_count;
        if (stats.init_mode == SeedMode::CenterRealPool) {
            ++center_real_pool;
        } else if (stats.init_mode == SeedMode::RabitqTopN) {
            ++rabitq_topn;
        } else if (stats.init_mode == SeedMode::CenterAnchor) {
            ++center_anchor;
        } else if (stats.init_mode == SeedMode::MicroEntry) {
            ++micro_entry;
        }
    }
};

class TreeRaBitQGraphIndex {
   public:
    explicit TreeRaBitQGraphIndex(
        TreeRaBitQGraphConfig config = TreeRaBitQGraphConfig(),
        RotatorType rotator_type = RotatorType::FhtKacRotator
    )
        : config_(config), rotator_type_(rotator_type) {
        if (config_.rabitq_total_bits < 1 || config_.rabitq_total_bits > 9) {
            throw std::invalid_argument("rabitq_total_bits must be in [1, 9]");
        }
        config_.n_centers = std::max<size_t>(1, config_.n_centers);
        config_.center_leaf_min_size =
            std::max<size_t>(1, config_.center_leaf_min_size);
        config_.graph_degree = std::max<size_t>(1, config_.graph_degree);
        config_.ef_search = std::max<size_t>(1, config_.ef_search);
        config_.center_topn_coarse_keep =
            std::max<size_t>(config_.center_topn_coarse_keep, config_.init_keep);
        config_.center_anchor_pool_size =
            std::max<size_t>(1, config_.center_anchor_pool_size);
        config_.center_anchor_take = std::max<size_t>(1, config_.center_anchor_take);
        config_.center_anchor_center_keep =
            std::max<size_t>(1, config_.center_anchor_center_keep);
        config_.micro_entry_bits = std::min<size_t>(
            std::max<size_t>(1, config_.micro_entry_bits), sizeof(size_t) * 8 - 1
        );
        config_.micro_entry_bucket_take =
            std::max<size_t>(1, config_.micro_entry_bucket_take);
        config_.micro_entry_probe = std::max<size_t>(1, config_.micro_entry_probe);
        config_.micro_entry_center_keep =
            std::max<size_t>(1, config_.micro_entry_center_keep);
        config_.graph_build_intra_candidates =
            std::max(config_.graph_build_intra_candidates, config_.graph_degree);
        config_.graph_build_cross_candidates =
            std::max<size_t>(1, config_.graph_build_cross_candidates);
        config_.graph_build_projection_dims =
            std::max<size_t>(1, config_.graph_build_projection_dims);
        config_.graph_build_center_neighbors =
            std::max<size_t>(1, config_.graph_build_center_neighbors);
        config_.graph_rerank_candidates =
            std::max<size_t>(1, config_.graph_rerank_candidates);
        config_.graph_search_neighbor_cap =
            std::min(config_.graph_search_neighbor_cap, config_.graph_degree);
        config_.graph_bridge_center_neighbors =
            std::max<size_t>(1, config_.graph_bridge_center_neighbors);
        config_.graph_bridge_points_per_center =
            std::max<size_t>(1, config_.graph_bridge_points_per_center);
        config_.graph_bridge_candidate_scan =
            std::max<size_t>(1, config_.graph_bridge_candidate_scan);
    }

    void construct(const float* base, size_t num, size_t dim);

    std::vector<PID> search(
        const float* query, size_t k, TreeRaBitQGraphQueryStats* stats = nullptr
    ) const;

    void set_search_params(
        size_t ef_search,
        size_t graph_rerank_candidates,
        size_t graph_search_neighbor_cap
    ) {
        config_.ef_search = std::max<size_t>(1, ef_search);
        config_.graph_rerank_candidates =
            std::max<size_t>(1, graph_rerank_candidates);
        config_.graph_search_neighbor_cap =
            std::min(graph_search_neighbor_cap, config_.graph_degree);
    }

    [[nodiscard]] const TreeRaBitQGraphConfig& config() const { return config_; }
    [[nodiscard]] size_t num_points() const { return num_; }
    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
    [[nodiscard]] size_t num_centers() const { return centers_.size() / dim_; }
    [[nodiscard]] size_t graph_edges() const { return graph_indices_.size(); }
    [[nodiscard]] size_t graph_bridge_edges_added() const {
        return graph_bridge_edges_added_;
    }
    [[nodiscard]] bool graph_reordered_by_center() const {
        return graph_reordered_by_center_;
    }
    [[nodiscard]] double avg_graph_degree() const {
        return num_ == 0 ? 0.0
                         : static_cast<double>(graph_indices_.size()) /
                               static_cast<double>(num_);
    }

   private:
    struct TreeNode {
        bool leaf = false;
        size_t left = 0;
        size_t right = 0;
        size_t split_dim = 0;
        float split_value = 0;
        PID center = 0;
    };

    struct CodeBank {
        size_t count = 0;
        size_t bin_bytes = 0;
        size_t ex_bytes = 0;
        std::vector<uint64_t> bin_storage;
        std::vector<uint64_t> ex_storage;

        static size_t words_for(size_t bytes) {
            return div_round_up(bytes, sizeof(uint64_t));
        }

        void reset(size_t new_count, size_t new_bin_bytes, size_t new_ex_bytes) {
            count = new_count;
            bin_bytes = new_bin_bytes;
            ex_bytes = new_ex_bytes;
            bin_storage.assign(words_for(count * bin_bytes), 0);
            ex_storage.assign(words_for(count * ex_bytes), 0);
        }

        char* bin(size_t i) {
            return reinterpret_cast<char*>(bin_storage.data()) + (i * bin_bytes);
        }

        const char* bin(size_t i) const {
            return reinterpret_cast<const char*>(bin_storage.data()) + (i * bin_bytes);
        }

        char* ex(size_t i) {
            if (ex_bytes == 0) {
                return nullptr;
            }
            return reinterpret_cast<char*>(ex_storage.data()) + (i * ex_bytes);
        }

        const char* ex(size_t i) const {
            if (ex_bytes == 0) {
                return nullptr;
            }
            return reinterpret_cast<const char*>(ex_storage.data()) + (i * ex_bytes);
        }
    };

    struct BatchCodeBank {
        size_t count = 0;
        size_t batch_count = 0;
        size_t batch_bytes = 0;
        std::vector<uint64_t> storage;

        static size_t words_for(size_t bytes) {
            return div_round_up(bytes, sizeof(uint64_t));
        }

        void reset(size_t new_count, size_t new_batch_bytes) {
            count = new_count;
            batch_count = div_round_up(count, fastscan::kBatchSize);
            batch_bytes = new_batch_bytes;
            storage.assign(words_for(batch_count * batch_bytes), 0);
        }

        char* batch(size_t i) {
            return reinterpret_cast<char*>(storage.data()) + (i * batch_bytes);
        }

        const char* batch(size_t i) const {
            return reinterpret_cast<const char*>(storage.data()) + (i * batch_bytes);
        }
    };

    struct ScoredPid {
        float distance = std::numeric_limits<float>::max();
        PID id = 0;

        friend bool operator<(const ScoredPid& a, const ScoredPid& b) {
            return a.distance < b.distance;
        }
        friend bool operator>(const ScoredPid& a, const ScoredPid& b) {
            return a.distance > b.distance;
        }
    };

    TreeRaBitQGraphConfig config_;
    RotatorType rotator_type_;
    size_t num_ = 0;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t ex_bits_ = 0;
    size_t root_ = 0;

    std::vector<float> base_;
    std::vector<float> base_sq_norms_;
    std::vector<float> rotated_base_;
    std::vector<float> centers_;
    std::vector<float> rotated_centers_;
    std::vector<float> center_pool_trigger_d2_;
    std::vector<PID> external_ids_;

    std::vector<TreeNode> tree_;
    std::vector<std::vector<PID>> leaf_ids_;
    std::vector<PID> point_center_;

    std::vector<std::vector<PID>> center_real_pool_;
    std::vector<std::vector<PID>> center_topn_;
    std::vector<std::vector<PID>> center_anchor_pool_;
    std::vector<std::vector<PID>> center_neighbors_;
    std::vector<std::vector<PID>> micro_entry_buckets_;
    std::vector<size_t> micro_entry_dims_;
    std::vector<float> micro_entry_centroids_;
    std::vector<uint32_t> micro_entry_counts_;
    size_t micro_entry_bucket_count_ = 0;
    std::vector<CodeBank> topn_codes_;
    std::vector<BatchCodeBank> topn_batch_codes_;
    CodeBank center_codes_;
    CodeBank point_codes_;

    std::vector<std::vector<PID>> graph_;
    std::vector<size_t> graph_offsets_;
    std::vector<PID> graph_indices_;
    size_t graph_bridge_edges_added_ = 0;
    bool graph_reordered_by_center_ = false;
    mutable std::vector<uint32_t> visit_marks_;
    mutable uint32_t visit_epoch_ = 0;

    std::unique_ptr<Rotator<float>> rotator_;
    quant::RabitqConfig index_config_;
    quant::RabitqConfig query_config_;
    ex_ipfunc ip_func_ = nullptr;

    size_t build_tree(std::vector<PID> ids, size_t target_leaves);
    PID make_leaf(const std::vector<PID>& ids);
    size_t choose_split_dim(const std::vector<PID>& ids) const;
    PID route_to_center(const float* query) const;

    void rotate_base_and_centers();
    void build_center_pools();
    void build_center_neighbors();
    void build_micro_entry_pools();
    void quantize_center_codes();
    void quantize_topn_codes();
    void quantize_point_codes();
    void build_graph();
    bool append_graph_edge(PID a, PID b);
    void add_bridge_edges();
    void order_graph_for_query();
    void reorder_graph_by_center();
    void finalize_graph_csr();
    std::vector<PID> search_fast(const float* query, size_t k) const;

    std::vector<PID> initial_seeds(
        const float* query,
        float query_sq_norm,
        const SplitSingleQuery<float>& query_wrapper,
        PID best_center,
        const std::vector<PID>& candidate_centers,
        float best_center_dist2,
        TreeRaBitQGraphQueryStats& stats
    ) const;

    std::vector<PID> initial_seeds_fast(
        const float* query,
        float query_sq_norm,
        const SplitSingleQuery<float>& query_wrapper,
        PID best_center,
        const std::vector<PID>& candidate_centers,
        float best_center_dist2
    ) const;

    PID refine_center(
        const float* query,
        const SplitSingleQuery<float>& query_wrapper,
        PID routed_center,
        bool use_routed_center,
        TreeRaBitQGraphQueryStats& stats,
        std::vector<PID>* exact_centers = nullptr
    ) const;

    PID refine_center_fast(
        const float* query,
        const SplitSingleQuery<float>& query_wrapper,
        PID routed_center,
        bool use_routed_center,
        std::vector<PID>* exact_centers = nullptr
    ) const;

    float estimate_code(
        const char* bin_data,
        const char* ex_data,
        const SplitSingleQuery<float>& query_wrapper,
        float g_add,
        float g_error
    ) const;

    float estimate_code_onebit(
        const char* bin_data,
        const SplitSingleQuery<float>& query_wrapper,
        float g_add,
        float g_error
    ) const;

    uint32_t next_visit_epoch() const;

    static float squared_norm(const float* data, size_t dim) {
        float sum = 0.0F;
        for (size_t i = 0; i < dim; ++i) {
            sum += data[i] * data[i];
        }
        return sum;
    }

    static float dot_product(const float* a, const float* b, size_t dim) {
        float sum = 0.0F;
        for (size_t i = 0; i < dim; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static float euclidean_sqr_fast(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
        if (dim == 128) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (size_t i = 0; i < 128; i += 16) {
                __m256 av0 = _mm256_loadu_ps(a + i);
                __m256 bv0 = _mm256_loadu_ps(b + i);
                __m256 diff0 = _mm256_sub_ps(av0, bv0);
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff0, diff0));

                __m256 av1 = _mm256_loadu_ps(a + i + 8);
                __m256 bv1 = _mm256_loadu_ps(b + i + 8);
                __m256 diff1 = _mm256_sub_ps(av1, bv1);
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(diff1, diff1));
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }
#else
        (void)a;
        (void)b;
#endif
        return euclidean_sqr<float>(a, b, dim);
    }

    float l2_to_point(const float* query, PID id) const {
        return euclidean_sqr_fast(query, base_.data() + (static_cast<size_t>(id) * dim_), dim_);
    }

    float l2_to_point(const float* query, float query_sq_norm, PID id) const {
        if (!config_.graph_distance_use_norm_dot) {
            return l2_to_point(query, id);
        }
        const size_t offset = static_cast<size_t>(id) * dim_;
        float dist = query_sq_norm + base_sq_norms_[id] -
                     (2.0F * dot_product(query, base_.data() + offset, dim_));
        return std::max(dist, 0.0F);
    }

    float l2_to_center(const float* query, PID center) const {
        return euclidean_sqr_fast(
            query, centers_.data() + (static_cast<size_t>(center) * dim_), dim_
        );
    }

    size_t micro_entry_bucket_offset(PID center, size_t bucket) const {
        return (static_cast<size_t>(center) * micro_entry_bucket_count_) + bucket;
    }

    size_t micro_entry_centroid_offset(PID center, size_t bucket) const {
        return micro_entry_bucket_offset(center, bucket) * dim_;
    }

    size_t micro_entry_bucket_for_point(PID center, PID id) const;
    size_t micro_entry_bucket_for_query(PID center, const float* rotated_query) const;

    float point_point_l2(PID a, PID b) const {
        if (!config_.graph_distance_use_norm_dot) {
            return euclidean_sqr_fast(
                base_.data() + (static_cast<size_t>(a) * dim_),
                base_.data() + (static_cast<size_t>(b) * dim_),
                dim_
            );
        }
        const size_t a_offset = static_cast<size_t>(a) * dim_;
        const size_t b_offset = static_cast<size_t>(b) * dim_;
        float dist = base_sq_norms_[a] + base_sq_norms_[b] -
                     (2.0F * dot_product(
                                 base_.data() + a_offset,
                                 base_.data() + b_offset,
                                 dim_
                             ));
        return std::max(dist, 0.0F);
    }

    PID external_id(PID id) const {
        return external_ids_.empty() ? id : external_ids_[id];
    }

    static void keep_smallest(std::vector<ScoredPid>& items, size_t keep);
    static void push_topk_smallest(
        std::vector<ScoredPid>& heap, ScoredPid item, size_t keep
    );
    static void finish_topk_smallest(std::vector<ScoredPid>& heap);
};

inline void TreeRaBitQGraphIndex::construct(const float* base, size_t num, size_t dim) {
    if (base == nullptr || num == 0 || dim == 0) {
        throw std::invalid_argument("TreeRaBitQGraphIndex::construct got empty data");
    }

    num_ = num;
    dim_ = dim;
    ex_bits_ = config_.rabitq_total_bits - 1;
    base_.assign(base, base + (num_ * dim_));
    external_ids_.resize(num_);
    std::iota(external_ids_.begin(), external_ids_.end(), static_cast<PID>(0));
    base_sq_norms_.clear();
    if (config_.graph_distance_use_norm_dot) {
        base_sq_norms_.resize(num_);
        for (size_t i = 0; i < num_; ++i) {
            base_sq_norms_[i] = squared_norm(base_.data() + (i * dim_), dim_);
        }
    }

    tree_.clear();
    leaf_ids_.clear();
    centers_.clear();
    point_center_.assign(num_, 0);
    graph_offsets_.clear();
    graph_indices_.clear();
    graph_bridge_edges_added_ = 0;
    graph_reordered_by_center_ = false;
    micro_entry_buckets_.clear();
    micro_entry_dims_.clear();
    micro_entry_centroids_.clear();
    micro_entry_counts_.clear();
    micro_entry_bucket_count_ = 0;

    size_t feasible_leaves = std::max<size_t>(1, num_ / config_.center_leaf_min_size);
    size_t target_leaves = std::min(config_.n_centers, feasible_leaves);
    std::vector<PID> all_ids(num_);
    std::iota(all_ids.begin(), all_ids.end(), static_cast<PID>(0));
    root_ = build_tree(std::move(all_ids), target_leaves);

    rotate_base_and_centers();
    index_config_ = quant::faster_config(padded_dim_, config_.rabitq_total_bits);
    query_config_ =
        quant::faster_config(padded_dim_, SplitSingleQuery<float>::kNumBits);
    ip_func_ = select_excode_ipfunc(ex_bits_);

    build_center_pools();
    build_center_neighbors();
    if (config_.use_micro_entry_seeds) {
        build_micro_entry_pools();
    }
    quantize_center_codes();
    if (!config_.use_center_anchor_seeds && !config_.use_micro_entry_seeds &&
        !config_.graph_reorder_by_center) {
        quantize_topn_codes();
    } else {
        topn_codes_.clear();
        topn_batch_codes_.clear();
    }
    if (config_.graph_search_use_rabitq && !config_.graph_reorder_by_center) {
        quantize_point_codes();
    }
    build_graph();
    visit_marks_.assign(num_, 0);
    visit_epoch_ = 0;
}

inline size_t TreeRaBitQGraphIndex::build_tree(
    std::vector<PID> ids, size_t target_leaves
) {
    size_t node_id = tree_.size();
    tree_.push_back(TreeNode());

    size_t feasible_leaves =
        std::max<size_t>(1, ids.size() / config_.center_leaf_min_size);
    target_leaves = std::min(target_leaves, feasible_leaves);

    if (target_leaves <= 1 || ids.size() <= config_.center_leaf_min_size) {
        tree_[node_id].leaf = true;
        tree_[node_id].center = make_leaf(ids);
        return node_id;
    }

    size_t left_target = std::max<size_t>(1, target_leaves / 2);
    size_t right_target = target_leaves - left_target;
    if (right_target == 0) {
        tree_[node_id].leaf = true;
        tree_[node_id].center = make_leaf(ids);
        return node_id;
    }

    size_t split_count = (ids.size() * left_target) / target_leaves;
    size_t left_min = left_target * config_.center_leaf_min_size;
    size_t right_min = right_target * config_.center_leaf_min_size;
    split_count = std::max(split_count, left_min);
    split_count = std::min(split_count, ids.size() - right_min);

    if (split_count == 0 || split_count >= ids.size()) {
        tree_[node_id].leaf = true;
        tree_[node_id].center = make_leaf(ids);
        return node_id;
    }

    size_t split_dim = choose_split_dim(ids);
    auto mid = ids.begin() + static_cast<std::ptrdiff_t>(split_count);
    std::nth_element(
        ids.begin(),
        mid,
        ids.end(),
        [&](PID a, PID b) {
            return base_[static_cast<size_t>(a) * dim_ + split_dim] <
                   base_[static_cast<size_t>(b) * dim_ + split_dim];
        }
    );

    float left_max = -std::numeric_limits<float>::infinity();
    float right_min_value = std::numeric_limits<float>::infinity();
    for (auto it = ids.begin(); it != mid; ++it) {
        left_max = std::max(left_max, base_[static_cast<size_t>(*it) * dim_ + split_dim]);
    }
    for (auto it = mid; it != ids.end(); ++it) {
        right_min_value =
            std::min(right_min_value, base_[static_cast<size_t>(*it) * dim_ + split_dim]);
    }

    std::vector<PID> left(ids.begin(), mid);
    std::vector<PID> right(mid, ids.end());

    tree_[node_id].split_dim = split_dim;
    tree_[node_id].split_value = (left_max + right_min_value) * 0.5F;
    tree_[node_id].left = build_tree(std::move(left), left_target);
    tree_[node_id].right = build_tree(std::move(right), right_target);
    return node_id;
}

inline PID TreeRaBitQGraphIndex::make_leaf(const std::vector<PID>& ids) {
    PID center_id = static_cast<PID>(leaf_ids_.size());
    leaf_ids_.push_back(ids);

    centers_.resize((static_cast<size_t>(center_id) + 1) * dim_, 0);
    float* center = centers_.data() + (static_cast<size_t>(center_id) * dim_);
    for (PID id : ids) {
        point_center_[id] = center_id;
        const float* point = base_.data() + (static_cast<size_t>(id) * dim_);
        for (size_t d = 0; d < dim_; ++d) {
            center[d] += point[d];
        }
    }
    float inv = 1.0F / static_cast<float>(ids.size());
    for (size_t d = 0; d < dim_; ++d) {
        center[d] *= inv;
    }
    return center_id;
}

inline size_t TreeRaBitQGraphIndex::choose_split_dim(const std::vector<PID>& ids) const {
    size_t best_dim = 0;
    double best_var = -1.0;
    for (size_t d = 0; d < dim_; ++d) {
        double sum = 0;
        double sum_sq = 0;
        for (PID id : ids) {
            double v = base_[static_cast<size_t>(id) * dim_ + d];
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / static_cast<double>(ids.size());
        double var = (sum_sq / static_cast<double>(ids.size())) - (mean * mean);
        if (var > best_var) {
            best_var = var;
            best_dim = d;
        }
    }
    return best_dim;
}

inline PID TreeRaBitQGraphIndex::route_to_center(const float* query) const {
    size_t node_id = root_;
    while (!tree_[node_id].leaf) {
        const TreeNode& node = tree_[node_id];
        node_id = (query[node.split_dim] <= node.split_value) ? node.left : node.right;
    }
    return tree_[node_id].center;
}

inline void TreeRaBitQGraphIndex::rotate_base_and_centers() {
    rotator_.reset(choose_rotator<float>(
        dim_, rotator_type_, rotator_impl::padding_requirement(dim_, rotator_type_)
    ));
    padded_dim_ = rotator_->size();

    rotated_base_.assign(num_ * padded_dim_, 0);
    for (size_t i = 0; i < num_; ++i) {
        rotator_->rotate(base_.data() + (i * dim_), rotated_base_.data() + (i * padded_dim_));
    }

    size_t centers_count = num_centers();
    rotated_centers_.assign(centers_count * padded_dim_, 0);
    for (size_t c = 0; c < centers_count; ++c) {
        rotator_->rotate(
            centers_.data() + (c * dim_), rotated_centers_.data() + (c * padded_dim_)
        );
    }
}

inline void TreeRaBitQGraphIndex::build_center_pools() {
    size_t centers_count = num_centers();
    center_real_pool_.assign(centers_count, {});
    center_topn_.assign(centers_count, {});
    center_anchor_pool_.assign(centers_count, {});
    center_pool_trigger_d2_.assign(
        centers_count, std::numeric_limits<float>::infinity()
    );

    size_t max_keep = std::max(
        {config_.center_real_pool_size,
         config_.center_anchor_pool_size,
         config_.center_topn_scan,
         config_.center_real_pool_trigger_topk}
    );
    max_keep = std::min(max_keep, num_);

    for (size_t c = 0; c < centers_count; ++c) {
        const float* center = centers_.data() + (c * dim_);
        std::vector<ScoredPid> scored(num_);
        for (size_t i = 0; i < num_; ++i) {
            scored[i] = {
                euclidean_sqr<float>(center, base_.data() + (i * dim_), dim_),
                static_cast<PID>(i)
            };
        }
        keep_smallest(scored, max_keep);

        size_t real_keep = std::min(config_.center_real_pool_size, scored.size());
        center_real_pool_[c].reserve(real_keep);
        for (size_t i = 0; i < real_keep; ++i) {
            center_real_pool_[c].push_back(scored[i].id);
        }

        size_t topn_keep = std::min(config_.center_topn_scan, scored.size());
        center_topn_[c].reserve(topn_keep);
        for (size_t i = 0; i < topn_keep; ++i) {
            center_topn_[c].push_back(scored[i].id);
        }

        size_t anchor_keep = std::min(config_.center_anchor_pool_size, scored.size());
        center_anchor_pool_[c].reserve(anchor_keep);
        auto append_anchor = [&](size_t rank) {
            PID id = scored[rank].id;
            if (std::find(
                    center_anchor_pool_[c].begin(), center_anchor_pool_[c].end(), id
                ) == center_anchor_pool_[c].end()) {
                center_anchor_pool_[c].push_back(id);
            }
        };
        if (anchor_keep > 0) {
            size_t dense_keep = std::min(
                anchor_keep, std::max<size_t>(8, anchor_keep / 4)
            );
            dense_keep = std::min(dense_keep, scored.size());
            for (size_t i = 0; i < dense_keep; ++i) {
                append_anchor(i);
            }

            size_t remaining = anchor_keep - center_anchor_pool_[c].size();
            if (remaining > 0 && dense_keep < scored.size()) {
                size_t span = scored.size() - dense_keep;
                for (size_t j = 0; j < remaining; ++j) {
                    size_t offset = (j * span) / remaining;
                    size_t rank = std::min(dense_keep + offset, scored.size() - 1);
                    append_anchor(rank);
                }
            }
            for (size_t rank = 0;
                 center_anchor_pool_[c].size() < anchor_keep && rank < scored.size();
                 ++rank) {
                append_anchor(rank);
            }
        }

        size_t trigger_keep =
            std::min(config_.center_real_pool_trigger_topk, scored.size());
        if (trigger_keep > 0) {
            center_pool_trigger_d2_[c] = scored[trigger_keep - 1].distance;
        }
    }
}

inline void TreeRaBitQGraphIndex::build_center_neighbors() {
    size_t centers_count = num_centers();
    center_neighbors_.assign(centers_count, {});
    if (centers_count <= 1) {
        return;
    }

    size_t keep = std::max(
        {config_.graph_build_center_neighbors,
         config_.graph_bridge_center_neighbors,
         config_.center_refine_neighbor_scan,
         config_.center_anchor_neighbor_centers}
    );
    keep = std::min(keep, centers_count - 1);
    for (size_t c = 0; c < centers_count; ++c) {
        std::vector<ScoredPid> scored;
        scored.reserve(centers_count - 1);
        const float* center = centers_.data() + (c * dim_);
        for (size_t other = 0; other < centers_count; ++other) {
            if (other == c) {
                continue;
            }
            scored.push_back(
                {euclidean_sqr<float>(center, centers_.data() + (other * dim_), dim_),
                 static_cast<PID>(other)}
            );
        }
        keep_smallest(scored, keep);
        center_neighbors_[c].reserve(scored.size());
        for (const auto& cand : scored) {
            center_neighbors_[c].push_back(cand.id);
        }
    }
}

inline size_t TreeRaBitQGraphIndex::micro_entry_bucket_for_point(PID center, PID id) const {
    size_t bits = std::min(config_.micro_entry_bits, padded_dim_);
    size_t bucket = 0;
    const float* point =
        rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_);
    const float* center_vec =
        rotated_centers_.data() + (static_cast<size_t>(center) * padded_dim_);
    size_t dim_offset = static_cast<size_t>(center) * bits;
    for (size_t bit = 0; bit < bits; ++bit) {
        size_t d = micro_entry_dims_.empty() ? bit : micro_entry_dims_[dim_offset + bit];
        if (point[d] >= center_vec[d]) {
            bucket |= (size_t{1} << bit);
        }
    }
    return bucket;
}

inline size_t TreeRaBitQGraphIndex::micro_entry_bucket_for_query(
    PID center, const float* rotated_query
) const {
    size_t bits = std::min(config_.micro_entry_bits, padded_dim_);
    size_t bucket = 0;
    const float* center_vec =
        rotated_centers_.data() + (static_cast<size_t>(center) * padded_dim_);
    size_t dim_offset = static_cast<size_t>(center) * bits;
    for (size_t bit = 0; bit < bits; ++bit) {
        size_t d = micro_entry_dims_.empty() ? bit : micro_entry_dims_[dim_offset + bit];
        if (rotated_query[d] >= center_vec[d]) {
            bucket |= (size_t{1} << bit);
        }
    }
    return bucket;
}

inline void TreeRaBitQGraphIndex::build_micro_entry_pools() {
    size_t centers_count = num_centers();
    size_t bits = std::min(config_.micro_entry_bits, padded_dim_);
    micro_entry_bucket_count_ = size_t{1} << bits;
    size_t total_buckets = centers_count * micro_entry_bucket_count_;

    micro_entry_buckets_.assign(total_buckets, {});
    micro_entry_dims_.assign(centers_count * bits, 0);
    micro_entry_counts_.assign(total_buckets, 0);
    micro_entry_centroids_.assign(total_buckets * dim_, 0.0F);

    std::vector<double> residual_sum(padded_dim_, 0.0);
    std::vector<double> residual_sum_sq(padded_dim_, 0.0);
    std::vector<double> residual_var(padded_dim_, 0.0);
    std::vector<size_t> dim_order(padded_dim_, 0);
    std::iota(dim_order.begin(), dim_order.end(), size_t{0});

    for (PID c = 0; c < static_cast<PID>(centers_count); ++c) {
        const auto& topn = center_topn_[c];
        std::fill(residual_sum.begin(), residual_sum.end(), 0.0);
        std::fill(residual_sum_sq.begin(), residual_sum_sq.end(), 0.0);

        const float* center_rotated =
            rotated_centers_.data() + (static_cast<size_t>(c) * padded_dim_);
        for (PID id : topn) {
            const float* point_rotated =
                rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_);
            for (size_t d = 0; d < padded_dim_; ++d) {
                double residual =
                    static_cast<double>(point_rotated[d] - center_rotated[d]);
                residual_sum[d] += residual;
                residual_sum_sq[d] += residual * residual;
            }
        }

        if (!topn.empty()) {
            double inv = 1.0 / static_cast<double>(topn.size());
            for (size_t d = 0; d < padded_dim_; ++d) {
                double mean = residual_sum[d] * inv;
                residual_var[d] = std::max(0.0, residual_sum_sq[d] * inv - mean * mean);
            }
            std::iota(dim_order.begin(), dim_order.end(), size_t{0});
            if (bits < dim_order.size()) {
                std::nth_element(
                    dim_order.begin(),
                    dim_order.begin() + static_cast<std::ptrdiff_t>(bits),
                    dim_order.end(),
                    [&](size_t a, size_t b) {
                        if (residual_var[a] == residual_var[b]) {
                            return a < b;
                        }
                        return residual_var[a] > residual_var[b];
                    }
                );
                dim_order.resize(bits);
            }
            std::sort(
                dim_order.begin(),
                dim_order.end(),
                [&](size_t a, size_t b) {
                    if (residual_var[a] == residual_var[b]) {
                        return a < b;
                    }
                    return residual_var[a] > residual_var[b];
                }
            );
            for (size_t bit = 0; bit < bits; ++bit) {
                micro_entry_dims_[static_cast<size_t>(c) * bits + bit] = dim_order[bit];
            }
            if (dim_order.size() != padded_dim_) {
                dim_order.resize(padded_dim_);
            }
        } else {
            for (size_t bit = 0; bit < bits; ++bit) {
                micro_entry_dims_[static_cast<size_t>(c) * bits + bit] = bit;
            }
        }

        for (PID id : topn) {
            size_t bucket = micro_entry_bucket_for_point(c, id);
            size_t offset = micro_entry_bucket_offset(c, bucket);
            ++micro_entry_counts_[offset];

            float* centroid = micro_entry_centroids_.data() + (offset * dim_);
            const float* point =
                base_.data() + (static_cast<size_t>(id) * dim_);
            for (size_t d = 0; d < dim_; ++d) {
                centroid[d] += point[d];
            }

            auto& entries = micro_entry_buckets_[offset];
            if (entries.size() < config_.micro_entry_bucket_take) {
                entries.push_back(id);
            }
        }

        for (size_t bucket = 0; bucket < micro_entry_bucket_count_; ++bucket) {
            size_t offset = micro_entry_bucket_offset(c, bucket);
            float* centroid = micro_entry_centroids_.data() + (offset * dim_);
            if (micro_entry_counts_[offset] == 0) {
                const float* center_vec =
                    centers_.data() + (static_cast<size_t>(c) * dim_);
                std::copy(center_vec, center_vec + dim_, centroid);
                continue;
            }
            float inv = 1.0F / static_cast<float>(micro_entry_counts_[offset]);
            for (size_t d = 0; d < dim_; ++d) {
                centroid[d] *= inv;
            }
        }
    }
}

inline void TreeRaBitQGraphIndex::quantize_center_codes() {
    size_t centers_count = num_centers();
    size_t bin_bytes =
        round_up_to_multiple(BinDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    size_t ex_bytes = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    center_codes_.reset(centers_count, bin_bytes, ex_bytes);

    std::vector<float> zero(padded_dim_, 0);
    for (size_t c = 0; c < centers_count; ++c) {
        quant::quantize_split_single(
            rotated_centers_.data() + (c * padded_dim_),
            zero.data(),
            padded_dim_,
            ex_bits_,
            center_codes_.bin(c),
            center_codes_.ex(c),
            METRIC_L2,
            index_config_
        );
    }
}

inline void TreeRaBitQGraphIndex::quantize_topn_codes() {
    size_t centers_count = num_centers();
    size_t bin_bytes =
        round_up_to_multiple(BinDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    size_t ex_bytes = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    size_t batch_bytes =
        round_up_to_multiple(BatchDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    topn_codes_.resize(centers_count);
    topn_batch_codes_.resize(centers_count);

    for (size_t c = 0; c < centers_count; ++c) {
        topn_codes_[c].reset(center_topn_[c].size(), bin_bytes, ex_bytes);
        topn_batch_codes_[c].reset(center_topn_[c].size(), batch_bytes);
        const float* center = rotated_centers_.data() + (c * padded_dim_);
        for (size_t i = 0; i < center_topn_[c].size(); ++i) {
            PID id = center_topn_[c][i];
            quant::quantize_split_single(
                rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_),
                center,
                padded_dim_,
                ex_bits_,
                topn_codes_[c].bin(i),
                topn_codes_[c].ex(i),
                METRIC_L2,
                index_config_
            );
        }

        std::vector<float> batch_data(fastscan::kBatchSize * padded_dim_, 0);
        for (size_t batch_id = 0; batch_id < topn_batch_codes_[c].batch_count; ++batch_id) {
            std::fill(batch_data.begin(), batch_data.end(), 0.0F);
            size_t offset = batch_id * fastscan::kBatchSize;
            size_t count =
                std::min(fastscan::kBatchSize, center_topn_[c].size() - offset);
            for (size_t j = 0; j < count; ++j) {
                PID id = center_topn_[c][offset + j];
                std::copy(
                    rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_),
                    rotated_base_.data() + ((static_cast<size_t>(id) + 1) * padded_dim_),
                    batch_data.data() + (j * padded_dim_)
                );
            }
            quant::quantize_one_batch(
                batch_data.data(),
                center,
                count,
                padded_dim_,
                topn_batch_codes_[c].batch(batch_id),
                METRIC_L2
            );
        }
    }
}

inline void TreeRaBitQGraphIndex::quantize_point_codes() {
    size_t bin_bytes =
        round_up_to_multiple(BinDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    size_t ex_bytes = config_.graph_search_full_rabitq
                          ? ExDataMap<float>::data_bytes(padded_dim_, ex_bits_)
                          : 0;
    point_codes_.reset(num_, bin_bytes, ex_bytes);

    for (size_t i = 0; i < num_; ++i) {
        PID center_id = point_center_[i];
        const float* center =
            rotated_centers_.data() + (static_cast<size_t>(center_id) * padded_dim_);
        if (config_.graph_search_full_rabitq) {
            quant::quantize_split_single(
                rotated_base_.data() + (i * padded_dim_),
                center,
                padded_dim_,
                ex_bits_,
                point_codes_.bin(i),
                point_codes_.ex(i),
                METRIC_L2,
                index_config_
            );
        } else {
            quant::quantize_compact_one_bit(
                rotated_base_.data() + (i * padded_dim_),
                center,
                padded_dim_,
                point_codes_.bin(i),
                METRIC_L2
            );
        }
    }
}

inline void TreeRaBitQGraphIndex::build_graph() {
    graph_.assign(num_, {});
    if (num_ <= 1) {
        return;
    }

    size_t centers_count = num_centers();
    constexpr PID kInvalidPid = std::numeric_limits<PID>::max();
    size_t new_degree_limit = std::max<size_t>(1, config_.graph_degree / 2);
    size_t old_degree_limit = config_.graph_degree;
    size_t ef_construction =
        std::max<size_t>(new_degree_limit, config_.graph_build_intra_candidates);
    size_t hint_keep =
        std::max<size_t>(old_degree_limit, config_.graph_build_cross_candidates);

    auto append_unique = [](std::vector<PID>& dst, PID id) {
        if (std::find(dst.begin(), dst.end(), id) == dst.end()) {
            dst.push_back(id);
        }
    };

    auto prune_candidate_pool = [&](PID query_id, std::vector<PID>& candidate_ids, size_t keep) {
        std::sort(candidate_ids.begin(), candidate_ids.end());
        candidate_ids.erase(
            std::unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end()
        );
        candidate_ids.erase(
            std::remove(candidate_ids.begin(), candidate_ids.end(), query_id),
            candidate_ids.end()
        );
        if (candidate_ids.size() <= keep) {
            return;
        }
        std::vector<ScoredPid> scored;
        scored.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            scored.push_back({point_point_l2(query_id, cand), cand});
        }
        keep_smallest(scored, keep);
        candidate_ids.clear();
        candidate_ids.reserve(scored.size());
        for (const auto& cand : scored) {
            candidate_ids.push_back(cand.id);
        }
    };

    auto heuristic_select = [&](PID query_id, std::vector<PID> candidate_ids, size_t max_neighbors) {
        if (max_neighbors == 0) {
            return std::vector<PID>();
        }
        prune_candidate_pool(query_id, candidate_ids, std::max(max_neighbors, candidate_ids.size()));

        std::vector<ScoredPid> items;
        items.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            items.push_back({point_point_l2(query_id, cand), cand});
        }
        std::sort(items.begin(), items.end());

        std::vector<PID> selected;
        selected.reserve(std::min(max_neighbors, items.size()));
        for (const auto& cand : items) {
            bool ok = true;
            for (PID sid : selected) {
                if (point_point_l2(sid, cand.id) < cand.distance) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                selected.push_back(cand.id);
                if (selected.size() >= max_neighbors) {
                    break;
                }
            }
        }

        if (selected.size() < std::min(max_neighbors, items.size())) {
            for (const auto& cand : items) {
                if (std::find(selected.begin(), selected.end(), cand.id) != selected.end()) {
                    continue;
                }
                selected.push_back(cand.id);
                if (selected.size() >= max_neighbors) {
                    break;
                }
            }
        }
        return selected;
    };

    auto prune_neighbors = [&](PID id) {
        auto& row = graph_[id];
        if (row.size() <= old_degree_limit) {
            return;
        }
        row = heuristic_select(id, row, old_degree_limit);
    };

    auto mutual_connect = [&](PID id, const std::vector<PID>& selected_in) {
        std::vector<PID> selected = heuristic_select(id, selected_in, new_degree_limit);
        graph_[id] = selected;
        for (PID nb : selected) {
            auto& row = graph_[nb];
            if (std::find(row.begin(), row.end(), id) == row.end()) {
                row.push_back(id);
            }
            prune_neighbors(nb);
        }
    };

    std::vector<uint32_t> build_marks(num_, 0);
    uint32_t build_epoch = 0;
    auto next_build_epoch = [&]() {
        ++build_epoch;
        if (build_epoch == 0) {
            std::fill(build_marks.begin(), build_marks.end(), 0);
            build_epoch = 1;
        }
        return build_epoch;
    };

    auto search_insert_candidates = [&](PID query_id,
                                        const std::vector<PID>& entry_points,
                                        PID insert_limit,
                                        size_t ef) {
        std::vector<PID> empty;
        if (insert_limit == 0) {
            return empty;
        }
        ef = std::max<size_t>(1, std::min(ef, static_cast<size_t>(insert_limit)));
        uint32_t epoch = next_build_epoch();
        std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
            candidate_heap;
        std::priority_queue<ScoredPid> best_heap;

        auto push_entry = [&](PID ep) {
            if (ep >= insert_limit || build_marks[ep] == epoch) {
                return;
            }
            build_marks[ep] = epoch;
            float d = point_point_l2(query_id, ep);
            candidate_heap.push({d, ep});
            best_heap.push({d, ep});
        };

        for (PID ep : entry_points) {
            push_entry(ep);
        }
        if (candidate_heap.empty()) {
            push_entry(0);
        }

        while (!candidate_heap.empty()) {
            ScoredPid current = candidate_heap.top();
            candidate_heap.pop();
            if (best_heap.size() >= ef && current.distance > best_heap.top().distance) {
                break;
            }

            for (PID nb : graph_[current.id]) {
                if (nb >= insert_limit || build_marks[nb] == epoch) {
                    continue;
                }
                build_marks[nb] = epoch;
                float d = point_point_l2(query_id, nb);
                if (best_heap.size() < ef || d < best_heap.top().distance) {
                    candidate_heap.push({d, nb});
                    best_heap.push({d, nb});
                    if (best_heap.size() > ef) {
                        best_heap.pop();
                    }
                }
            }
        }

        std::vector<ScoredPid> scored;
        scored.reserve(best_heap.size());
        while (!best_heap.empty()) {
            scored.push_back(best_heap.top());
            best_heap.pop();
        }
        std::sort(scored.begin(), scored.end());

        std::vector<PID> result;
        result.reserve(scored.size());
        for (const auto& cand : scored) {
            result.push_back(cand.id);
        }
        return result;
    };

    std::vector<PID> center_entry(centers_count, kInvalidPid);
    std::vector<float> center_entry_d2(
        centers_count, std::numeric_limits<float>::max()
    );
    PID global_entry = kInvalidPid;

    for (PID id = 0; id < num_; ++id) {
        PID center_id = point_center_[id];
        if (id == 0) {
            center_entry[center_id] = id;
            center_entry_d2[center_id] = l2_to_center(base_.data(), center_id);
            global_entry = id;
            continue;
        }

        std::vector<PID> entry_points;
        if (center_entry[center_id] != kInvalidPid && center_entry[center_id] < id) {
            append_unique(entry_points, center_entry[center_id]);
        }
        size_t neighbor_entries =
            std::min(config_.graph_build_center_neighbors, center_neighbors_[center_id].size());
        for (size_t i = 0; i < neighbor_entries; ++i) {
            PID nc = center_neighbors_[center_id][i];
            if (center_entry[nc] != kInvalidPid && center_entry[nc] < id) {
                append_unique(entry_points, center_entry[nc]);
            }
        }
        if (global_entry != kInvalidPid && global_entry < id) {
            append_unique(entry_points, global_entry);
        }
        if (entry_points.empty()) {
            entry_points.push_back(0);
        }

        std::vector<PID> candidates =
            search_insert_candidates(id, entry_points, id, ef_construction);

        std::vector<PID> hint_centers;
        hint_centers.reserve(neighbor_entries + 1);
        hint_centers.push_back(center_id);
        for (size_t i = 0; i < neighbor_entries; ++i) {
            hint_centers.push_back(center_neighbors_[center_id][i]);
        }
        size_t per_center_hint =
            std::max<size_t>(1, div_round_up(hint_keep, hint_centers.size()));
        size_t scan_prefix = per_center_hint * 8;
        for (PID hc : hint_centers) {
            const auto& topn = center_topn_[hc];
            size_t scan = std::min(topn.size(), scan_prefix);
            size_t added = 0;
            for (size_t j = 0; j < scan && added < per_center_hint; ++j) {
                PID other = topn[j];
                if (other < id && other != id) {
                    append_unique(candidates, other);
                    ++added;
                }
            }
        }
        prune_candidate_pool(id, candidates, std::max(ef_construction, hint_keep));
        if (candidates.empty() && global_entry != kInvalidPid) {
            candidates.push_back(global_entry);
        }

        std::vector<PID> selected = heuristic_select(id, candidates, new_degree_limit);
        if (selected.empty() && !candidates.empty()) {
            selected.push_back(candidates.front());
        }
        mutual_connect(id, selected);

        float center_d2 = l2_to_center(base_.data() + (static_cast<size_t>(id) * dim_), center_id);
        if (center_d2 < center_entry_d2[center_id]) {
            center_entry_d2[center_id] = center_d2;
            center_entry[center_id] = id;
        }
        global_entry = id;
    }

    if (config_.graph_build_bridge_edges) {
        add_bridge_edges();
    }
    if (config_.graph_query_adjacency_order) {
        order_graph_for_query();
    }
    if (config_.graph_reorder_by_center) {
        reorder_graph_by_center();
        if (!config_.use_center_anchor_seeds && !config_.use_micro_entry_seeds) {
            quantize_topn_codes();
        }
        if (config_.graph_search_use_rabitq) {
            quantize_point_codes();
        }
    }
    finalize_graph_csr();
}

inline bool TreeRaBitQGraphIndex::append_graph_edge(PID a, PID b) {
    if (a >= num_ || b >= num_ || a == b) {
        return false;
    }
    auto& row = graph_[a];
    if (std::find(row.begin(), row.end(), b) != row.end()) {
        return false;
    }
    row.push_back(b);
    return true;
}

inline void TreeRaBitQGraphIndex::add_bridge_edges() {
    graph_bridge_edges_added_ = 0;
    size_t centers_count = num_centers();
    if (centers_count <= 1 || graph_.empty()) {
        return;
    }

    size_t center_keep = std::min(
        config_.graph_bridge_center_neighbors,
        centers_count > 0 ? centers_count - 1 : 0
    );
    size_t points_keep = config_.graph_bridge_points_per_center;
    size_t scan_keep = config_.graph_bridge_candidate_scan;

    auto connect = [&](PID a, PID b) {
        if (append_graph_edge(a, b)) {
            ++graph_bridge_edges_added_;
        }
        if (append_graph_edge(b, a)) {
            ++graph_bridge_edges_added_;
        }
    };

    for (PID c = 0; c < centers_count; ++c) {
        const auto& left = center_topn_[c];
        if (left.empty()) {
            continue;
        }
        size_t neighbor_count = std::min(center_keep, center_neighbors_[c].size());
        size_t left_scan = std::min(scan_keep, left.size());
        size_t left_take = std::min(points_keep, left_scan);
        for (size_t ni = 0; ni < neighbor_count; ++ni) {
            PID nc = center_neighbors_[c][ni];
            if (nc >= centers_count || nc <= c) {
                continue;
            }
            const auto& right = center_topn_[nc];
            if (right.empty()) {
                continue;
            }
            size_t right_scan = std::min(scan_keep, right.size());
            size_t right_take =
                std::min(right_scan, std::max(points_keep, points_keep * 4));

            std::vector<ScoredPid> left_boundary;
            left_boundary.reserve(left_scan);
            const float* right_center =
                centers_.data() + (static_cast<size_t>(nc) * dim_);
            for (size_t li = 0; li < left_scan; ++li) {
                PID id = left[li];
                left_boundary.push_back(
                    {euclidean_sqr_fast(
                         right_center, base_.data() + (static_cast<size_t>(id) * dim_), dim_
                     ),
                     id}
                );
            }
            keep_smallest(left_boundary, left_take);

            std::vector<ScoredPid> right_boundary;
            right_boundary.reserve(right_scan);
            const float* left_center = centers_.data() + (static_cast<size_t>(c) * dim_);
            for (size_t ri = 0; ri < right_scan; ++ri) {
                PID id = right[ri];
                right_boundary.push_back(
                    {euclidean_sqr_fast(
                         left_center, base_.data() + (static_cast<size_t>(id) * dim_), dim_
                     ),
                     id}
                );
            }
            keep_smallest(right_boundary, right_take);

            for (const auto& left_item : left_boundary) {
                PID a = left_item.id;
                std::vector<ScoredPid> scored;
                scored.reserve(right_boundary.size());
                for (const auto& right_item : right_boundary) {
                    PID b = right_item.id;
                    if (a != b) {
                        scored.push_back({point_point_l2(a, b), b});
                    }
                }
                keep_smallest(scored, std::min(points_keep, scored.size()));
                for (const auto& cand : scored) {
                    connect(a, cand.id);
                }
            }
        }
    }
}

inline void TreeRaBitQGraphIndex::order_graph_for_query() {
    if (graph_.empty()) {
        return;
    }
    size_t cap_hint = config_.graph_search_neighbor_cap == 0
                          ? config_.graph_degree
                          : config_.graph_search_neighbor_cap;
    cap_hint = std::max<size_t>(1, cap_hint);

    for (PID id = 0; id < num_; ++id) {
        auto& row = graph_[id];
        if (row.empty()) {
            continue;
        }
        std::vector<PID> deduped;
        deduped.reserve(row.size());
        for (PID nb : row) {
            if (nb == id || nb >= num_) {
                continue;
            }
            if (std::find(deduped.begin(), deduped.end(), nb) == deduped.end()) {
                deduped.push_back(nb);
            }
        }

        std::vector<ScoredPid> local;
        std::vector<ScoredPid> cross;
        local.reserve(deduped.size());
        cross.reserve(deduped.size());
        PID center_id = point_center_[id];
        for (PID nb : deduped) {
            ScoredPid item{point_point_l2(id, nb), nb};
            if (point_center_[nb] == center_id) {
                local.push_back(item);
            } else {
                cross.push_back(item);
            }
        }
        std::sort(local.begin(), local.end());
        std::sort(cross.begin(), cross.end());

        size_t cross_front_budget =
            std::min(cross.size(), std::max<size_t>(1, cap_hint / 4));
        std::vector<PID> ordered;
        ordered.reserve(local.size() + cross.size());
        size_t li = 0;
        size_t ci = 0;
        while (li < local.size() || ci < cross_front_budget) {
            for (size_t repeat = 0; repeat < 2 && li < local.size(); ++repeat) {
                ordered.push_back(local[li++].id);
            }
            if (ci < cross_front_budget) {
                ordered.push_back(cross[ci++].id);
            }
        }
        while (li < local.size()) {
            ordered.push_back(local[li++].id);
        }
        while (ci < cross.size()) {
            ordered.push_back(cross[ci++].id);
        }
        row.swap(ordered);
    }
}

inline void TreeRaBitQGraphIndex::reorder_graph_by_center() {
    if (num_ == 0 || graph_.empty()) {
        return;
    }

    std::vector<PID> new_to_old(num_);
    std::iota(new_to_old.begin(), new_to_old.end(), static_cast<PID>(0));
    std::stable_sort(new_to_old.begin(), new_to_old.end(), [&](PID a, PID b) {
        PID ca = point_center_[a];
        PID cb = point_center_[b];
        if (ca != cb) {
            return ca < cb;
        }
        return a < b;
    });

    std::vector<PID> old_to_new(num_);
    for (PID new_id = 0; new_id < num_; ++new_id) {
        old_to_new[new_to_old[new_id]] = new_id;
    }

    auto remap_ids = [&](std::vector<PID>& ids) {
        for (PID& id : ids) {
            if (id < num_) {
                id = old_to_new[id];
            }
        }
    };

    std::vector<float> new_base(num_ * dim_);
    for (PID new_id = 0; new_id < num_; ++new_id) {
        PID old_id = new_to_old[new_id];
        std::copy_n(
            base_.data() + (static_cast<size_t>(old_id) * dim_),
            dim_,
            new_base.data() + (static_cast<size_t>(new_id) * dim_)
        );
    }
    base_.swap(new_base);

    if (!base_sq_norms_.empty()) {
        std::vector<float> new_norms(num_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            new_norms[new_id] = base_sq_norms_[new_to_old[new_id]];
        }
        base_sq_norms_.swap(new_norms);
    }

    if (!rotated_base_.empty()) {
        std::vector<float> new_rotated(num_ * padded_dim_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            PID old_id = new_to_old[new_id];
            std::copy_n(
                rotated_base_.data() + (static_cast<size_t>(old_id) * padded_dim_),
                padded_dim_,
                new_rotated.data() + (static_cast<size_t>(new_id) * padded_dim_)
            );
        }
        rotated_base_.swap(new_rotated);
    }

    if (!external_ids_.empty()) {
        std::vector<PID> new_external_ids(num_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            new_external_ids[new_id] = external_ids_[new_to_old[new_id]];
        }
        external_ids_.swap(new_external_ids);
    }

    std::vector<PID> new_point_center(num_);
    for (PID new_id = 0; new_id < num_; ++new_id) {
        new_point_center[new_id] = point_center_[new_to_old[new_id]];
    }
    point_center_.swap(new_point_center);

    for (auto& ids : leaf_ids_) {
        remap_ids(ids);
    }
    for (auto& ids : center_real_pool_) {
        remap_ids(ids);
    }
    for (auto& ids : center_topn_) {
        remap_ids(ids);
    }
    for (auto& ids : center_anchor_pool_) {
        remap_ids(ids);
    }
    for (auto& ids : micro_entry_buckets_) {
        remap_ids(ids);
    }

    std::vector<std::vector<PID>> new_graph(num_);
    for (PID old_id = 0; old_id < num_; ++old_id) {
        PID new_id = old_to_new[old_id];
        auto& dst = new_graph[new_id];
        dst.reserve(graph_[old_id].size());
        for (PID nb : graph_[old_id]) {
            if (nb < num_) {
                dst.push_back(old_to_new[nb]);
            }
        }
    }
    graph_.swap(new_graph);
    graph_reordered_by_center_ = true;
}

inline void TreeRaBitQGraphIndex::finalize_graph_csr() {
    graph_offsets_.assign(num_ + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < num_; ++i) {
        auto& row = graph_[i];
        std::vector<PID> deduped;
        deduped.reserve(row.size());
        for (PID id : row) {
            if (id == static_cast<PID>(i)) {
                continue;
            }
            if (std::find(deduped.begin(), deduped.end(), id) == deduped.end()) {
                deduped.push_back(id);
            }
        }
        row.swap(deduped);
        graph_offsets_[i] = total;
        total += row.size();
    }
    graph_offsets_[num_] = total;

    graph_indices_.clear();
    graph_indices_.reserve(total);
    for (const auto& row : graph_) {
        graph_indices_.insert(graph_indices_.end(), row.begin(), row.end());
    }

    std::vector<std::vector<PID>> empty_graph;
    graph_.swap(empty_graph);
}

inline std::vector<PID> TreeRaBitQGraphIndex::search(
    const float* query, size_t k, TreeRaBitQGraphQueryStats* stats
) const {
    if (query == nullptr || k == 0) {
        return {};
    }
    if (num_ == 0) {
        throw std::runtime_error("TreeRaBitQGraphIndex is not constructed");
    }
    if (stats == nullptr) {
        return search_fast(query, k);
    }

    using ProfileClock = std::chrono::high_resolution_clock;
    auto profile_now = []() { return ProfileClock::now(); };
    auto elapsed_ms = [](ProfileClock::time_point begin, ProfileClock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    auto total_begin = profile_now();

    TreeRaBitQGraphQueryStats local_stats;
    local_stats.topk = k;
    auto prepare_begin = profile_now();
    float query_sq_norm = squared_norm(query, dim_);
    std::vector<float> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());
    SplitSingleQuery<float> query_wrapper(
        rotated_query.data(), padded_dim_, ex_bits_, query_config_, METRIC_L2
    );
    auto prepare_end = profile_now();
    local_stats.query_prepare_ms = elapsed_ms(prepare_begin, prepare_end);

    auto route_begin = profile_now();
    PID routed_center = 0;
    if (config_.center_entry_mode != CenterEntryMode::RabitqOnly) {
        routed_center = route_to_center(query);
    }
    auto route_end = profile_now();
    local_stats.route_ms = elapsed_ms(route_begin, route_end);

    auto center_refine_begin = profile_now();
    PID best_center = routed_center;
    std::vector<PID> candidate_centers;
    if (config_.center_entry_mode == CenterEntryMode::RabitqOnly) {
        best_center =
            refine_center(query, query_wrapper, routed_center, false, local_stats, &candidate_centers);
    } else if (config_.center_entry_mode == CenterEntryMode::TreeThenRabitq) {
        best_center =
            refine_center(query, query_wrapper, routed_center, true, local_stats, &candidate_centers);
    }
    if (candidate_centers.empty()) {
        candidate_centers.push_back(best_center);
    }
    auto center_refine_end = profile_now();
    local_stats.center_refine_ms =
        elapsed_ms(center_refine_begin, center_refine_end);

    auto seed_begin = profile_now();
    float best_center_dist2 = l2_to_center(query, best_center);
    ++local_stats.exact_distance_evals;

    local_stats.routed_center = routed_center;
    local_stats.best_center = best_center;

    std::vector<PID> seeds =
        initial_seeds(
            query,
            query_sq_norm,
            query_wrapper,
            best_center,
            candidate_centers,
            best_center_dist2,
            local_stats
        );
    auto seed_end = profile_now();
    local_stats.seed_select_ms = elapsed_ms(seed_begin, seed_end);

    auto graph_prepare_begin = profile_now();
    std::vector<float> center_query_dist2;
    std::vector<float> center_query_norm;
    if (config_.graph_search_use_rabitq) {
        size_t centers_count = num_centers();
        center_query_dist2.resize(centers_count);
        center_query_norm.resize(centers_count);
        if (config_.graph_lazy_center_distance) {
            std::fill(center_query_dist2.begin(), center_query_dist2.end(), -1.0F);
            std::fill(center_query_norm.begin(), center_query_norm.end(), 0.0F);
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
        } else {
            for (size_t c = 0; c < centers_count; ++c) {
                float d2 = l2_to_center(query, static_cast<PID>(c));
                center_query_dist2[c] = d2;
                center_query_norm[c] = std::sqrt(std::max(d2, 0.0F));
            }
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
            local_stats.graph_center_exact_scores = centers_count;
            local_stats.exact_distance_evals += centers_count;
        }
    }
    auto graph_prepare_end = profile_now();
    local_stats.graph_prepare_ms =
        elapsed_ms(graph_prepare_begin, graph_prepare_end);

    auto ensure_center_query_distance = [&](PID center_id) {
        if (config_.graph_lazy_center_distance && center_query_dist2[center_id] < 0.0F) {
            float d2 = l2_to_center(query, center_id);
            center_query_dist2[center_id] = d2;
            center_query_norm[center_id] = std::sqrt(std::max(d2, 0.0F));
            ++local_stats.graph_center_exact_scores;
            ++local_stats.exact_distance_evals;
        }
    };

    auto graph_distance = [&](PID id) {
        ++local_stats.graph_distance_evals;
        if (!config_.graph_search_use_rabitq) {
            ++local_stats.exact_distance_evals;
            return l2_to_point(query, query_sq_norm, id);
        }
        PID center_id = point_center_[id];
        ensure_center_query_distance(center_id);
        ++local_stats.graph_est_evals;
        if (config_.graph_search_full_rabitq) {
            return estimate_code(
                point_codes_.bin(id),
                point_codes_.ex(id),
                query_wrapper,
                center_query_dist2[center_id],
                center_query_norm[center_id]
            );
        }
        return estimate_code_onebit(
            point_codes_.bin(id),
            query_wrapper,
            center_query_dist2[center_id],
            center_query_norm[center_id]
        );
    };

    auto graph_search_begin = profile_now();
    uint32_t visit_epoch = next_visit_epoch();
    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, config_.ef_search * config_.graph_degree));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, config_.ef_search * config_.graph_degree));
    size_t window_keep = std::max(k, config_.graph_rerank_candidates);
    std::priority_queue<ScoredPid> best_window;

    auto push_seen = [&](ScoredPid item) {
        seen.push_back(item);
        if (config_.graph_early_stop) {
            best_window.push(item);
            if (best_window.size() > window_keep) {
                best_window.pop();
            }
        }
    };

    for (PID id : seeds) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            continue;
        }
        visit_marks_[id] = visit_epoch;
        ++local_stats.visited_nodes;
        float dist = graph_distance(id);
        candidate_queue.push({dist, id});
        push_seen({dist, id});
        ++local_stats.seed_pushes;
        ++local_stats.heap_pushes;
    }

    if (candidate_queue.empty()) {
        ++local_stats.fallback_count;
        local_stats.fallback = true;
        local_stats.mode = SeedMode::Fallback;
        PID fallback = center_real_pool_[best_center].empty() ? 0 : center_real_pool_[best_center][0];
        visit_marks_[fallback] = visit_epoch;
        ++local_stats.visited_nodes;
        float dist = graph_distance(fallback);
        candidate_queue.push({dist, fallback});
        push_seen({dist, fallback});
        ++local_stats.seed_pushes;
        ++local_stats.heap_pushes;
    }

    while (!candidate_queue.empty() && local_stats.node_expansions < config_.ef_search) {
        if (config_.graph_early_stop && best_window.size() >= window_keep &&
            candidate_queue.top().distance > best_window.top().distance) {
            local_stats.graph_early_stop_count = 1;
            break;
        }

        ScoredPid current = candidate_queue.top();
        candidate_queue.pop();
        ++local_stats.heap_pops;
        ++local_stats.node_expansions;

        size_t neighbor_begin = graph_offsets_[current.id];
        size_t neighbor_end = graph_offsets_[static_cast<size_t>(current.id) + 1];
        size_t neighbor_count = neighbor_end - neighbor_begin;
        size_t neighbor_limit = config_.graph_search_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(config_.graph_search_neighbor_cap, neighbor_count);
        for (size_t ni = 0; ni < neighbor_limit; ++ni) {
            PID nb = graph_indices_[neighbor_begin + ni];
            ++local_stats.edges_scanned;
            if (visit_marks_[nb] == visit_epoch) {
                continue;
            }
            visit_marks_[nb] = visit_epoch;
            ++local_stats.visited_nodes;
            float dist = graph_distance(nb);
            candidate_queue.push({dist, nb});
            ++local_stats.heap_pushes;
            push_seen({dist, nb});
        }
    }
    auto graph_search_end = profile_now();
    local_stats.graph_search_ms = elapsed_ms(graph_search_begin, graph_search_end);

    local_stats.visited = seen.size();
    local_stats.jumps = local_stats.node_expansions;

    auto final_begin = profile_now();
    std::vector<PID> result;
    if (config_.graph_search_use_rabitq) {
        size_t rerank_keep =
            std::min(seen.size(), std::max(k, config_.graph_rerank_candidates));
        keep_smallest(seen, rerank_keep);

        std::vector<ScoredPid> exact_scores;
        exact_scores.reserve(rerank_keep);
        for (size_t i = 0; i < rerank_keep; ++i) {
            exact_scores.push_back(
                {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id}
            );
        }
        local_stats.exact_rerank_scores = rerank_keep;
        local_stats.final_candidates = rerank_keep;
        local_stats.exact_distance_evals += rerank_keep;
        keep_smallest(exact_scores, std::min(k, exact_scores.size()));

        result.reserve(std::min(k, exact_scores.size()));
        for (size_t i = 0; i < std::min(k, exact_scores.size()); ++i) {
            result.push_back(exact_scores[i].id);
        }
    } else {
        size_t rerank_keep =
            std::min(seen.size(), std::max(k, config_.graph_rerank_candidates));
        keep_smallest(seen, rerank_keep);
        local_stats.final_candidates = rerank_keep;

        result.reserve(std::min(k, seen.size()));
        for (size_t i = 0; i < std::min(k, seen.size()); ++i) {
            result.push_back(seen[i].id);
        }
    }
    auto final_end = profile_now();
    local_stats.final_select_ms = elapsed_ms(final_begin, final_end);
    local_stats.query_total_ms = elapsed_ms(total_begin, final_end);

    if (stats != nullptr) {
        *stats = local_stats;
    }
    for (PID& id : result) {
        id = external_id(id);
    }
    return result;
}

inline std::vector<PID> TreeRaBitQGraphIndex::search_fast(
    const float* query, size_t k
) const {
    float query_sq_norm = squared_norm(query, dim_);
    std::vector<float> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());
    SplitSingleQuery<float> query_wrapper(
        rotated_query.data(), padded_dim_, ex_bits_, query_config_, METRIC_L2
    );

    PID routed_center = 0;
    if (config_.center_entry_mode != CenterEntryMode::RabitqOnly) {
        routed_center = route_to_center(query);
    }
    PID best_center = routed_center;
    std::vector<PID> candidate_centers;
    if (config_.center_entry_mode == CenterEntryMode::RabitqOnly) {
        best_center =
            refine_center_fast(query, query_wrapper, routed_center, false, &candidate_centers);
    } else if (config_.center_entry_mode == CenterEntryMode::TreeThenRabitq) {
        best_center =
            refine_center_fast(query, query_wrapper, routed_center, true, &candidate_centers);
    }
    if (candidate_centers.empty()) {
        candidate_centers.push_back(best_center);
    }
    float best_center_dist2 = l2_to_center(query, best_center);

    std::vector<PID> seeds = initial_seeds_fast(
        query, query_sq_norm, query_wrapper, best_center, candidate_centers, best_center_dist2
    );

    std::vector<float> center_query_dist2;
    std::vector<float> center_query_norm;
    if (config_.graph_search_use_rabitq) {
        size_t centers_count = num_centers();
        center_query_dist2.resize(centers_count);
        center_query_norm.resize(centers_count);
        if (config_.graph_lazy_center_distance) {
            std::fill(center_query_dist2.begin(), center_query_dist2.end(), -1.0F);
            std::fill(center_query_norm.begin(), center_query_norm.end(), 0.0F);
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
        } else {
            for (size_t c = 0; c < centers_count; ++c) {
                float d2 = l2_to_center(query, static_cast<PID>(c));
                center_query_dist2[c] = d2;
                center_query_norm[c] = std::sqrt(std::max(d2, 0.0F));
            }
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
        }
    }

    auto ensure_center_query_distance = [&](PID center_id) {
        if (config_.graph_lazy_center_distance && center_query_dist2[center_id] < 0.0F) {
            float d2 = l2_to_center(query, center_id);
            center_query_dist2[center_id] = d2;
            center_query_norm[center_id] = std::sqrt(std::max(d2, 0.0F));
        }
    };

    auto graph_distance = [&](PID id) {
        if (!config_.graph_search_use_rabitq) {
            return l2_to_point(query, query_sq_norm, id);
        }
        PID center_id = point_center_[id];
        ensure_center_query_distance(center_id);
        if (config_.graph_search_full_rabitq) {
            return estimate_code(
                point_codes_.bin(id),
                point_codes_.ex(id),
                query_wrapper,
                center_query_dist2[center_id],
                center_query_norm[center_id]
            );
        }
        return estimate_code_onebit(
            point_codes_.bin(id),
            query_wrapper,
            center_query_dist2[center_id],
            center_query_norm[center_id]
        );
    };

    uint32_t visit_epoch = next_visit_epoch();
    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, config_.ef_search * config_.graph_degree));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, config_.ef_search * config_.graph_degree));
    size_t window_keep = std::max(k, config_.graph_rerank_candidates);
    std::priority_queue<ScoredPid> best_window;

    auto push_seen = [&](ScoredPid item) {
        seen.push_back(item);
        if (config_.graph_early_stop) {
            best_window.push(item);
            if (best_window.size() > window_keep) {
                best_window.pop();
            }
        }
    };

    for (PID id : seeds) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            continue;
        }
        visit_marks_[id] = visit_epoch;
        float dist = graph_distance(id);
        candidate_queue.push({dist, id});
        push_seen({dist, id});
    }

    if (candidate_queue.empty()) {
        PID fallback = center_real_pool_[best_center].empty() ? 0 : center_real_pool_[best_center][0];
        visit_marks_[fallback] = visit_epoch;
        float dist = graph_distance(fallback);
        candidate_queue.push({dist, fallback});
        push_seen({dist, fallback});
    }

    size_t expansions = 0;
    while (!candidate_queue.empty() && expansions < config_.ef_search) {
        if (config_.graph_early_stop && best_window.size() >= window_keep &&
            candidate_queue.top().distance > best_window.top().distance) {
            break;
        }

        ScoredPid current = candidate_queue.top();
        candidate_queue.pop();
        ++expansions;

        size_t neighbor_begin = graph_offsets_[current.id];
        size_t neighbor_end = graph_offsets_[static_cast<size_t>(current.id) + 1];
        size_t neighbor_count = neighbor_end - neighbor_begin;
        size_t neighbor_limit = config_.graph_search_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(config_.graph_search_neighbor_cap, neighbor_count);
        for (size_t ni = 0; ni < neighbor_limit; ++ni) {
            PID nb = graph_indices_[neighbor_begin + ni];
            if (visit_marks_[nb] == visit_epoch) {
                continue;
            }
            visit_marks_[nb] = visit_epoch;
            float dist = graph_distance(nb);
            candidate_queue.push({dist, nb});
            push_seen({dist, nb});
        }
    }

    std::vector<PID> result;
    if (config_.graph_search_use_rabitq) {
        size_t rerank_keep =
            std::min(seen.size(), std::max(k, config_.graph_rerank_candidates));
        keep_smallest(seen, rerank_keep);

        std::vector<ScoredPid> exact_scores;
        exact_scores.reserve(rerank_keep);
        for (size_t i = 0; i < rerank_keep; ++i) {
            exact_scores.push_back(
                {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id}
            );
        }
        keep_smallest(exact_scores, std::min(k, exact_scores.size()));

        result.reserve(std::min(k, exact_scores.size()));
        for (size_t i = 0; i < std::min(k, exact_scores.size()); ++i) {
            result.push_back(exact_scores[i].id);
        }
    } else {
        size_t rerank_keep =
            std::min(seen.size(), std::max(k, config_.graph_rerank_candidates));
        keep_smallest(seen, rerank_keep);

        result.reserve(std::min(k, seen.size()));
        for (size_t i = 0; i < std::min(k, seen.size()); ++i) {
            result.push_back(seen[i].id);
        }
    }

    for (PID& id : result) {
        id = external_id(id);
    }
    return result;
}

inline PID TreeRaBitQGraphIndex::refine_center(
    const float* query,
    const SplitSingleQuery<float>& query_wrapper,
    PID routed_center,
    bool use_routed_center,
    TreeRaBitQGraphQueryStats& stats,
    std::vector<PID>* exact_centers
) const {
    std::vector<ScoredPid> center_scores;
    size_t centers_count = num_centers();
    bool use_neighbor_scan =
        use_routed_center && config_.center_refine_neighbor_scan > 0 &&
        routed_center < center_neighbors_.size();
    size_t scan_limit = use_neighbor_scan
                            ? std::min(
                                  config_.center_refine_neighbor_scan,
                                  center_neighbors_[routed_center].size()
                              )
                            : centers_count;
    center_scores.reserve(scan_limit + 1);

    auto add_center_estimate = [&](PID c) {
        float est = estimate_code(center_codes_.bin(c), center_codes_.ex(c), query_wrapper, 0, 0);
        center_scores.push_back({est, static_cast<PID>(c)});
    };

    if (use_neighbor_scan) {
        add_center_estimate(routed_center);
        for (size_t i = 0; i < scan_limit; ++i) {
            add_center_estimate(center_neighbors_[routed_center][i]);
        }
    } else {
        for (size_t c = 0; c < centers_count; ++c) {
            add_center_estimate(static_cast<PID>(c));
        }
    }
    stats.center_estimates = center_scores.size();
    stats.rabitq_center_est_evals += center_scores.size();
    keep_smallest(center_scores, std::min(config_.center_scan_keep, center_scores.size()));

    bool has_routed = false;
    for (const auto& cand : center_scores) {
        has_routed = has_routed || cand.id == routed_center;
    }
    if (use_routed_center && !has_routed) {
        ++stats.exact_distance_evals;
        center_scores.push_back(
            {l2_to_center(query, routed_center), static_cast<PID>(routed_center)}
        );
    }

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(center_scores.size());
    for (const auto& cand : center_scores) {
        bool duplicate = false;
        for (const auto& prev : exact_scores) {
            if (prev.id == cand.id) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        exact_scores.push_back({l2_to_center(query, cand.id), cand.id});
    }
    stats.center_exact_scores = exact_scores.size();
    stats.exact_distance_evals += exact_scores.size();
    keep_smallest(exact_scores, std::min(config_.exact_center_keep, exact_scores.size()));
    if (exact_centers != nullptr) {
        exact_centers->clear();
        exact_centers->reserve(exact_scores.size());
        for (const auto& cand : exact_scores) {
            exact_centers->push_back(cand.id);
        }
    }
    return exact_scores.empty() ? routed_center : exact_scores.front().id;
}

inline PID TreeRaBitQGraphIndex::refine_center_fast(
    const float* query,
    const SplitSingleQuery<float>& query_wrapper,
    PID routed_center,
    bool use_routed_center,
    std::vector<PID>* exact_centers
) const {
    std::vector<ScoredPid> center_scores;
    size_t centers_count = num_centers();
    bool use_neighbor_scan =
        use_routed_center && config_.center_refine_neighbor_scan > 0 &&
        routed_center < center_neighbors_.size();
    size_t scan_limit = use_neighbor_scan
                            ? std::min(
                                  config_.center_refine_neighbor_scan,
                                  center_neighbors_[routed_center].size()
                              )
                            : centers_count;
    center_scores.reserve(scan_limit + 1);

    auto add_center_estimate = [&](PID c) {
        float est = estimate_code(center_codes_.bin(c), center_codes_.ex(c), query_wrapper, 0, 0);
        center_scores.push_back({est, static_cast<PID>(c)});
    };

    if (use_neighbor_scan) {
        add_center_estimate(routed_center);
        for (size_t i = 0; i < scan_limit; ++i) {
            add_center_estimate(center_neighbors_[routed_center][i]);
        }
    } else {
        for (size_t c = 0; c < centers_count; ++c) {
            add_center_estimate(static_cast<PID>(c));
        }
    }
    keep_smallest(center_scores, std::min(config_.center_scan_keep, center_scores.size()));

    bool has_routed = false;
    for (const auto& cand : center_scores) {
        has_routed = has_routed || cand.id == routed_center;
    }
    if (use_routed_center && !has_routed) {
        center_scores.push_back(
            {l2_to_center(query, routed_center), static_cast<PID>(routed_center)}
        );
    }

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(center_scores.size());
    for (const auto& cand : center_scores) {
        bool duplicate = false;
        for (const auto& prev : exact_scores) {
            if (prev.id == cand.id) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            exact_scores.push_back({l2_to_center(query, cand.id), cand.id});
        }
    }
    keep_smallest(exact_scores, std::min(config_.exact_center_keep, exact_scores.size()));
    if (exact_centers != nullptr) {
        exact_centers->clear();
        exact_centers->reserve(exact_scores.size());
        for (const auto& cand : exact_scores) {
            exact_centers->push_back(cand.id);
        }
    }
    return exact_scores.empty() ? routed_center : exact_scores.front().id;
}

inline std::vector<PID> TreeRaBitQGraphIndex::initial_seeds(
    const float* query,
    float query_sq_norm,
    const SplitSingleQuery<float>& query_wrapper,
    PID best_center,
    const std::vector<PID>& candidate_centers,
    float best_center_dist2,
    TreeRaBitQGraphQueryStats& stats
) const {
    if (best_center_dist2 <= center_pool_trigger_d2_[best_center]) {
        stats.mode = SeedMode::CenterRealPool;
        stats.init_mode = SeedMode::CenterRealPool;
        stats.trigger_pass = true;
        stats.trigger_pass_count = 1;
        const auto& pool = center_real_pool_[best_center];
        size_t take = std::min(config_.center_real_pool_take, pool.size());
        stats.entry_points_raw = pool.size();
        stats.entry_points_before_cap = pool.size();
        stats.entry_points_after_cap = take;
        stats.init_size = take;
        return std::vector<PID>(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(take));
    }

    if (config_.use_micro_entry_seeds && best_center < num_centers() &&
        micro_entry_bucket_count_ > 0) {
        stats.mode = SeedMode::MicroEntry;
        stats.init_mode = SeedMode::MicroEntry;

        std::vector<PID> seed_centers;
        auto append_center_unique = [&](PID center_id) {
            if (center_id >= num_centers()) {
                return;
            }
            if (std::find(seed_centers.begin(), seed_centers.end(), center_id) ==
                seed_centers.end()) {
                seed_centers.push_back(center_id);
            }
        };

        append_center_unique(best_center);
        for (PID center_id : candidate_centers) {
            append_center_unique(center_id);
            if (seed_centers.size() >= config_.micro_entry_center_keep) {
                break;
            }
        }

        std::vector<ScoredPid> bucket_scores;
        bucket_scores.reserve(seed_centers.size() * micro_entry_bucket_count_);
        for (PID center_id : seed_centers) {
            size_t query_bucket =
                micro_entry_bucket_for_query(center_id, query_wrapper.rotated_query());
            for (size_t bucket = 0; bucket < micro_entry_bucket_count_; ++bucket) {
                size_t offset = micro_entry_bucket_offset(center_id, bucket);
                if (micro_entry_counts_[offset] == 0) {
                    continue;
                }
                const auto& entries = micro_entry_buckets_[offset];
                stats.entry_points_raw += entries.size();
                float d = euclidean_sqr_fast(
                    query, micro_entry_centroids_.data() + (offset * dim_), dim_
                );
                if (bucket == query_bucket) {
                    d *= 0.95F;
                }
                bucket_scores.push_back({d, static_cast<PID>(offset)});
            }
        }
        stats.exact_distance_evals += bucket_scores.size();

        size_t bucket_keep =
            std::min(bucket_scores.size(), config_.micro_entry_probe * seed_centers.size());
        keep_smallest(bucket_scores, bucket_keep);

        std::vector<PID> candidate_ids;
        candidate_ids.reserve(bucket_keep * config_.micro_entry_bucket_take);
        auto append_unique = [&](PID id) {
            if (std::find(candidate_ids.begin(), candidate_ids.end(), id) ==
                candidate_ids.end()) {
                candidate_ids.push_back(id);
            }
        };
        for (const auto& bucket_score : bucket_scores) {
            const auto& entries = micro_entry_buckets_[bucket_score.id];
            for (PID id : entries) {
                append_unique(id);
            }
        }
        stats.entry_points_before_cap = candidate_ids.size();

        if (!candidate_ids.empty()) {
            std::vector<ScoredPid> scored;
            scored.reserve(candidate_ids.size());
            for (PID id : candidate_ids) {
                scored.push_back({l2_to_point(query, query_sq_norm, id), id});
            }
            stats.exact_distance_evals += scored.size();
            keep_smallest(scored, std::min(config_.init_keep, scored.size()));

            std::vector<PID> seeds;
            seeds.reserve(scored.size());
            for (const auto& cand : scored) {
                seeds.push_back(cand.id);
            }
            stats.entry_points_after_cap = seeds.size();
            stats.init_size = seeds.size();
            return seeds;
        }
    }

    if (config_.use_center_anchor_seeds && best_center < center_anchor_pool_.size()) {
        stats.mode = SeedMode::CenterAnchor;
        stats.init_mode = SeedMode::CenterAnchor;

        std::vector<PID> anchor_ids;
        std::vector<PID> seed_centers;
        auto append_unique = [&](PID id) {
            if (std::find(anchor_ids.begin(), anchor_ids.end(), id) == anchor_ids.end()) {
                anchor_ids.push_back(id);
            }
        };
        auto append_center_unique = [&](PID center_id) {
            if (center_id >= center_anchor_pool_.size()) {
                return;
            }
            if (std::find(seed_centers.begin(), seed_centers.end(), center_id) ==
                seed_centers.end()) {
                seed_centers.push_back(center_id);
            }
        };
        auto append_pool = [&](PID center_id) {
            if (center_id >= center_anchor_pool_.size()) {
                return;
            }
            const auto& pool = center_anchor_pool_[center_id];
            stats.entry_points_raw += pool.size();
            for (PID id : pool) {
                append_unique(id);
            }
        };

        append_center_unique(best_center);
        for (PID center_id : candidate_centers) {
            append_center_unique(center_id);
            if (seed_centers.size() >= config_.center_anchor_center_keep) {
                break;
            }
        }
        size_t primary_center_count = seed_centers.size();
        for (size_t ci = 0; ci < primary_center_count; ++ci) {
            PID center_id = seed_centers[ci];
            size_t neighbor_count = std::min(
                config_.center_anchor_neighbor_centers,
                center_id < center_neighbors_.size() ? center_neighbors_[center_id].size() : 0
            );
            for (size_t i = 0; i < neighbor_count; ++i) {
                append_center_unique(center_neighbors_[center_id][i]);
            }
        }
        for (PID center_id : seed_centers) {
            append_pool(center_id);
        }

        stats.entry_points_before_cap = anchor_ids.size();
        if (!anchor_ids.empty()) {
            std::vector<ScoredPid> scored;
            scored.reserve(anchor_ids.size());
            for (PID id : anchor_ids) {
                scored.push_back({l2_to_point(query, query_sq_norm, id), id});
            }
            stats.exact_distance_evals += scored.size();
            keep_smallest(scored, std::min(config_.center_anchor_take, scored.size()));

            std::vector<PID> seeds;
            seeds.reserve(scored.size());
            for (const auto& cand : scored) {
                seeds.push_back(cand.id);
            }
            stats.entry_points_after_cap = seeds.size();
            stats.init_size = seeds.size();
            return seeds;
        }
    }

    stats.mode = SeedMode::RabitqTopN;
    stats.init_mode = SeedMode::RabitqTopN;
    const auto& topn = center_topn_[best_center];
    const CodeBank& bank = topn_codes_[best_center];
    const BatchCodeBank& batch_bank = topn_batch_codes_[best_center];
    size_t probe = config_.center_topn_probe == 0
                       ? topn.size()
                       : std::min(config_.center_topn_probe, topn.size());
    std::vector<ScoredPid> scored;
    size_t coarse_keep = std::min(config_.center_topn_coarse_keep, probe);
    scored.reserve(coarse_keep);
    float center_norm = std::sqrt(std::max(best_center_dist2, 0.0F));
    stats.entry_points_raw = topn.size();
    stats.rabitq_topn_coarse_evals += probe;
    stats.topn_coarse_estimates = probe;

    std::vector<ScoredPid> coarse;
    if (coarse_keep < probe && batch_bank.batch_count > 0) {
        coarse.reserve(coarse_keep);
        SplitBatchQuery<float> batch_query(
            query_wrapper.rotated_query(), padded_dim_, ex_bits_, METRIC_L2
        );
        batch_query.set_g_add(center_norm);

        std::array<float, fastscan::kBatchSize> est_dist{};
        std::array<float, fastscan::kBatchSize> low_dist{};
        std::array<float, fastscan::kBatchSize> ip_x0_qr{};
        size_t probe_batches = div_round_up(probe, fastscan::kBatchSize);
        for (size_t batch_id = 0; batch_id < probe_batches; ++batch_id) {
            split_batch_estdist(
                batch_bank.batch(batch_id),
                batch_query,
                padded_dim_,
                est_dist.data(),
                low_dist.data(),
                ip_x0_qr.data(),
                true
            );
            size_t offset = batch_id * fastscan::kBatchSize;
            size_t count = std::min(fastscan::kBatchSize, probe - offset);
            for (size_t j = 0; j < count; ++j) {
                push_topk_smallest(
                    coarse, {est_dist[j], static_cast<PID>(offset + j)}, coarse_keep
                );
            }
        }
        finish_topk_smallest(coarse);
    } else {
        coarse.reserve(probe);
        for (size_t i = 0; i < probe; ++i) {
            coarse.push_back({0.0F, static_cast<PID>(i)});
        }
    }

    for (const auto& coarse_cand : coarse) {
        size_t i = coarse_cand.id;
        float est = estimate_code(
            bank.bin(i), bank.ex(i), query_wrapper, best_center_dist2, center_norm
        );
        scored.push_back({est, topn[i]});
    }
    stats.topn_full_estimates = scored.size();
    stats.rabitq_topn_refine_evals += scored.size();
    stats.entry_points_before_cap = scored.size();

    keep_smallest(scored, std::min(config_.init_keep, scored.size()));
    stats.exact_distance_evals += scored.size();
    for (auto& cand : scored) {
        cand.distance = l2_to_point(query, query_sq_norm, cand.id);
    }
    std::sort(scored.begin(), scored.end());

    std::vector<PID> seeds;
    seeds.reserve(scored.size());
    for (const auto& cand : scored) {
        seeds.push_back(cand.id);
    }
    stats.entry_points_after_cap = seeds.size();
    stats.init_size = seeds.size();
    return seeds;
}

inline std::vector<PID> TreeRaBitQGraphIndex::initial_seeds_fast(
    const float* query,
    float query_sq_norm,
    const SplitSingleQuery<float>& query_wrapper,
    PID best_center,
    const std::vector<PID>& candidate_centers,
    float best_center_dist2
) const {
    if (best_center_dist2 <= center_pool_trigger_d2_[best_center]) {
        const auto& pool = center_real_pool_[best_center];
        size_t take = std::min(config_.center_real_pool_take, pool.size());
        return std::vector<PID>(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(take));
    }

    if (config_.use_micro_entry_seeds && best_center < num_centers() &&
        micro_entry_bucket_count_ > 0) {
        std::vector<PID> seed_centers;
        auto append_center_unique = [&](PID center_id) {
            if (center_id >= num_centers()) {
                return;
            }
            if (std::find(seed_centers.begin(), seed_centers.end(), center_id) ==
                seed_centers.end()) {
                seed_centers.push_back(center_id);
            }
        };

        append_center_unique(best_center);
        for (PID center_id : candidate_centers) {
            append_center_unique(center_id);
            if (seed_centers.size() >= config_.micro_entry_center_keep) {
                break;
            }
        }

        std::vector<ScoredPid> bucket_scores;
        bucket_scores.reserve(seed_centers.size() * micro_entry_bucket_count_);
        for (PID center_id : seed_centers) {
            size_t query_bucket =
                micro_entry_bucket_for_query(center_id, query_wrapper.rotated_query());
            for (size_t bucket = 0; bucket < micro_entry_bucket_count_; ++bucket) {
                size_t offset = micro_entry_bucket_offset(center_id, bucket);
                if (micro_entry_counts_[offset] == 0) {
                    continue;
                }
                float d = euclidean_sqr_fast(
                    query, micro_entry_centroids_.data() + (offset * dim_), dim_
                );
                if (bucket == query_bucket) {
                    d *= 0.95F;
                }
                bucket_scores.push_back({d, static_cast<PID>(offset)});
            }
        }

        size_t bucket_keep =
            std::min(bucket_scores.size(), config_.micro_entry_probe * seed_centers.size());
        keep_smallest(bucket_scores, bucket_keep);

        std::vector<PID> candidate_ids;
        candidate_ids.reserve(bucket_keep * config_.micro_entry_bucket_take);
        auto append_unique = [&](PID id) {
            if (std::find(candidate_ids.begin(), candidate_ids.end(), id) ==
                candidate_ids.end()) {
                candidate_ids.push_back(id);
            }
        };
        for (const auto& bucket_score : bucket_scores) {
            const auto& entries = micro_entry_buckets_[bucket_score.id];
            for (PID id : entries) {
                append_unique(id);
            }
        }

        if (!candidate_ids.empty()) {
            std::vector<ScoredPid> scored;
            scored.reserve(candidate_ids.size());
            for (PID id : candidate_ids) {
                scored.push_back({l2_to_point(query, query_sq_norm, id), id});
            }
            keep_smallest(scored, std::min(config_.init_keep, scored.size()));

            std::vector<PID> seeds;
            seeds.reserve(scored.size());
            for (const auto& cand : scored) {
                seeds.push_back(cand.id);
            }
            return seeds;
        }
    }

    if (config_.use_center_anchor_seeds && best_center < center_anchor_pool_.size()) {
        std::vector<PID> anchor_ids;
        std::vector<PID> seed_centers;
        auto append_unique = [&](PID id) {
            if (std::find(anchor_ids.begin(), anchor_ids.end(), id) == anchor_ids.end()) {
                anchor_ids.push_back(id);
            }
        };
        auto append_center_unique = [&](PID center_id) {
            if (center_id >= center_anchor_pool_.size()) {
                return;
            }
            if (std::find(seed_centers.begin(), seed_centers.end(), center_id) ==
                seed_centers.end()) {
                seed_centers.push_back(center_id);
            }
        };
        auto append_pool = [&](PID center_id) {
            if (center_id >= center_anchor_pool_.size()) {
                return;
            }
            for (PID id : center_anchor_pool_[center_id]) {
                append_unique(id);
            }
        };

        append_center_unique(best_center);
        for (PID center_id : candidate_centers) {
            append_center_unique(center_id);
            if (seed_centers.size() >= config_.center_anchor_center_keep) {
                break;
            }
        }
        size_t primary_center_count = seed_centers.size();
        for (size_t ci = 0; ci < primary_center_count; ++ci) {
            PID center_id = seed_centers[ci];
            size_t neighbor_count = std::min(
                config_.center_anchor_neighbor_centers,
                center_id < center_neighbors_.size() ? center_neighbors_[center_id].size() : 0
            );
            for (size_t i = 0; i < neighbor_count; ++i) {
                append_center_unique(center_neighbors_[center_id][i]);
            }
        }
        for (PID center_id : seed_centers) {
            append_pool(center_id);
        }

        if (!anchor_ids.empty()) {
            std::vector<ScoredPid> scored;
            scored.reserve(anchor_ids.size());
            for (PID id : anchor_ids) {
                scored.push_back({l2_to_point(query, query_sq_norm, id), id});
            }
            keep_smallest(scored, std::min(config_.center_anchor_take, scored.size()));

            std::vector<PID> seeds;
            seeds.reserve(scored.size());
            for (const auto& cand : scored) {
                seeds.push_back(cand.id);
            }
            return seeds;
        }
    }

    const auto& topn = center_topn_[best_center];
    const CodeBank& bank = topn_codes_[best_center];
    const BatchCodeBank& batch_bank = topn_batch_codes_[best_center];
    size_t probe = config_.center_topn_probe == 0
                       ? topn.size()
                       : std::min(config_.center_topn_probe, topn.size());
    size_t coarse_keep = std::min(config_.center_topn_coarse_keep, probe);
    float center_norm = std::sqrt(std::max(best_center_dist2, 0.0F));

    std::vector<ScoredPid> coarse;
    if (coarse_keep < probe && batch_bank.batch_count > 0) {
        coarse.reserve(coarse_keep);
        SplitBatchQuery<float> batch_query(
            query_wrapper.rotated_query(), padded_dim_, ex_bits_, METRIC_L2
        );
        batch_query.set_g_add(center_norm);

        std::array<float, fastscan::kBatchSize> est_dist{};
        std::array<float, fastscan::kBatchSize> low_dist{};
        std::array<float, fastscan::kBatchSize> ip_x0_qr{};
        size_t probe_batches = div_round_up(probe, fastscan::kBatchSize);
        for (size_t batch_id = 0; batch_id < probe_batches; ++batch_id) {
            split_batch_estdist(
                batch_bank.batch(batch_id),
                batch_query,
                padded_dim_,
                est_dist.data(),
                low_dist.data(),
                ip_x0_qr.data(),
                true
            );
            size_t offset = batch_id * fastscan::kBatchSize;
            size_t count = std::min(fastscan::kBatchSize, probe - offset);
            for (size_t j = 0; j < count; ++j) {
                push_topk_smallest(
                    coarse, {est_dist[j], static_cast<PID>(offset + j)}, coarse_keep
                );
            }
        }
        finish_topk_smallest(coarse);
    } else {
        coarse.reserve(probe);
        for (size_t i = 0; i < probe; ++i) {
            coarse.push_back({0.0F, static_cast<PID>(i)});
        }
    }

    std::vector<ScoredPid> scored;
    scored.reserve(coarse.size());
    for (const auto& coarse_cand : coarse) {
        size_t i = coarse_cand.id;
        float est = estimate_code(
            bank.bin(i), bank.ex(i), query_wrapper, best_center_dist2, center_norm
        );
        scored.push_back({est, topn[i]});
    }

    keep_smallest(scored, std::min(config_.init_keep, scored.size()));
    for (auto& cand : scored) {
        cand.distance = l2_to_point(query, query_sq_norm, cand.id);
    }
    std::sort(scored.begin(), scored.end());

    std::vector<PID> seeds;
    seeds.reserve(scored.size());
    for (const auto& cand : scored) {
        seeds.push_back(cand.id);
    }
    return seeds;
}

inline float TreeRaBitQGraphIndex::estimate_code(
    const char* bin_data,
    const char* ex_data,
    const SplitSingleQuery<float>& query_wrapper,
    float g_add,
    float g_error
) const {
    float ip_x0_qr = 0;
    float est_dist = 0;
    float low_dist = 0;
    if (ex_bits_ == 0) {
        split_single_estdist(
            bin_data, query_wrapper, padded_dim_, ip_x0_qr, est_dist, low_dist, g_add, g_error
        );
    } else {
        split_single_fulldist(
            bin_data,
            ex_data,
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            est_dist,
            low_dist,
            ip_x0_qr,
            g_add,
            g_error
        );
    }
    return est_dist;
}

inline float TreeRaBitQGraphIndex::estimate_code_onebit(
    const char* bin_data,
    const SplitSingleQuery<float>& query_wrapper,
    float g_add,
    float g_error
) const {
    float ip_x0_qr = 0;
    float est_dist = 0;
    float low_dist = 0;
    split_single_estdist(
        bin_data, query_wrapper, padded_dim_, ip_x0_qr, est_dist, low_dist, g_add, g_error
    );
    return est_dist;
}

inline uint32_t TreeRaBitQGraphIndex::next_visit_epoch() const {
    ++visit_epoch_;
    if (visit_epoch_ == 0) {
        std::fill(visit_marks_.begin(), visit_marks_.end(), 0);
        visit_epoch_ = 1;
    }
    return visit_epoch_;
}

inline void TreeRaBitQGraphIndex::keep_smallest(
    std::vector<ScoredPid>& items, size_t keep
) {
    if (keep >= items.size()) {
        std::sort(items.begin(), items.end());
        return;
    }
    if (keep == 0) {
        items.clear();
        return;
    }
    auto nth = items.begin() + static_cast<std::ptrdiff_t>(keep);
    std::nth_element(items.begin(), nth, items.end());
    items.resize(keep);
    std::sort(items.begin(), items.end());
}

inline void TreeRaBitQGraphIndex::push_topk_smallest(
    std::vector<ScoredPid>& heap, ScoredPid item, size_t keep
) {
    if (keep == 0) {
        return;
    }
    if (heap.size() < keep) {
        heap.push_back(item);
        std::push_heap(heap.begin(), heap.end());
        return;
    }
    if (item.distance < heap.front().distance) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = item;
        std::push_heap(heap.begin(), heap.end());
    }
}

inline void TreeRaBitQGraphIndex::finish_topk_smallest(
    std::vector<ScoredPid>& heap
) {
    std::sort_heap(heap.begin(), heap.end());
}

}  // namespace rabitqlib::treegraph
