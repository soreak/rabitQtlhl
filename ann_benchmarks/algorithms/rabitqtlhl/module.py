from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ..base.module import BaseANN
except Exception:  # pragma: no cover - allows standalone smoke tests outside ann-benchmarks
    class BaseANN:  # type: ignore
        def get_additional(self):
            return {}


class _Params(ctypes.Structure):
    _fields_ = [
        ("n_centers", ctypes.c_size_t),
        ("center_leaf_min_size", ctypes.c_size_t),
        ("center_scan_keep", ctypes.c_size_t),
        ("exact_center_keep", ctypes.c_size_t),
        ("center_refine_neighbor_scan", ctypes.c_size_t),
        ("center_entry_mode", ctypes.c_int),
        ("center_real_pool_size", ctypes.c_size_t),
        ("center_real_pool_take", ctypes.c_size_t),
        ("center_real_pool_trigger_topk", ctypes.c_size_t),
        ("center_topn_scan", ctypes.c_size_t),
        ("center_topn_probe", ctypes.c_size_t),
        ("center_topn_coarse_keep", ctypes.c_size_t),
        ("init_keep", ctypes.c_size_t),
        ("use_center_anchor_seeds", ctypes.c_int),
        ("center_anchor_pool_size", ctypes.c_size_t),
        ("center_anchor_take", ctypes.c_size_t),
        ("center_anchor_neighbor_centers", ctypes.c_size_t),
        ("center_anchor_center_keep", ctypes.c_size_t),
        ("use_micro_entry_seeds", ctypes.c_int),
        ("micro_entry_bits", ctypes.c_size_t),
        ("micro_entry_bucket_take", ctypes.c_size_t),
        ("micro_entry_probe", ctypes.c_size_t),
        ("micro_entry_center_keep", ctypes.c_size_t),
        ("graph_degree", ctypes.c_size_t),
        ("ef_search", ctypes.c_size_t),
        ("graph_build_intra_candidates", ctypes.c_size_t),
        ("graph_build_cross_candidates", ctypes.c_size_t),
        ("graph_build_projection_dims", ctypes.c_size_t),
        ("graph_build_center_neighbors", ctypes.c_size_t),
        ("graph_search_use_rabitq", ctypes.c_int),
        ("graph_search_full_rabitq", ctypes.c_int),
        ("graph_rerank_candidates", ctypes.c_size_t),
        ("graph_early_stop", ctypes.c_int),
        ("graph_search_neighbor_cap", ctypes.c_size_t),
        ("graph_lazy_center_distance", ctypes.c_int),
        ("graph_distance_use_norm_dot", ctypes.c_int),
        ("graph_build_bridge_edges", ctypes.c_int),
        ("graph_bridge_center_neighbors", ctypes.c_size_t),
        ("graph_bridge_points_per_center", ctypes.c_size_t),
        ("graph_bridge_candidate_scan", ctypes.c_size_t),
        ("graph_query_adjacency_order", ctypes.c_int),
        ("graph_reorder_by_center", ctypes.c_int),
        ("rabitq_total_bits", ctypes.c_size_t),
        ("random_seed", ctypes.c_size_t),
    ]


def _load_library() -> ctypes.CDLL:
    here = Path(__file__).resolve().parent
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(here))
    env_path = os.environ.get("RABITQTLHL_LIB")
    candidates = [
        Path(env_path) if env_path else None,
        Path("/opt/rabitqtlhl/librabitqtlhl.so"),
        here / "rabitqtlhl.dll",
        here / "librabitqtlhl.so",
        here / "librabitqtlhl.dylib",
        here / "Release" / "rabitqtlhl.dll",
        here / "Debug" / "rabitqtlhl.dll",
    ]
    for path in candidates:
        if path is not None and path.exists():
            lib = ctypes.CDLL(str(path))
            break
    else:
        names = ", ".join(str(path) for path in candidates if path is not None)
        raise ImportError(
            "Could not find the RaBitQ-TLHL shared library. "
            f"Expected one of: {names}. Build it with CMake first."
        )

    lib.rabitqtlhl_default_params.restype = _Params
    lib.rabitqtlhl_create.argtypes = [ctypes.POINTER(_Params)]
    lib.rabitqtlhl_create.restype = ctypes.c_void_p
    lib.rabitqtlhl_free.argtypes = [ctypes.c_void_p]
    lib.rabitqtlhl_free.restype = None
    lib.rabitqtlhl_build.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.rabitqtlhl_build.restype = ctypes.c_int
    lib.rabitqtlhl_set_query_params.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.rabitqtlhl_set_query_params.restype = ctypes.c_int
    lib.rabitqtlhl_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.rabitqtlhl_search.restype = ctypes.c_int
    lib.rabitqtlhl_search_many.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.rabitqtlhl_search_many.restype = ctypes.c_int
    lib.rabitqtlhl_graph_edges.argtypes = [ctypes.c_void_p]
    lib.rabitqtlhl_graph_edges.restype = ctypes.c_size_t
    lib.rabitqtlhl_avg_graph_degree.argtypes = [ctypes.c_void_p]
    lib.rabitqtlhl_avg_graph_degree.restype = ctypes.c_double
    lib.rabitqtlhl_last_error.argtypes = [ctypes.c_void_p]
    lib.rabitqtlhl_last_error.restype = ctypes.c_char_p
    return lib


_LIB = _load_library()


class RabitQTLHL(BaseANN):
    def __init__(self, metric: str, method_param: dict[str, Any]):
        if metric != "euclidean":
            raise NotImplementedError("RaBitQ-TLHL currently supports only euclidean distance")

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.metric = metric
        self.method_param = dict(method_param)
        self.params = _LIB.rabitqtlhl_default_params()
        self.handle = None
        self._batch_results = None
        self._built = False

        self._apply_method_params()
        self.ef_search = int(self.params.ef_search)
        self.neighbor_cap = int(self.params.graph_search_neighbor_cap)
        self.rerank = int(self.params.graph_rerank_candidates)
        self._update_name()

    def _get(self, *names: str, default: Any = None) -> Any:
        for name in names:
            if name in self.method_param:
                return self.method_param[name]
        return default

    def _set_size(self, field: str, *names: str) -> None:
        value = self._get(*names, default=None)
        if value is not None:
            setattr(self.params, field, int(value))

    def _set_bool(self, field: str, *names: str) -> None:
        value = self._get(*names, default=None)
        if value is not None:
            setattr(self.params, field, 1 if bool(value) else 0)

    def _apply_method_params(self) -> None:
        self._set_size("n_centers", "nCenters", "n_centers")
        self._set_size("center_leaf_min_size", "centerLeafMinSize", "center_leaf_min_size")
        self._set_size("center_scan_keep", "centerScanKeep", "center_scan_keep")
        self._set_size("exact_center_keep", "exactCenterKeep", "exact_center_keep")
        self._set_size(
            "center_refine_neighbor_scan",
            "centerRefineNeighborScan",
            "center_refine_neighbor_scan",
        )
        self._set_size("center_entry_mode", "centerEntryMode", "center_entry_mode")
        self._set_size("center_topn_scan", "centerTopNScan", "center_topn_scan")
        self._set_size(
            "center_topn_coarse_keep", "centerTopNCoarseKeep", "center_topn_coarse_keep"
        )
        self._set_size("init_keep", "initKeep", "init_keep")

        self._set_size("graph_degree", "graphDegree", "graph_degree")
        self._set_size("graph_build_intra_candidates", "graphBuildIntra", "graph_build_intra_candidates")
        self._set_size("graph_build_cross_candidates", "graphBuildCross", "graph_build_cross_candidates")
        self._set_size("graph_build_center_neighbors", "graphBuildCenterNeighbors", "graph_build_center_neighbors")
        self._set_size("ef_search", "efSearch", "ef_search")
        self._set_size("graph_search_neighbor_cap", "neighborCap", "graph_search_neighbor_cap")
        self._set_size("graph_rerank_candidates", "rerank", "graph_rerank_candidates")

        self._set_bool("graph_search_use_rabitq", "graphSearchUseRaBitQ", "graph_search_use_rabitq")
        self._set_bool("graph_build_bridge_edges", "bridgeEdges", "graph_build_bridge_edges")
        self._set_bool("graph_query_adjacency_order", "queryAdjacencyOrder", "graph_query_adjacency_order")
        self._set_bool("graph_reorder_by_center", "reorderByCenter", "graph_reorder_by_center")
        self._set_size("graph_bridge_center_neighbors", "bridgeCenterNeighbors", "graph_bridge_center_neighbors")
        self._set_size("graph_bridge_points_per_center", "bridgePointsPerCenter", "graph_bridge_points_per_center")
        self._set_size("graph_bridge_candidate_scan", "bridgeCandidateScan", "graph_bridge_candidate_scan")

        self._set_bool("use_micro_entry_seeds", "useMicroEntry", "use_micro_entry_seeds")
        self._set_size("micro_entry_bits", "microEntryBits", "micro_entry_bits")
        self._set_size("micro_entry_bucket_take", "microEntryBucketTake", "micro_entry_bucket_take")
        self._set_size("micro_entry_probe", "microEntryProbe", "micro_entry_probe")
        self._set_size("micro_entry_center_keep", "microEntryCenterKeep", "micro_entry_center_keep")

        self._set_size("random_seed", "randomState", "random_state")

    def _last_error(self) -> str:
        if self.handle is None:
            return "null handle"
        msg = _LIB.rabitqtlhl_last_error(self.handle)
        return msg.decode("utf-8", errors="replace") if msg else "unknown error"

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(self._last_error())

    def _update_name(self) -> None:
        self.name = (
            "RaBitQ-TLHL "
            f"(nCenters={int(self.params.n_centers)}, "
            f"degree={int(self.params.graph_degree)}, "
            f"ef={self.ef_search}, "
            f"cap={self.neighbor_cap}, "
            f"rerank={self.rerank}, "
            f"micro={int(self.params.use_micro_entry_seeds)}, "
            f"bridge={int(self.params.graph_build_bridge_edges)}, "
            f"reorder={int(self.params.graph_reorder_by_center)})"
        )

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.handle = _LIB.rabitqtlhl_create(ctypes.byref(self.params))
        if not self.handle:
            raise RuntimeError("rabitqtlhl_create failed")
        self._check(
            _LIB.rabitqtlhl_build(
                self.handle,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(X.shape[0]),
                int(X.shape[1]),
            )
        )
        self._built = True
        self.set_query_arguments(self.ef_search, self.neighbor_cap, self.rerank)
        if X.shape[0] > 0:
            self.query(X[0], 1)

    def set_query_arguments(self, ef_search, neighbor_cap=None, rerank=None):
        self.ef_search = int(ef_search)
        self.neighbor_cap = int(self.neighbor_cap if neighbor_cap is None else neighbor_cap)
        self.rerank = int(self.rerank if rerank is None else rerank)
        if self.handle is not None:
            self._check(
                _LIB.rabitqtlhl_set_query_params(
                    self.handle, self.ef_search, self.neighbor_cap, self.rerank
                )
            )
        self._update_name()

    def query(self, v, n):
        if not self._built or self.handle is None:
            raise RuntimeError("Index has not been built. Call fit() first.")
        q = np.ascontiguousarray(np.asarray(v, dtype=np.float32).reshape(-1))
        out = np.empty(int(n), dtype=np.uint32)
        self._check(
            _LIB.rabitqtlhl_search(
                self.handle,
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(n),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
        )
        return out.astype(np.int32, copy=False)

    def batch_query(self, X, n):
        if not self._built or self.handle is None:
            raise RuntimeError("Index has not been built. Call fit() first.")
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty((X.shape[0], int(n)), dtype=np.uint32)
        self._check(
            _LIB.rabitqtlhl_search_many(
                self.handle,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(X.shape[0]),
                int(n),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
        )
        self._batch_results = out.astype(np.int32, copy=False)

    def get_batch_results(self):
        if self._batch_results is None:
            raise RuntimeError("No batch query results available. Call batch_query() first.")
        return self._batch_results

    def get_additional(self):
        if self.handle is None:
            return {}
        return {
            "graph_edges": int(_LIB.rabitqtlhl_graph_edges(self.handle)),
            "avg_graph_degree": float(_LIB.rabitqtlhl_avg_graph_degree(self.handle)),
            "ef_search": self.ef_search,
            "neighbor_cap": self.neighbor_cap,
            "rerank": self.rerank,
        }

    def freeIndex(self):
        self.done()

    def done(self):
        if self.handle is not None:
            _LIB.rabitqtlhl_free(self.handle)
        self.handle = None
        self._built = False
        self._batch_results = None

    def __del__(self):
        try:
            self.done()
        except Exception:
            pass
