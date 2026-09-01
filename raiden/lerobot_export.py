"""Export raw Raiden recordings directly to a LeRobot v3.0 dataset.

Input:  a raw recording directory produced by ``rd record`` — ``cameras/*.bag``
        (or ``*.svo2``) plus ``robot_data.npz`` and ``metadata.json``.
Output: a LeRobot v3.0 dataset (parquet + mp4 + meta/).

Unlike ``rd shardify``, this does **not** require ``rd convert`` to have run
first.  Camera frames are decoded and piped straight into the video encoders,
so the intermediate UnifiedDataset layer (lossless PNG + ``.npz`` depth +
per-frame pickles, ~250 MB per episode) is never written to disk.

All lowdim quantities — forward kinematics, hand-eye wrist extrinsics,
timestamp interpolation, camera alignment and trimming — are produced by
calling into :mod:`raiden.converter` itself, so the numbers are identical to
what ``rd convert`` would have written.  ``converter.py`` is not modified.

The LeRobot v3.0 container (parquet schema, ``meta/info.json``,
``meta/episodes``, ``meta/tasks.parquet``, ``meta/stats.json``, video packing)
is written directly with ``pyarrow`` and ``av``.  The ``lerobot`` package is
*not* imported: it requires Python >= 3.12 while Raiden is pinned to 3.11 by
the ``pyzed`` cp311 wheel.  The emitted layout is byte-compatible with
``lerobot`` 0.6.1 / ``CODEBASE_VERSION == "v3.0"``.

Usage::

    rd export_lerobot
    rd export_lerobot --output-dir data/lerobot --no-depth
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from raiden import converter as _conv
from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.utils import demonstration_status

CODEBASE_VERSION = "v3.0"

# ---------------------------------------------------------------------------
# LeRobot defaults (mirrored from lerobot/configs/video.py @ 0.6.1)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

DEPTH_QUANT_BITS = 12
DEPTH_QMAX = (1 << DEPTH_QUANT_BITS) - 1  # 4095
DEFAULT_DEPTH_MIN = 0.01
DEFAULT_DEPTH_MAX = 10.0
DEFAULT_DEPTH_SHIFT = 3.5

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QUANTILE_KEYS = [f"q{int(q * 100):02d}" for q in QUANTILES]

_MM_PER_M = 1000.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LeRobotExportConfig:
    """Parameters controlling the raw -> LeRobot v3.0 export."""

    output_dir: Path
    repo_id: str = "raiden/dataset"
    robot_type: str = "yam"
    fps: int = 30

    #: Cameras to emit depth video for.  Empty tuple = no depth at all.
    #: Wrist depth is excluded by default: it roughly triples dataset size and
    #: is not consumed by common VLA policies.
    depth_cameras: Tuple[str, ...] = ("scene_camera",)

    #: RGB encoder (LeRobot RGBEncoderConfig defaults).
    rgb_vcodec: str = "libsvtav1"
    rgb_pix_fmt: str = "yuv420p"
    rgb_crf: int = 30
    rgb_gop: int = 2
    rgb_preset: int = 12

    #: Depth encoder (LeRobot DepthEncoderConfig defaults: HEVC Main12 lossless).
    depth_vcodec: str = "libx265"
    depth_pix_fmt: str = "gray12le"
    depth_gop: int = 2
    depth_lossless: bool = True
    depth_crf: Optional[int] = None
    depth_min: float = DEFAULT_DEPTH_MIN
    depth_max: float = DEFAULT_DEPTH_MAX
    depth_shift: float = DEFAULT_DEPTH_SHIFT

    #: Resize RGB/depth to (H, W) before encoding.  None = native resolution.
    resize: Optional[Tuple[int, int]] = None

    #: Number of leading frames per episode to also archive losslessly, for
    #: pose estimation and scene reconstruction.  The video stream is lossy and
    #: range-quantizes depth; these keyframes are the untouched sensor output.
    #: 0 disables.  Written at native resolution regardless of --resize, so the
    #: stored intrinsics apply directly.
    keyframe_count: int = 1

    #: Rename cameras in the output feature keys.
    camera_name_map: Dict[str, str] = field(default_factory=dict)

    chunks_size: int = DEFAULT_CHUNK_SIZE
    data_files_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB
    video_files_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    max_episodes: int = -1

    #: Append to an existing dataset, skipping recordings already exported.
    #: False rebuilds the dataset from scratch.
    incremental: bool = True

    def rgb_options(self) -> Dict[str, str]:
        opts = {"crf": str(self.rgb_crf), "g": str(self.rgb_gop)}
        if self.rgb_vcodec == "libsvtav1":
            opts["preset"] = str(self.rgb_preset)
        return opts

    def depth_options(self) -> Dict[str, str]:
        params = ["log-level=none"]
        opts: Dict[str, str] = {"g": str(self.depth_gop)}
        if self.depth_lossless:
            params.insert(0, "lossless=1")
        elif self.depth_crf is not None:
            opts["crf"] = str(self.depth_crf)
        opts["x265-params"] = ":".join(params)
        return opts


# ---------------------------------------------------------------------------
# Depth quantization — ports lerobot.datasets.depth_utils.quantize_depth
# ---------------------------------------------------------------------------


def quantize_depth_mm(
    depth_mm: np.ndarray,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
) -> np.ndarray:
    """Quantize uint16 millimetre depth to 12-bit log codes (0..DEPTH_QMAX).

    ``depth_min`` / ``depth_max`` / ``shift`` are in **metres**; the input is in
    **millimetres**, matching what the Raiden converter produces.  Math is
    identical to LeRobot's ``quantize_depth(..., input_unit="mm")``.

    Raiden marks missing depth with 0.  ``log(0 + shift) < log(depth_min +
    shift)`` so a zero always clips to code 0, and no real reading reaches code
    0 (the closest observed sample in this dataset is 245 mm).  Code 0 is
    therefore an exact no-data sentinel — see :func:`dequantize_depth_mm`.
    """
    if depth_min + shift <= 0:
        raise ValueError(f"depth_min + shift must be > 0, got {depth_min + shift}")

    d = depth_mm.astype(np.float32)
    lo_u = np.float32(depth_min * _MM_PER_M)
    hi_u = np.float32(depth_max * _MM_PER_M)
    sh_u = np.float32(shift * _MM_PER_M)

    log_min = math.log(float(lo_u + sh_u))
    log_max = math.log(float(hi_u + sh_u))
    norm = (np.log(d + sh_u) - log_min) / (log_max - log_min)
    return np.clip(np.round(norm * DEPTH_QMAX), 0, DEPTH_QMAX).astype(np.uint16)


def dequantize_depth_mm(
    codes: np.ndarray,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
) -> np.ndarray:
    """Inverse of :func:`quantize_depth_mm`.  Code 0 comes back as NaN (no data)."""
    lo_u = depth_min * _MM_PER_M
    hi_u = depth_max * _MM_PER_M
    sh_u = shift * _MM_PER_M
    log_min = math.log(lo_u + sh_u)
    log_max = math.log(hi_u + sh_u)
    out = np.exp((codes.astype(np.float32) / DEPTH_QMAX) * (log_max - log_min) + log_min) - sh_u
    return np.where(codes == 0, np.nan, out).astype(np.float32)


# ---------------------------------------------------------------------------
# Video stream writer
# ---------------------------------------------------------------------------


class _VideoStreamWriter:
    """Encodes one video key, packing many episodes into each mp4 file.

    LeRobot v3.0 concatenates episodes into a shared mp4 until it exceeds
    ``video_files_size_in_mb``, then rolls to the next file.  Each episode
    records the (chunk, file, from_timestamp, to_timestamp) span it occupies.
    """

    def __init__(
        self,
        root: Path,
        video_key: str,
        fps: int,
        vcodec: str,
        pix_fmt: str,
        options: Dict[str, str],
        max_size_mb: int,
        chunks_size: int,
        is_depth: bool,
        start_chunk: int = 0,
        start_file: int = 0,
    ) -> None:
        self.root = root
        self.video_key = video_key
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.options = options
        self.max_size_mb = max_size_mb
        self.chunks_size = chunks_size
        self.is_depth = is_depth

        self.chunk_idx = start_chunk
        self.file_idx = start_file
        self._container: Optional[av.container.OutputContainer] = None
        self._stream = None
        self._frames_in_file = 0
        self._size: Optional[Tuple[int, int]] = None  # (H, W)
        self._ep_start_frame = 0
        self._ep_frames = 0

    # -- file lifecycle ----------------------------------------------------

    def _path(self) -> Path:
        return self.root / VIDEO_PATH.format(
            video_key=self.video_key, chunk_index=self.chunk_idx, file_index=self.file_idx
        )

    def _open(self, height: int, width: int) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), "w")
        self._stream = self._container.add_stream(self.vcodec, self.fps, options=self.options)
        self._stream.pix_fmt = self.pix_fmt
        self._stream.width = width
        self._stream.height = height
        self._size = (height, width)
        self._frames_in_file = 0

    def _close(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

    def _roll(self) -> None:
        """Close the current file and advance (chunk, file) indices."""
        self._close()
        self.file_idx += 1
        if self.file_idx >= self.chunks_size:
            self.file_idx = 0
            self.chunk_idx += 1

    # -- episode lifecycle -------------------------------------------------

    def begin_episode(self, height: int, width: int) -> None:
        if self._container is not None and self._size != (height, width):
            raise ValueError(
                f"{self.video_key}: frame size changed from {self._size} to "
                f"{(height, width)} mid-dataset. LeRobot requires one resolution "
                "per video stream — export these recordings separately or pass "
                "--resize."
            )
        if self._container is not None and self._path().stat().st_size / 1e6 >= self.max_size_mb:
            self._roll()
        if self._container is None:
            self._open(height, width)
        self._ep_start_frame = self._frames_in_file
        self._ep_frames = 0

    def write(self, image: np.ndarray) -> None:
        """Write one frame.  RGB: HxWx3 uint8 (RGB order).  Depth: HxW uint16 codes."""
        if self._container is None:
            h, w = image.shape[:2]
            self._open(h, w)
        if self.is_depth:
            frame = self._depth_frame(image)
        else:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._frames_in_file += 1
        self._ep_frames += 1

    def _depth_frame(self, codes: np.ndarray) -> av.VideoFrame:
        h, w = codes.shape[:2]
        frame = av.VideoFrame(w, h, self.pix_fmt)
        plane = frame.planes[0]
        buf = np.frombuffer(plane, dtype=np.uint16).reshape(h, plane.line_size // 2)
        buf[:, :w] = codes
        return frame

    def end_episode(self) -> Dict[str, Any]:
        """Return this episode's span metadata for meta/episodes."""
        return {
            f"videos/{self.video_key}/chunk_index": self.chunk_idx,
            f"videos/{self.video_key}/file_index": self.file_idx,
            f"videos/{self.video_key}/from_timestamp": self._ep_start_frame / self.fps,
            f"videos/{self.video_key}/to_timestamp": (self._ep_start_frame + self._ep_frames)
            / self.fps,
        }

    def finalize(self) -> None:
        self._close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _estimate_num_samples(
    n: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    if n < min_num_samples:
        min_num_samples = n
    return max(min_num_samples, min(int(n**power), max_num_samples))


def sample_indices(n: int) -> np.ndarray:
    """Evenly spaced sample indices — matches lerobot.datasets.compute_stats."""
    return np.round(np.linspace(0, n - 1, _estimate_num_samples(n))).astype(int)


def _stats_from_array(arr: np.ndarray, keepdims_axes: Optional[Tuple[int, ...]] = None) -> Dict:
    """Per-dimension stats in LeRobot's stats.json shape.

    ``arr`` is (N, D) for vectors, or (N, C, H, W) for images with
    ``keepdims_axes=(0, 2, 3)`` so results keep a leading channel axis.
    """
    a = arr.astype(np.float64)
    if keepdims_axes is None:
        axes: Tuple[int, ...] = (0,)
        kd = False
        flat = a
    else:
        axes = keepdims_axes
        kd = True
        flat = np.moveaxis(a, 1, -1).reshape(-1, a.shape[1])  # (N*H*W, C)

    out = {
        "min": np.min(a, axis=axes, keepdims=kd),
        "max": np.max(a, axis=axes, keepdims=kd),
        "mean": np.mean(a, axis=axes, keepdims=kd),
        "std": np.std(a, axis=axes, keepdims=kd),
        "count": np.array([a.shape[0]]),
    }
    q = np.quantile(flat, QUANTILES, axis=0)  # (len(QUANTILES), D)
    for i, key in enumerate(QUANTILE_KEYS):
        out[key] = q[i].reshape(out["mean"].shape) if kd else q[i]
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in out.items()}


def _merge_feature_stats(a: Dict, b: Dict, n1: int, n2: int) -> Dict:
    """Combine two stat dicts for a single feature (parallel mean/variance)."""
    out: Dict[str, Any] = {}
    for key in b:
        if key not in a:
            out[key] = b[key]
            continue
        va = np.array(a[key], dtype=np.float64)
        vb = np.array(b[key], dtype=np.float64)
        if key == "count":
            out[key] = (va + vb).tolist()
        elif key == "min":
            out[key] = np.minimum(va, vb).tolist()
        elif key == "max":
            out[key] = np.maximum(va, vb).tolist()
        elif key == "mean":
            out[key] = ((va * n1 + vb * n2) / (n1 + n2)).tolist()
        elif key == "std":
            m1 = np.array(a["mean"], dtype=np.float64)
            m2 = np.array(b["mean"], dtype=np.float64)
            m = (m1 * n1 + m2 * n2) / (n1 + n2)
            var = (n1 * (va**2 + (m1 - m) ** 2) + n2 * (vb**2 + (m2 - m) ** 2)) / (n1 + n2)
            out[key] = np.sqrt(var).tolist()
        else:
            # Quantiles cannot be combined exactly without the full sample;
            # lerobot's own aggregate_stats uses the same weighted approximation.
            out[key] = ((va * n1 + vb * n2) / (n1 + n2)).tolist()
    return out


def _merge_stats(a: Optional[Dict], b: Dict, weights: Tuple[int, int]) -> Dict:
    """Merge two dataset-level stat dicts, keyed by feature name."""
    if not a:
        return b
    n1, n2 = weights
    out = dict(a)
    for feature, stats in b.items():
        out[feature] = (
            _merge_feature_stats(a[feature], stats, n1, n2) if feature in a else stats
        )
    return out


# ---------------------------------------------------------------------------
# LeRobot v3.0 dataset writer
# ---------------------------------------------------------------------------


def _next_slot(chunk_idx: int, file_idx: int, chunks_size: int) -> Tuple[int, int]:
    """Advance a (chunk, file) pair, rolling into the next chunk when full."""
    file_idx += 1
    if file_idx >= chunks_size:
        file_idx = 0
        chunk_idx += 1
    return chunk_idx, file_idx


class LeRobotV3Writer:
    """Writes a LeRobot v3.0 dataset with pyarrow + av (no lerobot dependency)."""

    def __init__(self, root: Path, cfg: LeRobotExportConfig) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.root.mkdir(parents=True, exist_ok=True)

        self.features: Dict[str, Dict] = {}
        self.video_writers: Dict[str, _VideoStreamWriter] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.tasks: Dict[str, int] = {}
        self.stats: Optional[Dict[str, Dict]] = None
        self.total_frames = 0

        self._data_chunk = 0
        self._data_file = 0
        self._data_tables: List[pa.Table] = []
        self._data_file_frames = 0
        self._schema: Optional[pa.Schema] = None
        self._initialized = False

        #: recording dir (as posix str) -> episode_index, for incremental runs
        self.sources: Dict[str, int] = {}
        self._video_start: Dict[str, Tuple[int, int]] = {}
        self._resumed = False
        if (self.root / "meta" / "info.json").exists():
            if cfg.incremental:
                self._load_existing()
            else:
                self._clear_existing()

    def _clear_existing(self) -> None:
        """Drop a previous export so a rebuild does not leave orphaned files.

        Without this, parquet/mp4 slots written by an earlier run survive with
        nothing in ``meta/episodes`` referencing them — invisible to LeRobot but
        still occupying disk.
        """
        print(f"  rebuilding from scratch: removing previous export in {self.root}")
        for sub in ("data", "videos", "meta"):
            shutil.rmtree(self.root / sub, ignore_errors=True)

    # -- resume ------------------------------------------------------------

    def _load_existing(self) -> None:
        """Adopt an existing dataset so new episodes append instead of replacing it.

        Existing parquet and mp4 files are never reopened: like LeRobot's own
        resume path, the next episode starts a fresh (chunk, file) slot, so
        anything already written stays byte-identical.
        """
        info = json.loads((self.root / "meta" / "info.json").read_text())
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise ValueError(
                f"{self.root} is codebase_version {info.get('codebase_version')!r}, "
                f"expected {CODEBASE_VERSION!r}. Pass incremental=False to rebuild."
            )
        self.features = info["features"]
        self.total_frames = int(info.get("total_frames", 0))

        ep_path = self.root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        if ep_path.exists():
            table = pq.read_table(ep_path).to_pylist()
            self.episodes = table
        tasks_path = self.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            t = pq.read_table(tasks_path).to_pydict()
            self.tasks = dict(zip(t["task"], t["task_index"]))
        stats_path = self.root / "meta" / "stats.json"
        if stats_path.exists():
            self.stats = json.loads(stats_path.read_text())
        src_path = self.root / "meta" / "raiden_sources.json"
        if src_path.exists():
            self.sources = {
                e["recording"]: e["episode_index"]
                for e in json.loads(src_path.read_text())["episodes"]
            }

        # Continue after the last used slot for data and for every video key.
        if self.episodes:
            last = self.episodes[-1]
            self._data_chunk, self._data_file = _next_slot(
                int(last["data/chunk_index"]), int(last["data/file_index"]), self.cfg.chunks_size
            )
            for key, ft in self.features.items():
                if ft["dtype"] != "video":
                    continue
                ck = f"videos/{key}/chunk_index"
                fk = f"videos/{key}/file_index"
                if ck in last:
                    self._video_start[key] = _next_slot(
                        int(last[ck]), int(last[fk]), self.cfg.chunks_size
                    )

        self._schema = self._build_schema()
        self._initialized = True
        self._resumed = True
        self._init_video_writers()
        print(
            f"  resuming dataset: {len(self.episodes)} episode(s), "
            f"{self.total_frames} frame(s) already exported"
        )

    # -- schema ------------------------------------------------------------

    def init_features(self, features: Dict[str, Dict]) -> None:
        """Declare the dataset feature spec.  Must be called before the first episode."""
        if self._initialized:
            if self._resumed:
                self._verify_resumed_schema(features)
            return
        self.features = dict(features)
        # LeRobot appends these bookkeeping columns to every dataset.
        self.features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        self._init_video_writers()
        self._schema = self._build_schema()
        self._initialized = True

    def _init_video_writers(self) -> None:
        for key, ft in self.features.items():
            if ft["dtype"] != "video":
                continue
            is_depth = bool((ft.get("info") or {}).get("is_depth_map"))
            start_chunk, start_file = self._video_start.get(key, (0, 0))
            self.video_writers[key] = _VideoStreamWriter(
                root=self.root,
                video_key=key,
                fps=self.cfg.fps,
                vcodec=self.cfg.depth_vcodec if is_depth else self.cfg.rgb_vcodec,
                pix_fmt=self.cfg.depth_pix_fmt if is_depth else self.cfg.rgb_pix_fmt,
                options=self.cfg.depth_options() if is_depth else self.cfg.rgb_options(),
                max_size_mb=self.cfg.video_files_size_in_mb,
                chunks_size=self.cfg.chunks_size,
                is_depth=is_depth,
                start_chunk=start_chunk,
                start_file=start_file,
            )

    def _verify_resumed_schema(self, features: Dict[str, Dict]) -> None:
        """A resumed dataset must keep the schema its first episode established."""
        bookkeeping = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        existing = set(self.features) - bookkeeping
        incoming = set(features)
        if existing != incoming:
            raise ValueError(
                f"this recording's features do not match the existing dataset at "
                f"{self.root}"
                + (f"; missing {sorted(existing - incoming)}" if existing - incoming else "")
                + (f"; unexpected {sorted(incoming - existing)}" if incoming - existing else "")
                + ". Export it separately, or re-run with incremental=False "
                "(--reexport) to rebuild."
            )

    def _build_schema(self) -> pa.Schema:
        fields = []
        hf_features: Dict[str, Dict] = {}
        for key, ft in self.features.items():
            if ft["dtype"] == "video":
                continue
            shape = tuple(ft["shape"])
            if shape == (1,):
                pa_type = pa.float32() if ft["dtype"] == "float32" else pa.int64()
                hf_features[key] = {"dtype": ft["dtype"], "_type": "Value"}
            elif len(shape) == 1:
                pa_type = pa.list_(pa.float32(), shape[0])
                hf_features[key] = {
                    "feature": {"dtype": ft["dtype"], "_type": "Value"},
                    "length": shape[0],
                    "_type": "List",
                }
            else:
                raise ValueError(
                    f"feature {key!r} has shape {shape}; flatten multi-dimensional "
                    "features to 1-D before export"
                )
            fields.append(pa.field(key, pa_type))
        meta = {b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()}
        return pa.schema(fields, metadata=meta)

    # -- episodes ----------------------------------------------------------

    def begin_episode(self, sizes: Dict[str, Tuple[int, int]]) -> None:
        # The video-key set is fixed by the first episode; a recording with a
        # different camera set cannot share the dataset.
        missing = set(self.video_writers) - set(sizes)
        extra = set(sizes) - set(self.video_writers)
        if missing or extra:
            raise ValueError(
                "this recording's cameras do not match the dataset fixed by "
                "episode 0"
                + (f"; missing {sorted(missing)}" if missing else "")
                + (f"; unexpected {sorted(extra)}" if extra else "")
                + ". Export recordings with different camera sets separately."
            )
        for key, (h, w) in sizes.items():
            self.video_writers[key].begin_episode(h, w)

    def add_video_frame(self, key: str, image: np.ndarray) -> None:
        self.video_writers[key].write(image)

    def end_episode(
        self,
        columns: Dict[str, np.ndarray],
        task: str,
        image_samples: Dict[str, np.ndarray],
        source: Optional[str] = None,
    ) -> None:
        """Commit one episode.

        ``columns`` maps non-video feature name -> (N, D) or (N,) array.
        ``image_samples`` maps video key -> (S, C, H, W) sampled frames for stats.
        """
        ep_idx = len(self.episodes)
        n = len(next(iter(columns.values())))

        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        task_index = self.tasks[task]

        columns = dict(columns)
        columns["timestamp"] = (np.arange(n) / self.cfg.fps).astype(np.float32)
        columns["frame_index"] = np.arange(n, dtype=np.int64)
        columns["episode_index"] = np.full(n, ep_idx, dtype=np.int64)
        columns["index"] = np.arange(self.total_frames, self.total_frames + n, dtype=np.int64)
        columns["task_index"] = np.full(n, task_index, dtype=np.int64)

        # The feature schema is fixed by the first episode.  Recordings that
        # carry different channels (a leader-teleop session has leader_* data, a
        # SpaceMouse one does not) cannot share a dataset — fail loudly rather
        # than dropping columns or raising a bare KeyError downstream.
        expected = {f.name for f in self._schema}
        missing = expected - set(columns)
        extra = {k for k in columns if k not in expected}
        if missing or extra:
            raise ValueError(
                f"episode {ep_idx} does not match the dataset schema fixed by "
                f"episode 0"
                + (f"; missing {sorted(missing)}" if missing else "")
                + (f"; unexpected {sorted(extra)}" if extra else "")
                + ". Recordings with different control modes or camera sets must "
                "be exported as separate datasets."
            )

        ep_stats = {}
        for key, arr in columns.items():
            a = np.asarray(arr)
            ep_stats[key] = _stats_from_array(a if a.ndim > 1 else a.reshape(-1, 1))
        for key, samples in image_samples.items():
            norm = 1.0 if key in self._depth_keys() else 255.0
            ep_stats[key] = _stats_from_array(samples.astype(np.float64) / norm, (0, 2, 3))

        ep_meta = self._write_episode_data(columns, n)
        for key in self.video_writers:
            ep_meta.update(self.video_writers[key].end_episode())

        record = {
            "episode_index": ep_idx,
            "tasks": [task],
            "length": n,
            **ep_meta,
            **{f"stats/{k}/{s}": v for k, sd in ep_stats.items() for s, v in sd.items()},
        }
        self.episodes.append(record)
        if source is not None:
            self.sources[source] = ep_idx
        self.stats = _merge_stats(self.stats, ep_stats, (self.total_frames, n)) if self.stats else ep_stats
        self.total_frames += n

    def _depth_keys(self) -> set:
        return {
            k for k, ft in self.features.items() if (ft.get("info") or {}).get("is_depth_map")
        }

    def _write_episode_data(self, columns: Dict[str, np.ndarray], n: int) -> Dict[str, Any]:
        arrays = []
        for fld in self._schema:
            a = np.asarray(columns[fld.name])
            if pa.types.is_fixed_size_list(fld.type):
                flat = np.ascontiguousarray(a.astype(np.float32)).reshape(-1)
                arrays.append(
                    pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), fld.type.list_size)
                )
            else:
                arrays.append(pa.array(a.reshape(-1), type=fld.type))
        table = pa.Table.from_arrays(arrays, schema=self._schema)

        from_index = self.total_frames
        est_mb = sum(t.nbytes for t in self._data_tables) / 1e6
        if self._data_tables and est_mb >= self.cfg.data_files_size_in_mb:
            self._flush_data()
            self._data_file += 1
            if self._data_file >= self.cfg.chunks_size:
                self._data_file = 0
                self._data_chunk += 1
        self._data_tables.append(table)

        return {
            "data/chunk_index": self._data_chunk,
            "data/file_index": self._data_file,
            "dataset_from_index": from_index,
            "dataset_to_index": from_index + n,
        }

    def _flush_data(self) -> None:
        if not self._data_tables:
            return
        path = self.root / DATA_PATH.format(chunk_index=self._data_chunk, file_index=self._data_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.concat_tables(self._data_tables), path, compression="snappy", use_dictionary=True
        )
        self._data_tables = []

    # -- finalize ----------------------------------------------------------

    def finalize(self) -> None:
        self._flush_data()
        for writer in self.video_writers.values():
            writer.finalize()
        self._write_video_info()
        self._write_info()
        self._write_tasks()
        self._write_episodes()
        self._write_stats()
        self._write_sources()

    def _write_video_info(self) -> None:
        """Probe each encoded stream and fill features[key]['info'], as lerobot does."""
        for key, writer in self.video_writers.items():
            ft = self.features[key]
            is_depth = key in self._depth_keys()
            path = self.root / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)
            codec, pix_fmt, height, width = writer.vcodec, writer.pix_fmt, None, None
            if path.exists():
                with av.open(str(path)) as c:
                    st = c.streams.video[0]
                    # canonical_name normalizes the decoder pyav happens to pick
                    # (e.g. "libdav1d") to the codec family LeRobot records ("av1").
                    codec = getattr(st.codec_context.codec, "canonical_name", None) or (
                        st.codec_context.name
                    )
                    pix_fmt = st.pix_fmt
                    height, width = st.height, st.width
            info: Dict[str, Any] = {}
            if is_depth:
                info["is_depth_map"] = True
                info["depth_unit"] = "mm"
            info.update(
                {
                    "video.height": height if height is not None else ft["shape"][0],
                    "video.width": width if width is not None else ft["shape"][1],
                    "video.codec": codec,
                    "video.pix_fmt": pix_fmt,
                    "video.fps": self.cfg.fps,
                    "video.channels": 1 if is_depth else 3,
                    "has_audio": False,
                    "video.g": writer.options.get("g") and int(writer.options["g"]),
                    "video.crf": int(writer.options["crf"]) if "crf" in writer.options else None,
                    "video.preset": int(writer.options["preset"]) if "preset" in writer.options else None,
                    "video.fast_decode": 0,
                    "video.video_backend": "pyav",
                    "video.extra_options": (
                        {"x265-params": writer.options["x265-params"]}
                        if "x265-params" in writer.options
                        else {}
                    ),
                }
            )
            if is_depth:
                info.update(
                    {
                        "video.depth_min": self.cfg.depth_min,
                        "video.depth_max": self.cfg.depth_max,
                        "video.shift": self.cfg.depth_shift,
                        "video.use_log": True,
                    }
                )
            else:
                info["is_depth_map"] = False
            ft["info"] = info

    def _write_info(self) -> None:
        features = {}
        for key, ft in self.features.items():
            entry = {"dtype": ft["dtype"], "shape": list(ft["shape"]), "names": ft.get("names")}
            if "info" in ft:
                entry["info"] = ft["info"]
            features[key] = entry
        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": self.cfg.fps,
            "features": features,
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "chunks_size": self.cfg.chunks_size,
            "data_files_size_in_mb": self.cfg.data_files_size_in_mb,
            "video_files_size_in_mb": self.cfg.video_files_size_in_mb,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "robot_type": self.cfg.robot_type,
            "splits": {"train": f"0:{len(self.episodes)}"},
        }
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=4)

    def _write_tasks(self) -> None:
        tasks = sorted(self.tasks.items(), key=lambda kv: kv[1])
        # LeRobot reads this with pandas and expects `task` to be the *index*,
        # not a plain column — without the pandas index metadata the frames come
        # back with an integer `task` instead of the instruction string.
        pandas_meta = {
            "index_columns": ["task"],
            "column_indexes": [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ],
            "columns": [
                {
                    "name": "task_index",
                    "field_name": "task_index",
                    "pandas_type": "int64",
                    "numpy_type": "int64",
                    "metadata": None,
                },
                {
                    "name": "task",
                    "field_name": "task",
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                },
            ],
            "attributes": {},
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.3.3",
        }
        schema = pa.schema(
            [pa.field("task_index", pa.int64()), pa.field("task", pa.string())],
            metadata={b"pandas": json.dumps(pandas_meta).encode()},
        )
        table = pa.Table.from_arrays(
            [
                pa.array([i for _, i in tasks], pa.int64()),
                pa.array([t for t, _ in tasks], pa.string()),
            ],
            schema=schema,
        )
        pq.write_table(table, self.root / "meta" / "tasks.parquet")

    def _write_episodes(self) -> None:
        if not self.episodes:
            return
        keys = list(self.episodes[0].keys())
        cols = {k: [ep[k] for ep in self.episodes] for k in keys}
        path = self.root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), path)

    def _write_stats(self) -> None:
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(self.stats or {}, f, indent=4)

    def _write_sources(self) -> None:
        """Record which raw recording produced each episode.

        This is what makes re-running the export incremental — a recording
        already listed here is skipped.  LeRobot ignores the file.
        """
        entries = [
            {"episode_index": idx, "recording": rec}
            for rec, idx in sorted(self.sources.items(), key=lambda kv: kv[1])
        ]
        with open(self.root / "meta" / "raiden_sources.json", "w") as f:
            json.dump({"episodes": entries}, f, indent=2)


# ---------------------------------------------------------------------------
# Raw recording -> frames, reusing raiden.converter for all derived quantities
# ---------------------------------------------------------------------------


def _open_camera(path: Path):
    """Open a .bag or .svo2 recording for playback."""
    if path.suffix == ".bag":
        from raiden.cameras.realsense import RealSenseCamera

        return RealSenseCamera.from_bag(path.stem, path)
    if path.suffix == ".svo2":
        from raiden.cameras.zed import ZedCamera

        return ZedCamera.from_svo(path.stem, path, compute_sdk_depth=True)
    raise ValueError(f"unsupported camera recording: {path}")


def _camera_files(rec_path: Path) -> List[Path]:
    cams = rec_path / "cameras"
    return sorted(list(cams.glob("*.svo2")) + list(cams.glob("*.bag")))


def _scan_timestamps(
    rec_path: Path, cfg: LeRobotExportConfig
) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[dict]], Dict[str, Tuple[int, int]]]:
    """Pass 1 — decode every camera once to collect timestamps and intrinsics.

    Needed up front because ``_align_cameras_by_timestamp`` and
    ``_trim_cameras_to_episode_end`` decide the per-camera trim from the full
    timestamp series.  No pixels are kept.
    """
    timestamps: Dict[str, np.ndarray] = {}
    infos: Dict[str, Optional[dict]] = {}
    sizes: Dict[str, Tuple[int, int]] = {}
    for path in _camera_files(rec_path):
        name = path.stem
        cam = _open_camera(path)
        ts: List[int] = []
        while cam.grab():
            frame = cam.get_frame()
            ts.append(frame.timestamp_ns)
            if name not in sizes:
                sizes[name] = (frame.color.shape[0], frame.color.shape[1])
        try:
            infos[name] = cam.get_camera_info()
        except Exception:
            infos[name] = None
        cam.close()
        timestamps[name] = np.array(ts, dtype=np.int64)
        print(f"    {name}: {len(ts)} frames")
    return timestamps, infos, sizes


def _resolve_trims(
    rec_path: Path,
    rec_meta: dict,
    cam_timestamps: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, np.ndarray], int]:
    """Compute per-camera (start, end) frame trims exactly as ``rd convert`` does.

    ``converter._align_cameras_by_timestamp`` and
    ``converter._trim_cameras_to_episode_end`` are called against a scratch
    directory that contains no image files, so they compute the trims and
    rename/delete nothing.  This keeps a single source of truth for the
    alignment rules without touching converter.py.
    """
    originals = {name: ts.copy() for name, ts in cam_timestamps.items()}
    frame_counts = {name: len(ts) for name, ts in cam_timestamps.items()}

    rs_offsets = rec_meta.get("realsense_clock_offsets", {})
    adjusted: Dict[str, Optional[np.ndarray]] = {}
    for name, ts in cam_timestamps.items():
        offset = rs_offsets.get(name)
        adjusted[name] = ts + int(offset) if offset is not None else ts

    with tempfile.TemporaryDirectory(prefix="raiden_lerobot_") as tmp:
        scratch = Path(tmp)
        for name in cam_timestamps:
            (scratch / "rgb" / name).mkdir(parents=True, exist_ok=True)

        adjusted, frame_counts = _conv._align_cameras_by_timestamp(
            scratch,
            adjusted,
            frame_counts,
            camera_start_times_ns=rec_meta.get("camera_start_times_ns"),
        )
        adjusted, frame_counts = _conv._trim_cameras_to_episode_end(
            scratch,
            adjusted,
            frame_counts,
            rec_meta.get("episode_end_ns"),
        )

    n_min = min(frame_counts.values()) if frame_counts else 0

    trims: Dict[str, Tuple[int, int]] = {}
    final_ts: Dict[str, np.ndarray] = {}
    for name, ts in adjusted.items():
        if ts is None or len(ts) == 0:
            trims[name] = (0, 0)
            final_ts[name] = np.array([], dtype=np.int64)
            continue
        offset = rs_offsets.get(name)
        first = int(ts[0]) - (int(offset) if offset is not None else 0)
        start = int(np.searchsorted(originals[name], first))
        trims[name] = (start, start + n_min)
        final_ts[name] = ts[:n_min]
    return trims, final_ts, n_min


def _build_lowdim_arrays(
    rec_path: Path,
    cameras: List[str],
    n_frames: int,
    camera_infos: Dict[str, Optional[dict]],
    cam_timestamps: Dict[str, np.ndarray],
) -> List[dict]:
    """Run ``converter._build_lowdim`` into a scratch dir and read the frames back.

    Delegating keeps forward kinematics, hand-eye wrist extrinsics, the
    bimanual base transform and the ~100 Hz -> camera-rate interpolation
    identical to ``rd convert``; only the sink differs.  The scratch pickles
    are a few MB and are deleted on exit.
    """
    rec_meta: dict = {}
    meta_path = rec_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            rec_meta = json.load(f)

    robot_data: Optional[Dict[str, np.ndarray]] = None
    robot_path = rec_path / "robot_data.npz"
    if robot_path.exists():
        npz = np.load(robot_path, allow_pickle=False)
        robot_data = {k: npz[k] for k in npz.files}

    calib: Optional[dict] = None
    for candidate in (rec_path / "calibration_results.json", rec_path.parent / "calibration_results.json"):
        if candidate.exists():
            with open(candidate) as f:
                calib = json.load(f)
            break

    T_left_from_right: Optional[np.ndarray] = None
    if calib and "bimanual_transform" in calib:
        mat = calib["bimanual_transform"].get("right_base_to_left_base")
        if mat is not None:
            T_left_from_right = np.linalg.inv(np.array(mat, dtype=np.float32))

    cam_cfg = CameraConfig(CAMERA_CONFIG)
    wrist_joint_keys = {
        name: _conv._ROLE_TO_JOINT_KEY[role]
        for name in cameras
        if (role := cam_cfg.get_role(name)) in _conv._ROLE_TO_JOINT_KEY
    }

    with tempfile.TemporaryDirectory(prefix="raiden_lowdim_") as tmp:
        scratch = Path(tmp)
        _conv._build_lowdim(
            seq_dir=scratch,
            cameras=cameras,
            n_frames=n_frames,
            camera_infos=camera_infos,
            calib=calib,
            robot_data=robot_data,
            rec_meta=rec_meta,
            flip_cameras=_conv._FLIP_CAMERAS,
            right_base_to_left_base=T_left_from_right,
            cam_timestamps={k: v for k, v in cam_timestamps.items()},
            wrist_camera_joint_keys=wrist_joint_keys,
        )
        frames = []
        for i in range(n_frames):
            with open(scratch / "lowdim" / f"{i:010d}.pkl", "rb") as f:
                frames.append(pickle.load(f))
    return frames


#: Extra ``robot_data.npz`` channels carried into the dataset, beyond the four
#: that ``_build_lowdim`` already derives.  Keeping these makes the exported
#: dataset self-contained for demo reconstruction and for policies that consume
#: velocity/effort, so the raw ``.bag`` is no longer the only copy.
_EXTRA_ROBOT_COLUMNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("observation.velocity", ("follower_l_joint_vel", "follower_l_gripper_vel",
                              "follower_r_joint_vel", "follower_r_gripper_vel")),
    ("observation.effort", ("follower_l_joint_eff", "follower_l_gripper_eff",
                            "follower_r_joint_eff", "follower_r_gripper_eff")),
    ("observation.leader.joint_position", ("leader_l_joint_pos", "leader_r_joint_pos")),
    ("observation.leader.joint_velocity", ("leader_l_joint_vel", "leader_r_joint_vel")),
    ("observation.leader.joint_effort", ("leader_l_joint_eff", "leader_r_joint_eff")),
]


def _reference_timestamps(
    cameras: List[str], n_frames: int, cam_timestamps: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    """The camera timestamp grid ``_build_lowdim`` interpolates onto.

    Mirrors the selection rule in ``converter._build_lowdim``: prefer the first
    camera carrying wall-clock timestamps, else the first camera with a matching
    frame count.  :func:`_interp_robot_data` cross-checks the result against the
    converter's own output, so a divergence here fails loudly rather than
    silently producing misaligned columns.
    """
    wall_min = getattr(_conv, "_WALL_CLOCK_MIN_NS", 1_577_836_800_000_000_000)
    for require_wall_clock in (True, False):
        for name in cameras:
            ts = cam_timestamps.get(name)
            if ts is None or len(ts) != n_frames:
                continue
            if require_wall_clock and int(ts[0]) <= wall_min:
                continue
            # Returned as int64.  Nanosecond epoch values need 61 bits, so a
            # float64 round-trip silently rounds them to multiples of 256 —
            # fine for interpolation, lossy as a stored timestamp.
            return ts.astype(np.int64)
    return None


def _interp_robot_data(
    rec_path: Path,
    cameras: List[str],
    n_frames: int,
    cam_timestamps: Dict[str, np.ndarray],
    lowdim: List[dict],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Interpolate the remaining ``robot_data.npz`` channels onto the camera grid.

    Returns ``(columns, widths)``.  Channels absent from the recording (e.g. the
    right arm of a single-arm session) are simply skipped.
    """
    robot_path = rec_path / "robot_data.npz"
    if not robot_path.exists():
        return {}, {}
    npz = np.load(robot_path, allow_pickle=False)
    robot = {k: npz[k] for k in npz.files}

    ref_ts_ns = _reference_timestamps(cameras, n_frames, cam_timestamps)
    robot_ts_raw = robot.get("timestamps")
    if ref_ts_ns is None or robot_ts_raw is None or robot_ts_raw.dtype != np.int64:
        return {}, {}
    ref_ts = ref_ts_ns.astype(np.float64)  # interpolation only
    robot_ts = robot_ts_raw.astype(np.float64)

    def interp(key: str) -> Optional[np.ndarray]:
        arr = robot.get(key)
        if arr is None:
            return None
        if arr.ndim == 1:
            arr = arr[:, None]
        return np.stack(
            [np.interp(ref_ts, robot_ts, arr[:, d]) for d in range(arr.shape[1])], axis=1
        ).astype(np.float32)

    # Self-check: re-deriving a channel the converter also produces must match
    # bit for bit, otherwise our reference grid has drifted from its.
    check = interp("follower_l_joint_pos_7d")
    if check is not None and lowdim:
        ref = np.stack([np.asarray(f["joints"], np.float32) for f in lowdim])[:, : check.shape[1]]
        if not np.array_equal(check, ref):
            raise RuntimeError(
                "interpolation grid disagrees with converter._build_lowdim — "
                "the reference-timestamp rule in _reference_timestamps is stale"
            )

    columns: Dict[str, np.ndarray] = {}
    widths: Dict[str, int] = {}
    for name, keys in _EXTRA_ROBOT_COLUMNS:
        parts = [p for p in (interp(k) for k in keys) if p is not None]
        if parts:
            stacked = np.concatenate(parts, axis=1)
            columns[name] = stacked
            widths[name] = stacked.shape[1]

    columns["observation.timestamp_ns"] = ref_ts_ns
    widths["observation.timestamp_ns"] = 1
    return columns, widths


def _write_keyframe(
    root: Path,
    episode_index: int,
    camera: str,
    frame_index: int,
    color_bgr: np.ndarray,
    depth_mm: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    extrinsics: Optional[np.ndarray],
    timestamp_ns: Optional[int],
) -> None:
    """Archive one frame losslessly outside the video streams.

    The mp4s are lossy (RGB) and range-quantized (depth); pose estimation wants
    the raw sensor values.  These land in ``meta/keyframes/`` which LeRobot
    ignores, so the dataset still loads normally.

    Depth is the original uint16 millimetre array — not the 12-bit log codes —
    so the 0 = no-data sentinel and out-of-range readings survive intact.
    """
    d = root / "meta" / "keyframes" / f"episode-{episode_index:06d}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(d / f"{camera}_{frame_index:04d}_color.png"),
        color_bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if depth_mm is not None:
        cv2.imwrite(str(d / f"{camera}_{frame_index:04d}_depth.png"), depth_mm)
    meta: Dict[str, Any] = {
        "camera": camera,
        "frame_index": frame_index,
        "resolution": [int(color_bgr.shape[0]), int(color_bgr.shape[1])],
        "color": f"{camera}_{frame_index:04d}_color.png",
        "depth": f"{camera}_{frame_index:04d}_depth.png" if depth_mm is not None else None,
        "depth_units": "uint16 millimetres, 0 = no data",
        "color_format": "BGR (OpenCV order), lossless PNG",
    }
    if intrinsics is not None:
        meta["intrinsics"] = np.asarray(intrinsics, np.float64).reshape(3, 3).tolist()
    if extrinsics is not None:
        meta["extrinsics_cam2world"] = np.asarray(extrinsics, np.float64).reshape(4, 4).tolist()
    if timestamp_ns is not None:
        meta["timestamp_ns"] = int(timestamp_ns)
    with open(d / f"{camera}_{frame_index:04d}.json", "w") as f:
        json.dump(meta, f, indent=2)


def _out_name(camera: str, cfg: LeRobotExportConfig) -> str:
    return cfg.camera_name_map.get(camera, camera)


def _build_features(
    cameras: List[str],
    depth_cameras: List[str],
    sizes: Dict[str, Tuple[int, int]],
    lowdim0: dict,
    cfg: LeRobotExportConfig,
    extra_widths: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict]:
    features: Dict[str, Dict] = {}
    for cam in cameras:
        h, w = sizes[cam]
        features[f"observation.images.{_out_name(cam, cfg)}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
        }
    for cam in depth_cameras:
        h, w = sizes[cam]
        features[f"observation.images.depth_{_out_name(cam, cfg)}"] = {
            "dtype": "video",
            "shape": [h, w, 1],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        }
    for key, name in (
        ("joints", "observation.state"),
        ("action_joints", "action"),
        ("actual_poses", "observation.state.ee_pose"),
        ("action", "action.ee_pose"),
    ):
        if key in lowdim0:
            features[name] = {
                "dtype": "float32",
                "shape": [int(np.asarray(lowdim0[key]).size)],
                "names": None,
            }
    for cam in cameras:
        if cam in (lowdim0.get("extrinsics") or {}):
            features[f"observation.extrinsics.{_out_name(cam, cfg)}"] = {
                "dtype": "float32",
                "shape": [16],
                "names": None,
            }
        # Intrinsics are constant per camera, but a parquet column is the only
        # place LeRobot is guaranteed to preserve them across dataset edits —
        # unknown info.json keys are dropped when it rewrites the file.  Snappy
        # collapses the repeated rows to a few bytes.
        if cam in (lowdim0.get("intrinsics") or {}):
            features[f"observation.intrinsics.{_out_name(cam, cfg)}"] = {
                "dtype": "float32",
                "shape": [9],
                "names": None,
            }
    for name, width in (extra_widths or {}).items():
        features[name] = {
            "dtype": "int64" if name.endswith("_ns") else "float32",
            "shape": [width],
            "names": None,
        }
    return features


def export_recording(
    rec_path: Path, writer: LeRobotV3Writer, cfg: LeRobotExportConfig
) -> int:
    """Stream one raw recording into the LeRobot dataset.  Returns frame count."""
    rec_meta: dict = {}
    meta_path = rec_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            rec_meta = json.load(f)

    print(f"\n  {rec_path.parent.name}/{rec_path.name}")
    print("    scanning timestamps ...")
    cam_timestamps, camera_infos, native_sizes = _scan_timestamps(rec_path, cfg)
    if not cam_timestamps:
        print("    no camera recordings — skipped")
        return 0

    trims, final_ts, n_min = _resolve_trims(rec_path, rec_meta, cam_timestamps)
    if n_min == 0:
        print("    no frames survive alignment — skipped")
        return 0

    cameras = list(cam_timestamps.keys())
    depth_cameras = [c for c in cameras if c in cfg.depth_cameras]

    print(f"    building lowdim ({n_min} frames) ...")
    lowdim = _build_lowdim_arrays(rec_path, cameras, n_min, camera_infos, final_ts)

    sizes: Dict[str, Tuple[int, int]] = {
        name: (cfg.resize if cfg.resize else hw) for name, hw in native_sizes.items()
    }

    extra_columns, extra_widths = _interp_robot_data(
        rec_path, cameras, n_min, final_ts, lowdim
    )
    writer.init_features(
        _build_features(cameras, depth_cameras, sizes, lowdim[0], cfg, extra_widths)
    )

    video_sizes = {}
    for cam in cameras:
        video_sizes[f"observation.images.{_out_name(cam, cfg)}"] = sizes[cam]
    for cam in depth_cameras:
        video_sizes[f"observation.images.depth_{_out_name(cam, cfg)}"] = sizes[cam]
    writer.begin_episode(video_sizes)

    stat_idx = set(sample_indices(n_min).tolist())
    samples: Dict[str, List[np.ndarray]] = {k: [] for k in video_sizes}

    for path in _camera_files(rec_path):
        name = path.stem
        start, end = trims[name]
        flip = name in _conv._FLIP_CAMERAS
        rgb_key = f"observation.images.{_out_name(name, cfg)}"
        depth_key = (
            f"observation.images.depth_{_out_name(name, cfg)}" if name in depth_cameras else None
        )
        target_h, target_w = sizes[name]

        cam = _open_camera(path)
        idx = 0
        written = 0
        pbar = tqdm(total=n_min, unit="frame", desc=f"    {name}", dynamic_ncols=True)
        while cam.grab() and written < n_min:
            if idx < start:
                idx += 1
                continue
            frame = cam.get_frame()

            # Native geometry, post-flip: this is what the stored intrinsics
            # describe, and what the lossless keyframes archive.
            color_native = cv2.rotate(frame.color, cv2.ROTATE_180) if flip else frame.color
            depth_native = None
            if frame.depth is not None:
                depth_native = (frame.depth * 1000.0).clip(0, 65535).astype(np.uint16)
                if flip:
                    depth_native = cv2.rotate(depth_native, cv2.ROTATE_180)

            if written < cfg.keyframe_count:
                # Archived for every camera, including ones whose depth video is
                # skipped — one frame costs ~350 KB and pose estimation wants it.
                _write_keyframe(
                    root=Path(cfg.output_dir),
                    episode_index=len(writer.episodes),
                    camera=_out_name(name, cfg),
                    frame_index=written,
                    color_bgr=np.ascontiguousarray(color_native),
                    depth_mm=None if depth_native is None else np.ascontiguousarray(depth_native),
                    intrinsics=(lowdim[written].get("intrinsics") or {}).get(name),
                    extrinsics=(lowdim[written].get("extrinsics") or {}).get(name),
                    timestamp_ns=int(final_ts[name][written]) if len(final_ts.get(name, [])) > written else None,
                )

            color = color_native
            if (color.shape[0], color.shape[1]) != (target_h, target_w):
                color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            # ascontiguousarray also *copies*: `color` may be a view onto the
            # librealsense frame pool, and retaining views exhausts it.
            rgb = np.ascontiguousarray(color[:, :, ::-1])  # BGR -> RGB
            writer.add_video_frame(rgb_key, rgb)
            if written in stat_idx:
                samples[rgb_key].append(np.transpose(rgb, (2, 0, 1)).copy())

            if depth_key is not None and depth_native is not None:
                depth_mm = depth_native
                if depth_mm.shape != (target_h, target_w):
                    depth_mm = cv2.resize(
                        depth_mm, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                    )
                codes = quantize_depth_mm(
                    depth_mm, cfg.depth_min, cfg.depth_max, cfg.depth_shift
                )
                writer.add_video_frame(depth_key, codes)
                if written in stat_idx:
                    samples[depth_key].append(codes[None].astype(np.float32).copy())

            written += 1
            idx += 1
            pbar.update(1)
        pbar.close()
        cam.close()

        if written < n_min:
            raise RuntimeError(
                f"{name}: only {written} of {n_min} frames decoded — recording truncated?"
            )

    columns: Dict[str, np.ndarray] = {}
    for key, name in (
        ("joints", "observation.state"),
        ("action_joints", "action"),
        ("actual_poses", "observation.state.ee_pose"),
        ("action", "action.ee_pose"),
    ):
        if key in lowdim[0]:
            columns[name] = np.stack([np.asarray(f[key], dtype=np.float32) for f in lowdim])
    for cam in cameras:
        if cam in (lowdim[0].get("extrinsics") or {}):
            columns[f"observation.extrinsics.{_out_name(cam, cfg)}"] = np.stack(
                [np.asarray(f["extrinsics"][cam], dtype=np.float32).reshape(16) for f in lowdim]
            )
        if cam in (lowdim[0].get("intrinsics") or {}):
            k = np.asarray(lowdim[0]["intrinsics"][cam], dtype=np.float32).reshape(9)
            columns[f"observation.intrinsics.{_out_name(cam, cfg)}"] = np.tile(k, (n_min, 1))
    columns.update(extra_columns)

    task = str(rec_meta.get("task_instruction") or rec_meta.get("task_name") or "")
    writer.end_episode(
        columns,
        task,
        {k: np.stack(v) for k, v in samples.items() if v},
        source=rec_path.resolve().as_posix(),
    )
    print(f"    ✓ {n_min} frames -> episode {len(writer.episodes) - 1}")
    return n_min


def run_lerobot_export(
    recording_dirs: Sequence[Path], cfg: LeRobotExportConfig
) -> LeRobotV3Writer:
    """Export a list of raw recording directories into one LeRobot v3.0 dataset."""
    writer = LeRobotV3Writer(cfg.output_dir, cfg)

    dirs = [Path(r) for r in recording_dirs]
    skipped = [d for d in dirs if d.resolve().as_posix() in writer.sources]
    dirs = [d for d in dirs if d.resolve().as_posix() not in writer.sources]
    if skipped:
        print(f"  skipping {len(skipped)} already-exported recording(s)")
    if cfg.max_episodes > 0:
        dirs = dirs[: cfg.max_episodes]

    if not dirs:
        print(f"\n✓ nothing new to export — {len(writer.episodes)} episode(s) already in {cfg.output_dir}")
        return writer

    total = 0
    for rec in dirs:
        total += export_recording(Path(rec), writer, cfg)
    writer.finalize()

    print(f"\n✓ LeRobot v3.0 dataset: {cfg.output_dir}")
    print(f"  {len(writer.episodes)} episodes ({len(dirs)} new, {total} new frames), "
          f"{writer.total_frames} frames total, {len(writer.tasks)} task(s)")
    size_mb = sum(p.stat().st_size for p in Path(cfg.output_dir).rglob("*") if p.is_file()) / 1e6
    print(f"  {size_mb:.1f} MB on disk")
    return writer


def _recordings_in(task_path: Path) -> List[Path]:
    return sorted(d for d in task_path.iterdir() if d.is_dir() and (d / "cameras").exists())


def resolve_raw_recordings(
    data_dir: str = "data",
    tasks: Optional[Sequence[str]] = None,
    episodes: Optional[Sequence[str]] = None,
) -> List[Tuple[Path, List[Path]]]:
    """Resolve which recordings to export; returns ``[(task_dir, recording_dirs)]``.

    With no ``tasks``, an fzf picker lists everything under ``<data_dir>/raw/``.
    Naming tasks explicitly skips the picker.  ``episodes`` filters by directory
    name within each task and accepts either the padded form (``0003``) or a bare
    integer (``3``).
    """
    base = Path(data_dir) / "raw"
    if tasks:
        task_paths = []
        for name in tasks:
            # accept a task name, or a path to the task directory
            candidate = Path(name) if Path(name).is_dir() else base / name
            if not candidate.is_dir():
                available = sorted(d.name for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
                raise SystemExit(
                    f"task not found: {name!r} (looked in {candidate}).\n"
                    f"Available under {base}: {', '.join(available) or '(none)'}"
                )
            task_paths.append(candidate)
    else:
        task_paths = [Path(t) for t in _conv.select_tasks(data_dir)]

    wanted: Optional[set] = None
    if episodes:
        wanted = set()
        for e in episodes:
            e = e.strip()
            wanted.add(e)
            if e.isdigit():
                wanted.add(f"{int(e):04d}")

    out: List[Tuple[Path, List[Path]]] = []
    for task_path in task_paths:
        recs = _recordings_in(task_path)
        if wanted is not None:
            recs = [r for r in recs if r.name in wanted]
            if not recs:
                have = ", ".join(d.name for d in _recordings_in(task_path)) or "(none)"
                raise SystemExit(
                    f"no matching episodes in {task_path}: asked for "
                    f"{sorted(wanted)}, available: {have}"
                )

        # Drop anything not marked successful, matching ``rd convert``.  Without
        # this a failed or unmarked take would be exported and — with --upload —
        # published to the Hub as though it were a good demonstration.
        # "unknown" is kept: recordings predating the status field.
        keep = []
        for r in recs:
            status = demonstration_status(r)
            if status in ("success", "unknown"):
                keep.append(r)
            else:
                print(f"  Skipping {task_path.name}/{r.name} (status={status})")
        recs = keep

        if recs:
            out.append((task_path, recs))
    return out


def select_raw_recordings(data_dir: str = "data") -> List[Tuple[Path, List[Path]]]:
    """fzf-select raw task directories; returns [(task_dir, recording_dirs)]."""
    return resolve_raw_recordings(data_dir)
