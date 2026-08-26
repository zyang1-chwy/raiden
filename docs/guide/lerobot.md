# Exporting to LeRobot

The `rd export_lerobot` command converts **raw** recordings directly into a
[LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset — parquet plus
mp4 — for use with the Hugging Face ecosystem.

It reads `.bag` / `.svo2` files and `robot_data.npz` and pipes decoded frames
straight into the video encoders. The intermediate UnifiedDataset layer that
`rd convert` writes (lossless PNG, `.npz` depth, per-frame pickles — roughly
250 MB per episode) is never created.

```bash
rd export_lerobot
```

An fzf selector lists raw tasks newest first. Each selected task becomes its own
dataset under `<output-dir>/<task_name>/`.

## Batch and incremental runs

Each selected task is exported as a whole: every recording directory under
`data/raw/<task>/` becomes an episode. Re-running is **incremental** — the
exporter records which raw recording produced each episode in
`meta/raiden_sources.json` and skips anything already there, so adding
demonstrations means re-running the same command:

```console
$ rd export_lerobot        # first run
    ✓ 145 frames -> episode 0
    ✓ 666 frames -> episode 1
  2 episodes (2 new, 811 new frames), 811 frames total

$ rd export_lerobot        # after recording one more
  resuming dataset: 2 episode(s), 811 frame(s) already exported
  skipping 2 already-exported recording(s)
    ✓ 703 frames -> episode 2
  3 episodes (1 new, 703 new frames), 1514 frames total

$ rd export_lerobot        # nothing new
  ✓ nothing new to export — 3 episode(s) already in data/lerobot/stand
```

New episodes are appended into a fresh `(chunk, file)` slot rather than rewriting
existing files, matching LeRobot's own resume behaviour — already-written parquet
and mp4 files stay byte-identical:

```
data/chunk-000/file-000.parquet     episodes 0-1   (run 1, untouched)
data/chunk-000/file-001.parquet     episode  2     (run 2)
```

Unlike `rd convert`, this does not consult the metadata database or the
`converted` flag — the source manifest inside the dataset is the record, so the
dataset is self-describing and moving it does not lose the state.

Pass `--reexport` to rebuild a dataset from scratch. It deletes the previous
`data/`, `videos/` and `meta/` first, so no orphaned files are left behind.

A resumed export must keep the schema its first episode established; a recording
with different cameras or channels is rejected with an explicit error rather than
silently corrupting the dataset.

## Relationship to `rd convert`

`rd export_lerobot` does not replace `rd convert` — it is a second, independent
sink for the same source data. Both produce identical numbers: forward
kinematics, hand-eye wrist extrinsics, the bimanual base transform, camera
alignment, episode-end trimming and the ~100 Hz → camera-rate interpolation are
all performed by calling into `raiden/converter.py` itself. Only the output
container differs.

Use `rd convert` when you want browsable frames on disk (`rd visualize`,
`rd shardify`). Use `rd export_lerobot` when you want a training dataset.

## Output layout

```
<output-dir>/<task_name>/
    meta/info.json                              # codebase_version "v3.0"
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet             # many episodes per file
    videos/observation.images.scene_camera/chunk-000/file-000.mp4
    videos/observation.images.depth_scene_camera/chunk-000/file-000.mp4
```

Episodes are concatenated into shared parquet and mp4 files (up to 100 MB and
200 MB respectively), exactly as LeRobot v3.0 specifies.

## Feature mapping

| LeRobot feature | Source | Shape |
|---|---|---|
| `observation.images.<cam>` | camera colour stream | video `(H, W, 3)` |
| `observation.images.depth_<cam>` | camera depth stream | video `(H, W, 1)` |
| `observation.state` | `joints` — measured follower joints | `(7,)` / `(14,)` |
| `action` | `action_joints` — commanded joints | `(7,)` / `(14,)` |
| `observation.state.ee_pose` | `actual_poses` — FK(actual) | `(13,)` / `(26,)` |
| `action.ee_pose` | `action` — FK(commanded) | `(13,)` / `(26,)` |
| `observation.extrinsics.<cam>` | per-frame cam2world, row-major | `(16,)` |
| `observation.intrinsics.<cam>` | pinhole `K`, row-major | `(9,)` |
| `observation.velocity` | follower joint + gripper velocity | `(7,)` / `(14,)` |
| `observation.effort` | follower joint + gripper effort | `(7,)` / `(14,)` |
| `observation.leader.joint_position` | leader arm position | `(6,)` / `(12,)` |
| `observation.leader.joint_velocity` | leader arm velocity | `(6,)` / `(12,)` |
| `observation.leader.joint_effort` | leader arm effort | `(6,)` / `(12,)` |
| `observation.timestamp_ns` | wall-clock capture time | `int64` |
| `task` | `task_instruction` from `metadata.json` | string |

Single-arm recordings produce the 7/13-dim variants, bimanual the 14/26-dim
ones. Extrinsics and intrinsics are flattened to 16 and 9 elements rather than
stored as `Array2D`, which keeps the parquet schema to plain fixed-size lists.

Every channel of `robot_data.npz` is carried across, interpolated onto the camera
grid exactly as the pose channels are, so the dataset is self-contained for
training and demo reconstruction without the raw `.bag`. Channels a recording
does not have are simply absent — a SpaceMouse session has no `leader_*`
columns.

`observation.timestamp_ns` is the original wall-clock capture time in
nanoseconds, stored as `int64`. LeRobot's own `timestamp` column remains
`frame_index / fps`; use `observation.timestamp_ns` when you need to line frames
up against another clock. It is deliberately not a float — nanosecond epoch
values need 61 bits and a float64 round-trip rounds them to multiples of 256.

!!! warning "One schema per dataset"
    The feature schema and camera set are fixed by the first episode. Recordings
    that differ — a leader-teleop session mixed with a SpaceMouse one, or two
    different camera sets — are rejected with an explicit error. Export them as
    separate datasets.

## Depth

Depth is encoded the way LeRobot does it: quantized to 12-bit **logarithmic**
codes and stored in an HEVC Main 12 `gray12le` stream, losslessly by default.

!!! warning "Do not compress depth lossily"
    Lossy depth is tempting — HEVC at `crf 18` is 6× smaller — but it smears
    object boundaries: measured on this dataset, mean error rises to 13 mm and
    p99 to 148 mm. Lossless costs about 56 KB/frame and is the default.

Only the scene camera gets depth by default:

```bash
rd export_lerobot --depth-cameras scene_camera   # default
rd export_lerobot --depth-cameras none           # RGB only — ~40x smaller
```

Wrist depth roughly triples dataset size and is not consumed by common VLA
policies.

### The no-data sentinel

Raiden marks missing depth with `0`, which is pervasive — every episode in a
typical dataset has 17–53 % zero pixels, concentrated in a vertical band on each
image edge. Quantization maps `0` to code `0`, and no real reading reaches code
`0`, so **code 0 is an exact no-data mask**.

LeRobot's own dequantizer does not know this and returns `depth_min` (10 mm) for
those pixels. Use `raiden.lerobot_export.dequantize_depth_mm`, which returns
`NaN` instead:

```python
from raiden.lerobot_export import dequantize_depth_mm
depth_mm = dequantize_depth_mm(codes)     # NaN where there is no reading
```

Readings beyond `--depth-max` (10 m by default) saturate at that value.

## Options

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Root data directory; reads `<data-dir>/raw/` |
| `--output-dir` | `data/lerobot` | Output directory |
| `--repo-id` | `raiden/<task>` | Dataset repo id in `meta/info.json` |
| `--robot-type` | `yam` | `robot_type` in `meta/info.json` |
| `--fps` | `30` | Frame rate in `meta/info.json` |
| `--depth-cameras` | `scene_camera` | Cameras to emit depth for; `none` disables |
| `--rgb-codec` | `libsvtav1` | `libsvtav1` or `libx264` |
| `--rgb-crf` | `30` | RGB quality; lower is better and larger |
| `--rgb-gop` | `2` | RGB keyframe interval — see below |
| `--depth-lossless` / `--no-depth-lossless` | on | Lossless depth |
| `--resize` | native | Resize to `HxW` before encoding |
| `--max-episodes` | `-1` | Limit episodes exported |

### Tuning `--rgb-gop`

LeRobot's default of `2` puts a keyframe every other frame so single-frame
random access stays fast. It costs about 5× the bitrate. If you train on
contiguous action chunks rather than isolated frames, `--rgb-gop 30` is a large
saving:

| setting | 694-frame episode, 640×360 |
|---|---|
| lossless PNG (`rd convert`) | 197 MB |
| `--rgb-gop 2` (default) | 4.0 MB |
| `--rgb-gop 30` | 0.8 MB |

## Measured sizes

A 694-frame single-camera episode (`stand/0000`, 640×360):

| output | size |
|---|---|
| `rd convert` layer (PNG + npz + pkl) | 249.5 MB |
| LeRobot, RGB + lossless scene depth | 44.5 MB |
| LeRobot, RGB only, `--rgb-gop 30` | 1.0 MB |

## Lossless keyframes for pose estimation

The video streams are lossy (AV1 ~38 dB) and range-quantize depth. For pose
estimation or scene reconstruction you usually want the untouched sensor output
of the opening frame, so the exporter archives it separately:

```
meta/keyframes/episode-000000/
    scene_camera_0000_color.png     # lossless PNG, BGR, native resolution
    scene_camera_0000_depth.png     # uint16 millimetres — the raw array, not
                                    # the 12-bit log codes, so 0 = no data and
                                    # out-of-range readings survive
    scene_camera_0000.json          # intrinsics K, extrinsics cam2world,
                                    # timestamp_ns, resolution
```

Verified bit-identical to what `rd convert` writes for the same frame. Written
for **every** camera, including ones whose depth video is skipped — one frame is
a few hundred KB and pose estimation wants all of it. Always at native
resolution, so the stored `K` applies directly even when `--resize` is set.

`--keyframe-count N` archives the first N frames per episode (default 1, `0`
disables). Cost is roughly 200–460 KB per camera per frame:

| | 640×360 scene | 640×480 wrist |
|---|---|---|
| colour PNG | 124 KB | 324 KB |
| depth PNG | 68 KB | 132 KB |

LeRobot ignores `meta/keyframes/`, so the dataset still loads normally.

!!! warning "No stereo pair is recorded"
    `raiden/cameras/realsense.py` enables only `rs.stream.color` and
    `rs.stream.depth`. The RealSense IR stereo pair is never written to the
    `.bag`, so it cannot be exported — the keyframe gives you RGB-D (colour plus
    metric depth already aligned to colour), not stereo. Recording IR would
    require enabling `rs.stream.infrared` in the recorder, and would only affect
    future recordings.

!!! note "Extrinsics need calibration"
    Keyframe `extrinsics_cam2world` falls back to identity when no
    `calibration_results.json` is found next to the recording. Run
    `rd calibrate` first if you need camera pose in the robot base frame.

## What is not preserved

The export is a view of the recording, not an archive. Two things are dropped
and cannot be recovered from it:

- **Frames removed by alignment and episode-end trimming.** These can be
  substantial — in `pick_red_block/0010` the scene camera records 642 frames and
  the episode keeps 321, because the wrist camera started later.
- **The ZED right-eye image.** Only the left view is encoded, so depth cannot be
  re-derived later with a different stereo backend.

RGB is also lossy (AV1 CRF 30, ~38 dB PSNR) and depth saturates beyond
`depth_max`. Keep the raw recordings if you may want to re-derive depth or
re-cut episode boundaries.

## Resolution

One mp4 stream holds one resolution. Recordings of differing size must be
exported as separate datasets, or normalized with `--resize`. The exporter
raises a clear error rather than producing a corrupt stream.

## Reading the result

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="raiden/stand", root="data/lerobot/stand")
frame = ds[0]
frame["observation.images.scene_camera"]   # (3, H, W) float32 in [0, 1]
frame["observation.state"]                 # (7,) or (14,)
frame["task"]                              # the instruction string
```

!!! note "LeRobot needs Python ≥ 3.12; Raiden is pinned to 3.11"
    Raiden cannot import `lerobot` — the `pyzed` wheel pins it to Python 3.11
    while `lerobot` 0.6.1 requires 3.12+. The exporter therefore writes the v3.0
    container directly with `pyarrow` and `av` and has no `lerobot` dependency.
    Install `lerobot` in a separate environment to train from the output.
