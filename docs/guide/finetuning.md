# Fine-tuning a VLA on your own demonstrations

Record demonstrations on the arm, export them to a LeRobot dataset, fine-tune a
pretrained policy on it, run the result back on the arm.

```
rd record  ->  rd export_lerobot --upload  ->  lerobot-train  ->  rd serve + lerobot_client.py
   raw bags        HF dataset repo             HF model repo         the arm
```

**Scope.** This path has been walked with **PI0.5** (`lerobot/pi05_base`) on a
single YAM arm, and the details below — the camera slot names, the action width
fix, the prompt, the client's defaults — are PI0.5's. PI0 loads through the same
client (`--policy-family pi0`) but has not been trained or evaluated this way.
Other families (SmolVLA, ACT, diffusion) need their own recipe; only the
recording and export stages carry over unchanged.

**The rule that costs the most when broken:** a checkpoint belongs to the LeRobot
version that trained it. Between releases the prompt a PI0.5 checkpoint receives
and the image preprocessing both changed, so the same weights behave differently
under a different version — the arm still moves, roughly correctly, and misses.
Pin one version and use it at both ends. This guide uses **0.4.1**.

## 1. One-time setup

### The robot host

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Clone **this fork** with
submodules, then install `rd` with the extras matching your cameras:

```bash
git clone --recurse-submodules git@github.com:zyang1-chwy/raiden.git && cd raiden
uv run python scripts/install_pyzed.py        # ZED only, after installing the ZED SDK
uv tool install -e ".[zed]"                   # or ".[realsense]"
rd --help
```

Re-run `uv tool install --reinstall -e ".[<extras>]"` after changing extras or
pulling.

The fork is not optional here. Three things this guide depends on exist only in
it: `rd export_lerobot` and its Hugging Face uploader (stage 3), `--control-hz`
on `rd serve` (stage 5), and the recording changes described below. Upstream
TRI-ML raiden has none of them.

### CAN interface names

Each arm sits on its own CAN interface, and raiden looks the arms up **by
interface name**, not by discovery order:

| Interface | Arm |
|---|---|
| `can_follower_l` | Left follower |
| `can_leader_l` | Left leader |
| `can_follower_r` | Right follower |
| `can_leader_r` | Right leader |

A single-arm setup needs only the two left names — the one arm is always treated
as the left arm. The names must be **persistent aliases**, not whatever `can0`
the kernel hands out this boot; follow i2rt's
[persistent CAN IDs](https://github.com/i2rt-robotics/i2rt/blob/7b6d5016f05ca63f9ef0185b7143e63f2c7a5708/docs/getting-started/hardware-setup.md#persistent-can-ids)
guide to set them.

All arms look identical on the bus, so name them **one at a time**: disconnect
everything, connect one arm, `ip link show` to see which `can*` appeared, assign
that arm's persistent name, disconnect, repeat. Getting this wrong swaps leader
and follower, or left and right, and nothing later will tell you so.

After every reboot the interfaces come up with:

```bash
rd reset_can                                        # all of them, 1 Mbit/s
rd reset_can --interfaces can_follower_l            # or just one
```

### Foot pedal and cameras

The foot pedal is optional but makes recording one-handed:

```bash
sudo bash scripts/install_footpedal_udev.sh
```

Cameras are declared in `~/.config/raiden/camera.json`, one entry per camera with
its serial, type, role, fps and resolution. `rd list_devices` prints what is
actually connected, serials included.

Pin the camera exposure and white balance now, and write the values down. A
policy trained on a few dozen episodes has seen exactly one lighting condition,
and auto white balance is free to settle somewhere else on another day. Whatever
values the recording session used are the values every later rollout needs.

Then calibrate: `rd record_calibration_poses` followed by `rd calibrate` for
hand-eye and scene extrinsics.

### The training machine

Training does not have to run on the robot host — the dataset travels through the
Hub. It needs its own environment, and this is where the version pin lives:

```bash
uv venv --python 3.11 .venv-train && source .venv-train/bin/activate
uv pip install "lerobot==0.4.1"
uv pip install "git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
```

That transformers commit is Hugging Face's `fix/lerobot_openpi` branch pinned so
it cannot move. PI0/PI0.5 need its attention replacement; on stock transformers
the model constructor raises `An incorrect transformer version is used`, which
names neither the package nor the version it wants.

Put a write token in `.env` at the repo root for the dataset and checkpoint
uploads: `echo 'HF_TOKEN=hf_xxx' >> .env`.

## 2. Record

```bash
rd record --arms single                                   # leader-follower
rd record --arms single --control spacemouse --vel-scale 0.07 --rot-scale 0.8
```

You are prompted for the task name and the instruction. **The instruction becomes
the language prompt the policy is trained on**, so type it identically every
session — it is not a label, it is an input.

What runs while you teleoperate:

- cameras at 30 fps, writing `.svo2` (ZED) or `.bag` (RealSense) — the raw sensor
  stream, stereo pair and depth included;
- both arms' joint positions and velocities at ~100 Hz into `robot_data.npz`,
  timestamped on the reference camera's clock so the streams can be aligned later
  without guesswork;
- the leader arm driving the follower, which is what makes `follower_*_joint_cmd`
  (what the leader asked for) and `follower_*_joint_pos` (where the arm got to)
  two different streams. The policy is trained to predict the first from the
  second.

Controls during a session:

| Input | Action |
|---|---|
| Leader button / left pedal | start recording; pressing again stops it immediately |
| Middle pedal, or leader top button, or `Enter` | mark **success** |
| Right pedal, or leader bottom button, or `f` | mark **failure** |
| any other key | discard |

The marking prompt appears the moment recording stops and **deletes the episode
if you do not answer within 30 seconds** — an unmarked demo is not kept as
`pending`, it is gone. Discarding is a real choice: a bungled demonstration
should be thrown away at the arm rather than filtered out later.

Discarded episodes leave a gap in the numbering and the gap stays: the next
episode takes one past the highest existing number, never a freed one. An
episode whose recording never completed is wiped and its number reused.

Each kept episode lands in `data/raw/<task>/NNNN/` with `metadata.json` (task,
instruction, duration, status, camera start timestamps), `robot_data.npz`, and
one file per camera under `cameras/`.

Before recording fifty of them, record one and check it: `rd replay` puts the
motion back on the arm, `rd visualize` shows the streams in Rerun. Vary what the
policy should generalise over — object position first — and hold everything else
still. Teleoperate at a steady pace; the policy copies the command stream it is
given, jerk included.

## 3. Export to a LeRobot dataset and upload

```bash
rd export_lerobot --task <task> --upload
```

Raw recordings become a LeRobot v3.0 dataset — parquet plus mp4 — under
`data/lerobot/<task>/`, pushed to `<your-hf-user>/<task>`. Re-running is
incremental: `meta/raiden_sources.json` remembers which recording produced each
episode, so adding demonstrations means running the same command again.

Three features matter for training:

| feature | what it holds |
|---|---|
| `observation.state` | the follower's **measured** joints, 6 + gripper per arm |
| `action` | the **commanded** joints, same layout |
| `observation.images.<camera>` | one video stream per camera at `--fps` (default 30) |

Because the command leads the measurement, `action[t]` is never `state[t]`. That
offset is real and the policy learns it; it is also the yardstick for judging a
rollout in stage 6, so measure it once on your own dataset and keep the number.

Worth knowing: `--depth-cameras none` drops depth entirely, which is what you
want for a VLA — a depth stream routed into a three-channel vision tower is a
silent failure, not an error. `--episodes 0000,0003` exports a subset,
`--resize 256x256` shrinks frames at encode time, `--reexport` rebuilds from
scratch, `--upload-only` pushes a dataset that is already on disk.

## 4. Train

```bash
lerobot-train \
  --policy.path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id=<your-hf-user>/<task> \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.scene_camera": "observation.images.base_0_rgb",
                 "observation.images.left_wrist_camera": "observation.images.left_wrist_0_rgb"}' \
  --output_dir=outputs/train/<job> \
  --job_name=<job> \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=2000 \
  --wandb.enable=false
```

`--rename_map` puts your camera names into the slots the base checkpoint was
pretrained with (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`). Fewer
cameras than the base expects is fine. Note that supplying a rename map skips the
feature-consistency check entirely, so read the mapping twice.

**Pin the base checkpoint's revision.** `lerobot/pi05_base` moves; a later commit
added a processor step older LeRobot cannot load, and `from_pretrained` then fails
with `Failed to load processor step from registry`. Record the revision you used.

Four things that bite, none of which announce themselves:

1. **`--dataset.episodes` does not filter a v3.0 dataset.** It selects the files
   containing those episodes, and a small dataset packs everything into one
   parquet file — so the frame index still covers every episode while the log
   reports the count you asked for. A real holdout means filtering the parquet
   rows and `meta/episodes` on disk.
2. **The saved config keeps the base's padded action width.** PI0.5 pads actions
   to 32 internally, and the dataset's real width only fills `output_features`
   when the config has not already defined it. Training is unaffected, but the
   checkpoint claims a 32-D action while its statistics are your robot's width,
   and inference fails with `The size of tensor a (32) must match tensor b (7)`.
   Fixed at export, below.
3. **Reported loss is diluted by that padding** — it averages over all 32
   dimensions when only yours carry signal, so the printed number is roughly
   `dof/32` of the true per-dimension error. Compare runs, not absolutes.
4. **Gradient checkpointing is what buys the batch size.** PI0.5's attention
   materialises the full matrix over ~760 tokens per sample: without it batch 8
   is the throughput optimum and batch 16 runs out of memory on a 96 GB card;
   with it batch 32 fits at roughly the same samples per second.

### Export the checkpoint

Copy it and correct the declared action width. Weights are untouched; the
unnormaliser already holds your robot's statistics.

```python
import json, shutil
from pathlib import Path

DOF = 7
src = Path("outputs/train/<job>/checkpoints/020000/pretrained_model")
dst = Path("checkpoints/<job>_7d")
shutil.copytree(src, dst, dirs_exist_ok=True)

cfg = json.loads((dst / "config.json").read_text())
cfg["output_features"]["action"]["shape"] = [DOF]
(dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
```

Load the export, feed it a few frames from the dataset, and check the chunk comes
back `(1, chunk_size, DOF)`, finite, different for different observations, and
near the recorded action at those frames. Then upload, recording the step count
and the LeRobot version in the message — nothing inside a checkpoint says which
version produced it, and that is exactly what you will want to know later:

```bash
hf upload <your-hf-user>/<name> checkpoints/<job>_7d . \
  --commit-message "PI0.5 SFT on <task>, step 20000, ${DOF}D action, lerobot 0.4.1"
```

## 5. Run it on the arm

Two terminals: the server owns the robot, the client owns the model.

```bash
# terminal 1 — the robot  (--control-hz is a fork addition; without it the
# server executes at its 10 Hz default no matter what the client sends)
rd serve --arms single --action-type joint --control-hz 30

# terminal 2 — the policy
python scripts/lerobot_client.py \
  --policy <your-hf-user>/<name> --task "<the recorded instruction>" --arm left \
  --cameras scene_camera:base_0_rgb left_wrist_camera:left_wrist_0_rgb \
  --control-hz 30 --sequential --settle-tol 0.02 --debug \
  --record-dir /abs/path/to/records/run01
```

Four things must match training, and each fails quietly:

- **`--control-hz`, on both sides**, equal to the dataset's fps. Each action was
  learned as the target reached one frame after the last one; running slower
  stretches every motion and feeds back states the policy never saw.
- **`--task`**, character for character the recorded instruction. A VLA
  conditions on it and an empty or altered prompt still returns plausible actions.
- **`--cameras`**, the same slot mapping as `--rename_map` at training time.
- **`--runtime`**, the LeRobot version that trained the checkpoint. Where that
  release is not installed in the venv you run from, the client can read a
  vendored copy instead; `--aida-root` points at the checkout holding one (the
  flag name is historical — any checkout with the vendored runtime works).

Start with `--dry-run`, which runs inference and prints the action without moving
anything. Then pick an execution mode:

| flag | behaviour | when |
|---|---|---|
| default | replans as soon as the action queue empties, never rests | normal running |
| `--sequential` | runs each chunk to its end, waits for the arm to settle, replans | diagnosis: removes every timing effect |
| `--rtc` | generates the next chunk while the current one runs and blends them | smoothest; needs a LeRobot with RTC |

## 6. Reading a run back

`--record-dir` saves what the model actually saw and produced: `model_io.npz`
(per dispatched action the state and the action; per inference the state given to
the model, that state after the preprocessor, and the whole chunk both normalised
and in joint radians), `model_io.json`, and `inputs/` with the frames each plan
was conditioned on.

```bash
python scripts/analyze_model_io.py /abs/path/to/records/run01 --full
```

Two numbers per inference:

- **`d_last`** — how far the arm was from the previous chunk's last action when
  the new one was planned. This is tracking, and if it is large nothing about the
  policy can be judged from that run.
- **`d_first`** — how far the new chunk's first action sits from the arm's
  current pose. Compare it with the `|action - state|` spread you measured in
  stage 3. Near the median is normal. Far above the maximum, pointing away from
  where the chunk then travels, means the plan opens by dragging the arm back.

When a grasp fails, check the gripper against its range in the dataset first. If
the fingers closed further than they ever did in a demonstration, they closed on
nothing, and everything after that is the policy retrying from a state it has
never seen.

## When it does not work

In this order, because each one invalidates the next:

1. **Runtime version** — does the LeRobot loading the checkpoint match the one
   that trained it?
2. **Prompt** — is `--task` exactly the recorded instruction?
3. **Cameras** — right slots, and the same exposure and white balance as the
   recording session.
4. **Rate** — is the whole chain at the dataset's fps, on both sides?
5. **Tracking** — is `d_last` small, i.e. did the arm reach what it was told?
6. **The policy** — only once the five above hold.
