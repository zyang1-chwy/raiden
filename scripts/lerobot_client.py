"""Drive a LeRobot-trained policy on YAM through the raiden chiral policy server.

Which interpreter to use depends on the mode:

``--mock-policy`` needs only ``chiral`` and ``numpy``, both of which raiden's own
venv already has, so run it from there.  ``--policy`` additionally needs
``lerobot`` and ``torch``, and LeRobot requires Python >= 3.12 while raiden is
pinned to 3.11 by the ``pyzed`` wheel — the two cannot share a venv, so that mode
runs in the LeRobot environment.  ``--replay`` is like ``--mock-policy``: it loads no model, so it needs only
``chiral``, ``numpy``, ``pyarrow`` and ``huggingface_hub`` and must run from
raiden's venv — the LeRobot environment has no ``chiral`` and so cannot reach
the server at all.  ``chiral`` is not on PyPI; install it there with
``pip install git+ssh://git@github.com/TRI-ML/chiral``.

Start the server first — without it the client blocks in chiral's reconnect
loop, printing "waiting for server"::

    rd serve --arms single --action-type joint

Then, from raiden's venv, the transport smoke test — no checkpoint, commands a
pure hold::

    .venv/bin/python scripts/lerobot_client.py --mock-policy --arm left --dry-run
    .venv/bin/python scripts/lerobot_client.py --mock-policy --arm left --mock-amplitude 0.05

And from the LeRobot environment, a real, *finetuned* policy::

    python scripts/lerobot_client.py --policy path/to/ckpt --task "pick the red block" \
        --arm left --cameras scene_camera:base_0_rgb left_wrist_camera:left_wrist_0_rgb

Action layout
-------------
``rd serve --action-type joint`` takes a 14-D vector laid out as
``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`` — left arm
first, matching what the server's ``get_metadata()`` advertises as
``left_joints(7)+right_joints(7)``, what ``_smooth_command`` /
``_check_joint_delta`` in ``raiden/server.py`` actually execute, and
``rd export_lerobot``'s ``action`` feature, which the converter builds
left-then-right.

A single-arm policy therefore writes its 7-D output into ``action[7:14]`` for the
right arm (the default here) or ``action[0:7]`` for the left.  The unused half is
latched to that arm's post-homing position so it holds still instead of being
commanded to zero.

Chunk timing
------------
Three ways to cross a chunk boundary, in increasing order of how much they let
the arm rest:

``--rtc``
    No boundary at all — the next chunk is generated on a background thread and
    blended into the tail of the current one.

default
    LeRobot's ``select_action`` refills its queue only once empty, so one
    inference runs per ``n_action_steps`` actions.  The next chunk is planned
    from the freshest observation the moment the queue empties -- the arm is
    still moving when it is taken, and nothing pauses.

``--sequential``
    Same, plus a wait until the arm is measurably within ``--settle-tol`` of the
    last commanded target before the dwell starts.

The dwell exists because a policy trained on quasi-static demonstrations expects
to be asked "what next?" from a pose it has actually reached.  Observed mid-flight,
it plans back toward where the arm was, which reads as the gripper being dragged
away from the object at every boundary.

Base checkpoints will not work
------------------------------
A *base* VLA checkpoint (``lerobot/pi05_base``, ``lerobot/smolvla_base``) ships
with an empty normalizer — no dataset statistics — and a padded 32-D action
space.  Normalization stats are baked in at finetune time from the training
dataset, so a base checkpoint emits normalized nonsense in the wrong dimension.
This script refuses to run one; finetune on the dataset ``rd export_lerobot``
produces first.
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

import yam_policy_041  # noqa: E402  (the 0.4.1 model half, on the AIDA vendored runtime)
from yam_lerobot_policy import (  # the model half; this file is the raiden half
    DOF,
    TRAIN_REFERENCE,
    IORecord,
    RTCDriver,
    RTCSettings,
    SequentialChunker,
    YamLeRobotPolicy,
    load_action_normalizers,
)

BIMANUAL_DOF = DOF * 2

# Slot each arm occupies in the 14-D joint action vector.
_SLOTS = {"left": slice(0, DOF), "right": slice(DOF, BIMANUAL_DOF)}

# raiden/server.py:_DEFAULT_MAX_JOINT_DELTA — the server e-stops when a commanded
# joint is further than this from the measured one, which bounds how big a step
# the replay ramp (and the replay itself) may take.
_SERVER_MAX_JOINT_DELTA = 0.2

# Proprio stream names published by raiden/server.py:proprio_configs().
_PROPRIO = {"left": "follower_l_joint_pos", "right": "follower_r_joint_pos"}


def _parse_cameras(specs):
    """Parse ``--cameras`` entries of the form ``server_name`` or ``server_name:dataset_name``."""
    out = []
    for spec in specs:
        server_name, _, dataset_name = spec.partition(":")
        out.append((server_name, dataset_name or server_name))
    return out


def _camera_ages_ms(obs):
    """Age of each camera frame in this observation, right now, in milliseconds.

    The server stamps frames with ``time.time_ns()`` the instant
    ``wait_for_frames()`` returns (raiden/server.py:1421) and ships that as
    ``CameraInfo.timestamp``.  The client runs on the same host, so the two are
    on the same Unix clock and the subtraction is meaningful.

    Covers resize, encode, transport and decode.  It does NOT include the
    sensor-to-driver latency before ``wait_for_frames`` returned, so read it as
    a lower bound on the true age of the pixels.
    """
    now = time.time_ns()
    ages = {}
    for cam in getattr(obs, "cameras", ()):
        ts = getattr(cam, "timestamp", 0) or 0
        if ts > 0:
            ages[cam.name] = (now - ts) / 1e6
    return ages


def _read_state(obs, arms):
    """Concatenate follower joint positions in the converter's left-then-right order."""
    parts = []
    for arm in arms:
        value = obs.proprios.get(_PROPRIO[arm])
        if value is None:
            raise SystemExit(
                f"server published no {_PROPRIO[arm]!r} stream — is the {arm} arm connected?"
            )
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def _images(obs, cameras):
    """The server's frames, keyed by the model image slot each one feeds.

    Images stay (H, W, C) uint8; the policy packs them into a frame and the
    preprocessor permutes to (C, H, W) and scales to [0, 1].
    """
    out = {}
    for server_name, slot in cameras:
        try:
            out[slot] = np.ascontiguousarray(obs[server_name].image)
        except KeyError:
            raise SystemExit(
                f"server has no camera {server_name!r}; "
                f"available: {', '.join(c.name for c in obs.cameras)}"
            )
    return out


def _action_debug_line(step, action, state, prev):
    """One-line per-step summary of what the policy asked for.

    ``d_state`` is the largest arm-joint gap between the action and the current
    pose — how far the policy wants to move.  ``d_prev`` is the largest change
    since the previous action — whether the policy is varying at all.  A
    ``d_prev`` that stays near zero every step means a constant is being emitted,
    which is what an untrained head does: the arm reaches one pose and holds,
    looking static even though actions are flowing normally.

    Gripper elements are excluded from ``d_state`` (linear position, not
    radians) but kept in ``d_prev``, where any variation is worth seeing.
    """
    values = np.array2string(
        action, precision=3, suppress_small=True, max_line_width=250, separator=" "
    )
    n_arms = max(1, len(action) // DOF)
    joints = np.concatenate([np.arange(i * DOF, i * DOF + 6) for i in range(n_arms)])
    joints = joints[joints < len(action)]

    d_state = (
        float(np.abs(action[joints] - np.asarray(state)[joints]).max())
        if len(state) == len(action)
        else float("nan")
    )
    d_prev = float(np.abs(action - prev).max()) if prev is not None else float("nan")
    return f"[{step:5d}] action={values}  d_state={d_state:7.4f}  d_prev={d_prev:7.4f}"


def _load_replay_episode(source, episode, field, revision=None):
    """Load one episode's recorded vectors from a LeRobot dataset.

    Reads the parquet shards with pyarrow rather than going through
    ``LeRobotDataset``: the episode's videos are never fetched (they are most of
    the download), and — the reason it matters here — replay must run from
    raiden's venv, which has ``chiral`` and ``pyarrow`` but neither ``pandas``
    nor ``lerobot``.

    Returns ``(frames, fps)`` where *frames* is (T, action_dim) float32.
    """
    import glob
    import json
    import os

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    def _info_fps(path):
        try:
            with open(path) as f:
                return json.load(f).get("fps")
        except Exception:
            return None

    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "data", "**", "*.parquet"), recursive=True))
        if not files:
            raise SystemExit(f"no data/**/*.parquet under {source}")
        fps = _info_fps(os.path.join(source, "meta", "info.json"))
    else:
        from huggingface_hub import hf_hub_download, list_repo_files

        names = sorted(
            f for f in list_repo_files(source, repo_type="dataset")
            if f.startswith("data/") and f.endswith(".parquet")
        )
        if not names:
            raise SystemExit(f"no data/**/*.parquet in dataset repo {source!r}")
        files = [
            hf_hub_download(source, n, repo_type="dataset", revision=revision) for n in names
        ]
        try:
            fps = _info_fps(
                hf_hub_download(source, "meta/info.json", repo_type="dataset", revision=revision)
            )
        except Exception:
            fps = None

    parts = []
    seen = set()
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", field])
        seen.update(table["episode_index"].to_pylist())
        rows = table.filter(pc.equal(table["episode_index"], episode))
        if rows.num_rows:
            parts.append(rows)
    if not parts:
        raise SystemExit(
            f"episode {episode} not found in {source} — available: "
            f"{sorted(seen)[:8]}{'…' if len(seen) > 8 else ''}"
        )
    import pyarrow as pa

    rows = pa.concat_tables(parts).sort_by([("frame_index", "ascending")])
    return np.asarray(rows[field].to_pylist(), dtype=np.float32), fps


class _ReplaySource:
    """Emit a recorded episode frame by frame, in place of a policy.

    The point is to separate the policy from everything around it.  These are
    the exact vectors a human teleoperated, pushed through the same slot
    placement, safety gate, dispatch and smoothing a policy's output goes
    through.  If the arm retraces the demo and grasps the block, the transport,
    units, action layout and server timing are all sound and the fault is the
    policy's.  If it does not, the fault is below the policy and no amount of
    retraining will fix it.
    """

    def __init__(self, frames, stride):
        self._frames = frames[::stride]
        self._index = 0
        self.stride = stride

    def __len__(self):
        return len(self._frames)

    @property
    def first(self):
        return self._frames[0]

    def done(self):
        return self._index >= len(self._frames)

    def pop(self):
        action = self._frames[self._index]
        self._index += 1
        return action

    def progress(self):
        return self._index, len(self._frames)


class _NormalizeRoundTrip:
    """Send a recorded action the way a policy's output would travel.

    Normalize it with the statistics training used, treat the result as if the
    model had emitted it, then unnormalize back to joint radians and command
    that.  Any discrepancy between what was recorded and what the arm is told
    to do is contributed by the normalization pair alone — everything else is
    identical to the plain replay, which already reaches the block.

    Also records how far the normalized values stray outside [-1, 1], the range
    pi05's state tokenizer assumes and clips to.
    """

    def __init__(self, normalize, unnormalize):
        self._normalize = normalize
        self._unnormalize = unnormalize
        self.errors = []
        self.outside = 0
        self.values = 0
        self.extreme = 0.0

    def __call__(self, action):
        normalized = self._normalize(action)
        flat = np.asarray(normalized.detach().to("cpu").float().numpy()).reshape(-1)
        self.outside += int((np.abs(flat) > 1.0).sum())
        self.values += flat.size
        self.extreme = max(self.extreme, float(np.abs(flat).max()))
        out = self._unnormalize(normalized)
        self.errors.append(float(np.abs(out - np.asarray(action, dtype=np.float32)).max()))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", help="finetuned LeRobot checkpoint directory or Hub repo id")
    ap.add_argument(
        "--runtime",
        choices=["0.4.1", "0.6.1"],
        default="0.4.1",
        help="which LeRobot runs the checkpoint. The default 0.4.1 is the subset vendored in "
        "the AIDA repo and the one every checkpoint there was trained with — the same runtime the "
        "Isaac Lab rollout uses, so a robot run and a simulator rollout differ in the robot "
        "and not in the runtime. 0.6.1 is the installed package, kept for comparing the two; "
        "it writes a shorter prompt and moves the actions (see "
        "docs/implementation-notes/2026-09-11-pi05-runtime-version-contract.md). 0.4.1 needs "
        "the patched transformers branch and has no --rtc",
    )
    ap.add_argument(
        "--aida-root",
        default=yam_policy_041.DEFAULT_AIDA_ROOT,
        metavar="DIR",
        help=f"AIDA checkout holding the vendored 0.4.1 runtime, for --runtime 0.4.1 "
        f"(default {yam_policy_041.DEFAULT_AIDA_ROOT})",
    )
    ap.add_argument(
        "--policy-family",
        choices=["pi05", "pi0"],
        default="pi05",
        help="which vendored policy class --runtime 0.4.1 loads; 0.6.1 reads it from the config",
    )
    ap.add_argument(
        "--mock-policy",
        action="store_true",
        help="skip LeRobot entirely and command a hold, to test the transport and action wiring",
    )
    ap.add_argument(
        "--mock-amplitude",
        type=float,
        default=0.0,
        help="with --mock-policy, sine amplitude in rad on joint 0 (default 0.0 = pure hold)",
    )
    ap.add_argument(
        "--dataset-meta",
        default=None,
        metavar="DATASET_ROOT",
        help="take feature shapes and normalization stats from this LeRobot dataset instead of "
        "from the checkpoint. Makes a base checkpoint runnable for bring-up — actions are "
        "correctly scaled but carry no learned behaviour",
    )
    ap.add_argument(
        "--revision",
        default=None,
        metavar="SHA_OR_REF",
        help="pin the checkpoint to a specific Hub commit or branch (default: main, which "
        "moves when a newer checkpoint is pushed to the same repo)",
    )
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="diagnostic: execute a whole chunk, wait for the arm to settle, then plan the "
        "next from a fresh observation. Removes all chunk-timing effects so a remaining "
        "fault is the policy's. Cannot be combined with --rtc",
    )
    ap.add_argument(
        "--settle-tol",
        type=float,
        default=0.05,
        help="--sequential: arm counts as settled when every arm joint is within this many "
        "rad of the last commanded target (default: 0.05; training |action-state| p99 is 0.083)",
    )
    ap.add_argument(
        "--settle-timeout",
        type=float,
        default=3.0,
        help="--sequential: give up waiting for the arm to settle after this many seconds "
        "(default: 3.0)",
    )
    ap.add_argument(
        "--chunk-dwell",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="with --sequential, rest this long at the end of every chunk before planning "
        "the next. The last commanded pose is re-sent at --control-hz for the whole rest, so "
        "the arm stays exactly where the chunk left it instead of drifting. Only --sequential "
        "rests; the default chunked mode and --rtc never do. Also bounds the settle rest "
        "before a --replay starts (0 disables)",
    )
    ap.add_argument(
        "--rtc",
        action="store_true",
        help="Real-Time Chunking: generate the next chunk on a background thread while the "
        "current one executes, conditioned on its unexecuted tail. Removes the stall and "
        "snap-back at chunk boundaries. Flow-matching policies only (pi0/pi05/smolvla)",
    )
    ap.add_argument(
        "--rtc-horizon",
        type=int,
        default=10,
        help="RTC execution horizon: how many steps of the current chunk the next one is "
        "asked to agree with (default: 10)",
    )
    ap.add_argument(
        "--rtc-guidance",
        type=float,
        default=10.0,
        help="RTC max guidance weight — how hard the new chunk is pulled toward the old one "
        "over the overlap (default: 10.0)",
    )
    ap.add_argument(
        "--rtc-queue-threshold",
        type=int,
        default=40,
        help="start the next inference once the action queue drops to this many actions. "
        "Sets the replan interval: roughly (chunk_size - this) actions run between replans, "
        "so a low value leaves the arm executing a stale plan (default: 40 of a 50-step chunk)",
    )
    ap.add_argument(
        "--rtc-schedule",
        choices=["ZEROS", "ONES", "LINEAR", "EXP"],
        default="LINEAR",
        help="RTC prefix-attention schedule: how agreement decays across the overlap "
        "(default: LINEAR)",
    )
    ap.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        metavar="N",
        help="execute only N of each predicted chunk before re-inferring (default: the "
        "checkpoint's own value). 1 replans every control step; must not exceed chunk_size",
    )
    ap.add_argument(
        "--replay",
        default=None,
        metavar="DATASET",
        help="no policy: replay one recorded episode from this LeRobot dataset (Hub repo id "
        "or local root) through the whole client path. Isolates the policy from the "
        "transport — if the replay reaches the block, everything below the policy is sound",
    )
    ap.add_argument(
        "--replay-episode", type=int, default=0, metavar="N",
        help="which episode to replay (default: 0)",
    )
    ap.add_argument(
        "--replay-field",
        choices=["action", "observation.state"],
        default="action",
        help="column to replay. 'action' is what the teleoperator commanded and what a policy "
        "is trained to emit, so it is the like-for-like comparison; 'observation.state' is "
        "where the follower actually went (default: action)",
    )
    ap.add_argument(
        "--replay-stride",
        type=int,
        default=None,
        metavar="N",
        help="take every Nth recorded frame (default: dataset_fps / --control-hz, so the demo "
        "plays back at its original speed). 1 replays every frame, which runs it in slow "
        "motion when the dataset was recorded faster than --control-hz",
    )
    ap.add_argument(
        "--replay-approach",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="ramp from the homed pose to the episode's first recorded pose over this many "
        "seconds before replaying. The demo does not start at the home pose, and the server "
        "e-stops on a jump over 0.2 rad, so this is what makes the replay startable",
    )
    ap.add_argument(
        "--replay-normalize",
        default=None,
        metavar="CHECKPOINT",
        help="--replay only: before sending each recorded action, push it through this "
        "checkpoint's normalizer and back out through its unnormalizer, as a policy's "
        "output would travel. Isolates the normalization pair — if the arm still reaches "
        "the block, normalization is not what stops the policy",
    )
    ap.add_argument("--uri", default="ws://localhost:8765", help="chiral server URI")
    ap.add_argument("--task", default=None, help="language instruction (required by VLA policies)")
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=["scene_camera"],
        metavar="NAME[:DATASET_NAME]",
        help="server cameras to feed the policy, optionally renamed to the training key",
    )
    ap.add_argument(
        "--arm",
        choices=["right", "left", "both"],
        default="right",
        help="which slot of the 14-D action the policy drives (default: right -> action[7:14])",
    )
    ap.add_argument(
        "--state-arms",
        choices=["right", "left", "both"],
        default=None,
        help="arms making up observation.state (default: same as --arm)",
    )
    ap.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
        help="action rate; must not exceed the server's _CONTROL_HZ (10.0) or commands queue up",
    )
    ap.add_argument("--obs-hz", type=float, default=30.0, help="observation polling rate")
    ap.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.8,
        help="refuse to start if the first action is further than this (rad) from the current pose",
    )
    ap.add_argument("--device", default="cuda", help="torch device")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N actions (-1 = run until Ctrl-C)")
    ap.add_argument("--dry-run", action="store_true", help="run inference and print the action without moving the robot")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="print every generated action, with its distance from the current pose "
        "(d_state) and from the previous action (d_prev)",
    )
    ap.add_argument(
        "--record-dir",
        default=None,
        metavar="DIR",
        help="write the model's real inputs and outputs here: model_io.npz (per "
        "dispatched action, the state and the action; per inference, the state given "
        "to the model, that state after the preprocessor, and the whole chunk it "
        "produced, both normalised and in joint radians), model_io.json (readable "
        "summary) and inputs/ (the frames the model was given, lossless). Read it "
        "back with scripts/analyze_model_io.py. Pass an absolute path.",
    )
    ap.add_argument(
        "--no-record-images",
        action="store_true",
        help="keep only the numbers in --record-dir, not the camera frames",
    )
    args = ap.parse_args()

    modes = [bool(args.policy), bool(args.mock_policy), bool(args.replay)]
    if sum(modes) != 1:
        ap.error("pass exactly one of --policy, --mock-policy or --replay")
    if args.replay and (args.sequential or args.rtc):
        ap.error("--replay has no policy to chunk; --sequential and --rtc do not apply")
    if args.replay_stride is not None and args.replay_stride < 1:
        ap.error("--replay-stride must be at least 1")
    if args.replay_normalize and not args.replay:
        ap.error("--replay-normalize only applies to --replay")
    if args.sequential and args.rtc:
        ap.error("--sequential and --rtc are mutually exclusive: one removes chunk overlap, "
                 "the other exists to create it")
    if args.sequential and args.mock_policy:
        ap.error("--sequential needs a real policy; the mock emits one action, not a chunk")

    import chiral

    state_arms_choice = args.state_arms or args.arm
    state_arms = ["left", "right"] if state_arms_choice == "both" else [state_arms_choice]
    action_arms = ["left", "right"] if args.arm == "both" else [args.arm]
    expected_action_dim = DOF * len(action_arms)
    cameras = _parse_cameras(args.cameras)

    if args.control_hz > 10.0:
        print(
            f"warning: --control-hz {args.control_hz} exceeds the server's _CONTROL_HZ "
            "(10.0 in raiden/server.py); each command takes 1/_CONTROL_HZ to execute on a "
            "single-worker executor, so actions will fall behind. Raise _CONTROL_HZ to match.",
            file=sys.stderr,
        )

    replay = None
    roundtrip = None
    if args.replay:
        frames, fps = _load_replay_episode(
            args.replay, args.replay_episode, args.replay_field, args.revision
        )
        if frames.shape[1] != expected_action_dim:
            raise SystemExit(
                f"episode {args.replay_episode} has {frames.shape[1]}-D "
                f"{args.replay_field}, but --arm {args.arm} expects {expected_action_dim}-D"
            )
        stride = args.replay_stride
        if stride is None:
            stride = max(1, int(round((fps or args.control_hz) / args.control_hz)))
        replay = _ReplaySource(frames, stride)
        secs = len(replay) / args.control_hz
        print(
            f"replay: {args.replay} episode {args.replay_episode}, field {args.replay_field}\n"
            f"  {len(frames)} recorded frames @ {fps or '?'} fps -> stride {stride} -> "
            f"{len(replay)} actions @ {args.control_hz} Hz ({secs:.1f}s)"
        )
        if args.replay_normalize:
            print(f"  normalization round trip via {args.replay_normalize} …", flush=True)
            roundtrip = _NormalizeRoundTrip(
                *load_action_normalizers(args.replay_normalize, args.device, args.revision)
            )
            probe = roundtrip(frames[0])
            print(f"  first action round-trip error: "
                  f"{float(np.abs(probe - frames[0]).max()):.3e} rad")
            roundtrip.errors.clear()
            roundtrip.outside = roundtrip.values = 0
            roundtrip.extreme = 0.0
        gaps = np.abs(np.diff(frames[::stride][:, :6], axis=0)).max(axis=1)
        print(f"  largest step between consecutive replayed actions: {gaps.max():.4f} rad "
              f"(server e-stops above {_SERVER_MAX_JOINT_DELTA})")
        if gaps.max() > _SERVER_MAX_JOINT_DELTA:
            raise SystemExit(
                f"stride {stride} produces a {gaps.max():.4f} rad step, which the server "
                f"would e-stop on. Lower --replay-stride (1 replays every recorded frame)."
            )

    policy = torch = None
    if args.policy:
        import torch

        print(f"loading policy from {args.policy} …", flush=True)
        if not (args.task or "").strip():
            raise SystemExit(
                "--task is required: a VLA conditions on the language prompt, and an empty "
                "one still returns plausible-looking actions. Pass the dataset's own task "
                'string, e.g. --task "Pick up the red block"'
            )
        if args.runtime == "0.4.1":
            if args.rtc:
                raise SystemExit(
                    "--rtc needs lerobot.policies.rtc, which arrived after 0.4.1; "
                    "run it with --runtime 0.6.1"
                )
            if args.revision:
                raise SystemExit(
                    "--revision is resolved by the 0.6.1 loader; with --runtime 0.4.1 pass an "
                    "already-pinned local checkpoint directory to --policy"
                )
            policy = yam_policy_041.YamPi041Policy(
                args.policy,
                task=args.task,
                action_dim=expected_action_dim,
                device=args.device,
                slots=[slot for _, slot in cameras],
                dataset_meta=args.dataset_meta,
                n_action_steps=args.n_action_steps,
                family=args.policy_family,
                aida_root=args.aida_root,
            )
        else:
            policy = YamLeRobotPolicy(
                args.policy,
                task=args.task,
                action_dim=expected_action_dim,
                device=args.device,
                dataset_meta=args.dataset_meta,
                slots=[slot for _, slot in cameras],
                n_action_steps=args.n_action_steps,
                revision=args.revision,
                rtc=RTCSettings(args.rtc_horizon, args.rtc_guidance, args.rtc_schedule)
                if args.rtc else None,
            )
        if args.dataset_meta:
            print(
                f"  !! features and stats taken from {args.dataset_meta} — if this is a base\n"
                "  !! checkpoint its actions are scaled correctly but are NOT scene-responsive"
            )
        n_action_steps = policy.n_action_steps
        if args.rtc:
            print(
                f"  RTC on: horizon={args.rtc_horizon} queue_threshold={args.rtc_queue_threshold} "
                f"schedule={args.rtc_schedule} guidance={args.rtc_guidance}"
            )
        elif n_action_steps > 1:
            print(
                f"  note: {n_action_steps} actions are replayed open-loop per inference "
                f"({n_action_steps / args.control_hz:.1f}s at {args.control_hz} Hz) — "
                "lower --n-action-steps to replan sooner"
            )
            print(
                "  no rest between chunks: each one is planned from a moving arm "
                "(--sequential settles and rests first)"
            )
        else:
            print(f"  replanning every control step ({1000 / args.control_hz:.0f} ms budget)")

    with chiral.PolicyClient(args.uri) as env:
        meta = env.get_metadata()
        if meta.get("action_type") != "joint":
            raise SystemExit(
                f"server is running action_type={meta.get('action_type')!r}; "
                "restart it with: rd serve --action-type joint"
            )
        print(f"connected — cameras: {', '.join(meta.get('cameras', []))}")

        print("homing …", flush=True)
        obs, _ = env.reset()

        # Latch the arms the policy does not drive so they hold their homed pose
        # instead of being commanded to zero.
        hold = np.zeros(BIMANUAL_DOF, dtype=np.float32)
        for arm in ("left", "right"):
            value = obs.proprios.get(_PROPRIO[arm])
            if value is not None:
                hold[_SLOTS[arm]] = np.asarray(value, dtype=np.float32).reshape(-1)

        start = time.perf_counter()

        def infer(observation, capture=None):
            """One action for this step; with `capture`, also the model's real I/O."""
            state = _read_state(observation, state_arms)
            if args.mock_policy:
                out = np.concatenate(
                    [np.asarray(observation.proprios[_PROPRIO[a]], np.float32).reshape(-1) for a in action_arms]
                )
                if args.mock_amplitude:
                    out = out.copy()
                    out[0] += args.mock_amplitude * np.sin(2 * np.pi * 0.25 * (time.perf_counter() - start))
                return out

            try:
                return policy.act(state, _images(observation, cameras), capture=capture)
            except ValueError as exc:
                raise SystemExit(
                    f"{exc}\nCheck that --arm, --state-arms and --cameras match the dataset "
                    "the checkpoint was trained on."
                )

        def to_full_action(policy_action):
            """Place the policy's output into its slot of the 14-D joint action vector."""
            full = hold.copy()
            for i, arm in enumerate(action_arms):
                full[_SLOTS[arm]] = policy_action[i * DOF : (i + 1) * DOF]
            return full

        # Loop state hoisted above next_action, which reads both: `steps` to spot
        # chunk boundaries, `period` to pace the dwell.
        steps = 0
        period = 1.0 / args.control_hz
        chunk_steps = policy.n_action_steps if policy is not None else 1

        rec = None
        if args.record_dir:
            rec = IORecord(
                args.record_dir, dict(cameras), args.task or "",
                save_images=not args.no_record_images,
            )
            print(f"recording model I/O to {rec.dir}")

        # Holds the run in memory when --debug is on without --record-dir, so the
        # printed numbers still come out of the recorded arrays.
        debug_only = None

        # Last 14-D action actually dispatched — what the dwell re-commands.  A
        # list so the closures below can rebind the value.
        last_full = [None]

        def dwell(seconds, label):
            """Hold the last commanded pose for *seconds* before the next chunk.

            Sleeping alone would already hold — the server only moves when an
            action arrives — but re-sending the same target keeps the dispatch
            stream alive and leaves the server's ``prev_cmd`` equal to where the
            arm is actually resting, so the next chunk's first action is
            interpolated from that pose instead of from a stale setpoint.

            The rest also buys the arm time to finish tracking the chunk's last
            target, so the observation the next chunk is planned from is taken
            from a stationary robot.
            """
            if seconds <= 0 or last_full[0] is None:
                return
            print(f"  {label}: resting {seconds:.1f}s at the chunk's last pose", flush=True)
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                t0 = time.perf_counter()
                env.put_action(last_full[0])
                left = deadline - time.perf_counter()
                time.sleep(max(0.0, min(period - (time.perf_counter() - t0), left)))

        def _log_plan(capture, observation, latency):
            """Record one inference and print where its plan opens.

            Called at every chunk boundary, the only place the model actually
            runs -- mid-chunk steps just pop a queue.  Everything comes from
            ``capture``, the arrays the model consumed and produced, so the two
            printed distances are the ones the saved record reproduces.

            Without ``--record-dir`` a throwaway recorder holds the run in memory
            so ``--debug`` still prints the same numbers from the same arrays.
            """
            nonlocal debug_only
            target = rec
            if target is None:
                if not args.debug:
                    return
                if debug_only is None:
                    debug_only = IORecord(None, dict(cameras), args.task or "", save_images=False)
                target = debug_only
            entry = target.plan(steps, capture, _camera_ages_ms(observation), latency)
            print(IORecord.line(entry), flush=True)

        seq = None
        if args.sequential:
            seq = SequentialChunker(
                policy, steps=policy.n_action_steps,
                settle_tol=args.settle_tol, settle_timeout=args.settle_timeout,
            )
            capture = {}
            n = seq.plan(_read_state(obs, state_arms), _images(obs, cameras), capture=capture)
            print(f"  sequential mode: {n}-action chunks, settle tol {args.settle_tol} rad, "
                  f"dwell {args.chunk_dwell:.1f}s (first chunk {seq.last_latency:.2f}s)")
            _log_plan(capture, obs, seq.last_latency)

        rtc = None
        if args.rtc:
            rtc = RTCDriver(
                policy, args.control_hz, queue_threshold=args.rtc_queue_threshold,
            )
            rtc.submit(_read_state(obs, state_arms), _images(obs, cameras))
            rtc.start()
            print("  waiting for the first RTC chunk …", flush=True)
            if not rtc.wait_for_first():
                raise SystemExit("no RTC chunk produced within 120s")
            print(f"  first chunk ready ({rtc.last_latency:.2f}s), queue={rtc.qsize()}")

        def next_action(observation):
            """One action: replayed, from the sequential chunk, the RTC queue, or fresh."""
            if replay is not None:
                if replay.done():
                    return None
                action = replay.pop()
                return action if roundtrip is None else roundtrip(action)
            if seq is not None:
                if seq.needs_chunk():
                    # Let the arm arrive before observing, so the next chunk is
                    # planned from a stationary robot rather than a moving one.
                    target = seq._chunk[-1] if seq._chunk is not None else None
                    if target is not None:
                        gap, ok = seq.settle(
                            lambda: (_read_state(env.latest_obs, state_arms)
                                     if env.latest_obs is not None else None),
                            target,
                        )
                        seq.settle_gaps.append(gap)
                        print(f"  chunk {seq.chunks}: settled gap {gap:.4f} rad"
                              f"{'' if ok else '  (TIMED OUT — arm never arrived)'}", flush=True)
                    before = _camera_ages_ms(env.latest_obs) if env.latest_obs else {}
                    dwell(args.chunk_dwell, f"chunk {seq.chunks}")
                    fresh = env.latest_obs or observation
                    ages = _camera_ages_ms(fresh)
                    if ages:
                        moved = {k: before[k] - ages[k] + args.chunk_dwell * 1000
                                 for k in ages if k in before}
                        print("  image age at plan: "
                              + ", ".join(f"{k} {v:.0f}ms" for k, v in ages.items())
                              + ("  | frame advanced during the rest: "
                                 + ", ".join(f"{k} {v:.0f}ms" for k, v in moved.items())
                                 if moved else ""), flush=True)
                    state_at_plan = _read_state(fresh, state_arms)
                    capture = {}
                    started = time.perf_counter()
                    seq.plan(state_at_plan, _images(fresh, cameras), capture=capture)
                    _log_plan(capture, fresh, time.perf_counter() - started)
                return seq.pop()
            if rtc is None:
                # Default mode: select_action refills its queue only once empty, so a
                # chunk boundary is exactly every n_action_steps dispatched actions.
                # Take the freshest observation for the inference, but do not rest --
                # --chunk-dwell belongs to --sequential, which asks for a stationary arm.
                if chunk_steps > 1 and steps > 0 and steps % chunk_steps == 0:
                    observation = env.latest_obs or observation
                    ages = _camera_ages_ms(observation)
                    if ages:
                        print("  image age at inference: "
                              + ", ".join(f"{k} {v:.0f}ms" for k, v in ages.items()), flush=True)
                if steps % chunk_steps:
                    # Mid-chunk: select_action only pops, so nothing was planned here.
                    return infer(observation)
                capture = {}
                started = time.perf_counter()
                action = infer(observation, capture=capture)
                _log_plan(capture, observation, time.perf_counter() - started)
                return action
            rtc.submit(_read_state(observation, state_arms), _images(observation, cameras))
            if rtc.error is not None:
                raise SystemExit(f"RTC producer failed: {rtc.error}")
            return rtc.get()

        # Gate the first action: the server e-stops on a large jump, and a policy
        # starting from the home pose is the most likely moment to trip it.
        first_policy_action = next_action(obs)
        first = to_full_action(first_policy_action)
        if args.debug:
            print(_action_debug_line(0, first_policy_action, _read_state(obs, state_arms), None))
        for arm in action_arms:
            current = obs.proprios.get(_PROPRIO[arm])
            if current is None:
                continue
            # Arm joints only — the gripper (index 6) is a linear position, not radians.
            delta = float(np.abs(first[_SLOTS[arm]][:6] - np.asarray(current)[:6]).max())
            print(f"first action: {arm} arm max joint delta {delta:.4f} rad")
            if replay is not None:
                # Expected: a demo does not begin at the home pose.  The approach
                # ramp below closes this gap in steps the server tolerates, so the
                # distance is information, not a fault.
                print(f"  (replay: covered by the {args.replay_approach:.1f}s approach ramp)")
                continue
            if delta > args.max_joint_delta:
                raise SystemExit(
                    f"aborting before sending: {arm} delta {delta:.4f} rad exceeds "
                    f"{args.max_joint_delta:.4f}. The server would emergency-stop. Check that "
                    "--state-arms, --cameras and the training resolution match the dataset."
                )
        # The gate ran a real inference, and in --sequential it also opened the
        # first chunk.  Resetting here used to throw both away: the default mode
        # then paid for an inference nothing executed, and the sequential chunk
        # started at its second action, which is why plans landed on step 59, 119
        # rather than 60, 120.  Dispatch it instead.
        pending = [first_policy_action] if (policy is not None and replay is None) else []

        if args.dry_run:
            print(f"dry run — 14-D action would be:\n{first}")
            return

        env.start_obs_stream(hz=args.obs_hz)
        env.start_action_dispatch(hz=args.control_hz)

        if replay is not None:
            # Walk from wherever homing left the arm to the episode's opening pose in
            # steps below the server's e-stop threshold, then wait for the arm to
            # arrive so the replay starts from the pose the demo started from.
            start = np.zeros(BIMANUAL_DOF, dtype=np.float32)
            for arm in ("left", "right"):
                value = (env.latest_obs or obs).proprios.get(_PROPRIO[arm])
                if value is not None:
                    start[_SLOTS[arm]] = np.asarray(value, dtype=np.float32).reshape(-1)
            span = float(np.abs(first[:6] - start[:6]).max())
            n = max(1, int(round(args.replay_approach * args.control_hz)))
            safe = _SERVER_MAX_JOINT_DELTA * 0.5
            if span / n > safe:
                n = int(math.ceil(span / safe))
                print(f"  extending the approach to {n / args.control_hz:.1f}s to keep each "
                      f"step under {safe:.2f} rad")
            print(f"approaching the episode's first pose ({span:.4f} rad) over "
                  f"{n / args.control_hz:.1f}s …", flush=True)
            for i in range(1, n + 1):
                t0 = time.perf_counter()
                env.put_action(start + (first - start) * (i / n))
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            last_full[0] = first.copy()
            dwell(max(args.chunk_dwell, 0.5), "approach")
            reached = _read_state(env.latest_obs or obs, action_arms)
            gap = float(np.abs(reached[:6] - first[_SLOTS[action_arms[0]]][:6]).max())
            print(f"  arrived: residual {gap:.4f} rad — starting replay", flush=True)

        prev_action = first_policy_action  # so step 1 reports a real d_prev
        print(f"running at {args.control_hz} Hz — Ctrl-C to stop")
        try:
            while args.max_steps < 0 or steps < args.max_steps:
                t0 = time.perf_counter()
                latest = env.latest_obs
                if latest is None:
                    time.sleep(0.005)
                    continue
                if replay is not None and replay.done():
                    # Frame 0 is reached by the approach ramp, so the loop dispatches
                    # the remaining len(replay) - 1.
                    print(f"replay finished: {steps} action(s) sent, "
                          f"{len(replay)} frames in the episode")
                    break
                policy_action = pending.pop() if pending else next_action(latest)
                if policy_action is None:
                    # Queue ran dry — hold the last target rather than command a
                    # stale or zero action; the server keeps the previous setpoint.
                    time.sleep(0.005)
                    continue
                if args.debug:
                    print(
                        _action_debug_line(
                            steps + 1,
                            policy_action,
                            _read_state(latest, state_arms),
                            prev_action,
                        ),
                        flush=True,
                    )
                prev_action = policy_action
                last_full[0] = to_full_action(policy_action)
                env.put_action(last_full[0])
                for target in (rec, debug_only):
                    if target is not None:
                        target.step(steps, _read_state(latest, state_arms), policy_action)
                steps += 1
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            if roundtrip is not None and roundtrip.errors:
                e = np.array(roundtrip.errors)
                pct = 100.0 * roundtrip.outside / max(1, roundtrip.values)
                print(
                    f"normalization round trip: {len(e)} action(s), error "
                    f"med={np.median(e):.3e} max={e.max():.3e} rad\n"
                    f"  normalized values outside [-1,1]: {pct:.1f}% "
                    f"(largest |value| {roundtrip.extreme:.2f}) — pi05's state tokenizer "
                    "clips at 1.0"
                )
            if seq is not None and seq.settle_gaps:
                g = np.array(seq.settle_gaps)
                print(
                    f"sequential: {seq.chunks} chunk(s), settle gap "
                    f"med={np.median(g):.4f} max={g.max():.4f} rad, "
                    f"{seq.settle_timeouts} timeout(s)  [{TRAIN_REFERENCE}]"
                )
            if rtc is not None:
                rtc.stop()
                gaps = [g for g in rtc.replan_gaps if g > 0]
                interval = (sum(gaps) / len(gaps)) if gaps else float("nan")
                print(
                    f"RTC: {rtc.inferences} chunk(s), last latency {rtc.last_latency:.2f}s, "
                    f"{rtc.stalls} stall(s), replanned every {interval:.1f} steps "
                    f"({interval / args.control_hz:.2f}s of plan age)"
                )
            if rec is not None:
                out = rec.save()
                if out is not None:
                    d_first = [e["d_first"] for e in rec.plans]
                    d_last = [e["d_last"] for e in rec.plans if not math.isnan(e["d_last"])]
                    if d_first:
                        print(
                            f"model I/O: {len(rec.plans)} inference(s), "
                            f"d_first med={np.median(d_first):.4f} max={np.max(d_first):.4f}"
                            + (f", d_last med={np.median(d_last):.4f} "
                               f"max={np.max(d_last):.4f}" if d_last else "")
                            + f"  [{TRAIN_REFERENCE}]"
                        )
                    print(f"model I/O written to {out}")
            print(f"{steps} action(s) sent; disconnecting (the server e-stops on disconnect)")


if __name__ == "__main__":
    main()
