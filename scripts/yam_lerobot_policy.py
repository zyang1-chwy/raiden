"""Run a LeRobot checkpoint on a YAM arm -- the model half, with no robot in it.

Everything here talks to LeRobot and numpy and nothing else: no chiral, no
proprio stream names, no arm slots, no dispatch loop.  ``scripts/lerobot_client.py``
owns that wiring and calls into this module, which keeps the two failure modes
apart -- a policy that plans badly, and a robot that executes badly -- and lets
the model path be exercised from recorded frames with no hardware attached.

The split mirrors the simulator side in the AIDA repo, where
``src/sim/policies/yam_lerobot.py`` holds the adapter and
``scripts/sim/run_lerobot_policy_yam.py`` holds the driver.

An observation is a joint-space ``state`` vector plus ``images``, a dict keyed by
the model's own image slots (``base_0_rgb``, ``left_wrist_0_rgb``); mapping a
camera's name onto a slot is the caller's job.  :class:`YamLeRobotPolicy` packs
that into the frame the processors read, runs the model, and hands back a
plain ``(action_dim,)`` command.

Three ways to get actions out of it:

``policy.act(state, images)``
    One action per call, from the queue LeRobot refills whenever it empties --
    so one inference per ``n_action_steps`` calls.
``policy.chunk(state, images)``
    The whole chunk at once, for a caller that wants to own the execution.
    :class:`SequentialChunker` runs one to completion between inferences.
``RTCDriver``
    Real-Time Chunking: the next chunk is generated on a background thread while
    the current one runs, and blended into its tail.

Pass a ``capture`` dict to ``act`` or ``chunk`` and it comes back holding what
actually crossed the model boundary -- the frame, the normalised state, and the
chunk both as the model emitted it and in joint radians.  :class:`IORecord`
stores those and computes every distance from them.
"""

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOF = 7  # 6 revolute joints + 1 gripper, per arm

# How far the commanded action sits from the measured state in the demonstrations,
# as the largest gap over the six arm joints in a frame.  Measured over all 20,822
# frames of YzyLmc/red_block_only.  It is the yardstick for every distance this
# module reports: the command leads the follower by about this much everywhere in
# an episode, including while the arm is lifting after a grasp.
TRAIN_ACTION_STATE = {"p50": 0.031, "p99": 0.070, "max": 0.094}
TRAIN_REFERENCE = (f"training |action-state| p50 {TRAIN_ACTION_STATE['p50']}, "
                   f"p99 {TRAIN_ACTION_STATE['p99']}, max {TRAIN_ACTION_STATE['max']}")
STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."


def build_frame(state, images, task):
    """Pack one observation into the frame LeRobot's processors read.

    ``images`` maps model image slots to HWC uint8 RGB arrays.  Everything is
    checked here because the failures are otherwise silent: a stray NaN passes
    straight through the normaliser, and an empty task string leaves a VLA
    conditioned on nothing at all while still returning plausible-looking
    actions.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if not state.size:
        raise ValueError("state is empty.")
    if not np.isfinite(state).all():
        raise ValueError("state must be finite.")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string -- a VLA conditions on it.")
    if not images:
        raise ValueError("no camera images: a VLA needs at least one.")

    frame = {STATE_KEY: state}
    for slot, image in images.items():
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{slot} image must be HWC RGB, got shape {array.shape}.")
        frame[IMAGE_PREFIX + slot] = np.ascontiguousarray(array)
    return frame


def extract_action(raw_action, action_dim):
    """Return the ``(action_dim,)`` command from what the model emitted."""
    action = np.asarray(raw_action, dtype=np.float32)
    if action.ndim == 2:
        action = action[0]
    if not np.isfinite(action).all():
        raise ValueError("model action must be finite.")
    if action.shape[0] != action_dim:
        raise ValueError(
            f"policy emitted a {action.shape[0]}-D action but {action_dim}-D was expected. "
            "A 32-D action means an unfinetuned base checkpoint (32 = max_action_dim "
            "padding); finetune on the exported dataset first."
        )
    return action


@dataclass
class RTCSettings:
    """Real-Time Chunking knobs, kept apart from the CLI that fills them in."""

    horizon: int = 10
    guidance: float = 10.0
    schedule: str = "LINEAR"


def enable_rtc(cfg, rtc, path):
    """Attach an RTC config so the policy builds its RTCProcessor.

    Must happen before the policy is constructed: ``init_rtc_processor`` runs in
    ``__init__`` and only wires the processor in when ``rtc_config`` is present.
    """
    if rtc is None:
        return cfg
    if not hasattr(cfg, "rtc_config"):
        raise SystemExit(
            f"RTC needs a flow-matching policy (pi0, pi05, smolvla); "
            f"{getattr(cfg, 'type', '?')} in {path} has no rtc_config"
        )
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc import RTCConfig

    cfg.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=rtc.horizon,
        max_guidance_weight=rtc.guidance,
        prefix_attention_schedule=RTCAttentionSchedule(rtc.schedule),
    )
    return cfg


def _stats_width(pipeline, key):
    """Return the width of ``key``'s normalization statistics, or None if absent."""
    for step in getattr(pipeline, "steps", []):
        stats = getattr(step, "stats", None) or {}
        entry = stats.get(key)
        if not entry:
            continue
        for field in ("q01", "q99", "mean", "std", "min", "max"):
            value = entry.get(field) if hasattr(entry, "get") else None
            if value is not None and np.ndim(value) > 0:
                return int(np.asarray(value).reshape(-1).shape[0])
    return None


def _align_action_dim(cfg, postprocessor, path):
    """Narrow a padded ``action`` feature to the width the saved statistics imply.

    pi0-family checkpoints finetuned from a base can keep the base config's
    padded ``max_action_dim`` (32) in ``output_features`` while their statistics
    are the real robot's width.  The policy unpads to the *config* width, so the
    unnormalizer then hits ``size of tensor a (32) must match tensor b (7)``.
    The statistics come from the training dataset and are authoritative, so the
    config is corrected to match them.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    feature = cfg.output_features.get("action")
    width = _stats_width(postprocessor, "action")
    if feature is None or width is None or feature.shape[0] == width:
        return cfg

    print(
        f"  action width {feature.shape[0]} in {path} config disagrees with its "
        f"{width}-D statistics; using {width} (padded config, real stats)"
    )
    cfg.output_features = dict(cfg.output_features)
    cfg.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(width,))
    return cfg


def _resolve_revision(path, revision):
    """Resolve a Hub repo id at a specific revision to a local snapshot directory.

    Pinning matters once a repo accumulates checkpoints: successive training runs
    are pushed as new commits to the same repo id, so ``main`` silently moves.
    Resolving to a path up front also means every downstream loader — config,
    weights, processors — sees the same revision without threading the argument
    through each one.
    """
    import os

    if not revision:
        return path
    if os.path.isdir(path):
        raise SystemExit(f"--revision applies to Hub repos, but {path} is a local directory")

    from huggingface_hub import snapshot_download

    resolved = snapshot_download(path, revision=revision)
    print(f"  revision {revision} -> {resolved}")
    return resolved


def _set_action_steps(cfg, n_action_steps, path):
    """Override how many predicted actions are executed before re-inferring.

    ``chunk_size`` is how many steps the model was trained to predict and is not
    adjustable after training; ``n_action_steps`` is how many of that chunk get
    executed before the policy looks at a fresh observation.  Lowering it tightens
    the loop — at the limit of 1 the policy replans every control step.
    """
    if n_action_steps is None:
        return cfg
    chunk = getattr(cfg, "chunk_size", None)
    if chunk is not None and n_action_steps > chunk:
        raise SystemExit(
            f"--n-action-steps {n_action_steps} exceeds this checkpoint's chunk_size "
            f"({chunk}); the model only predicts {chunk} steps per inference"
        )
    if n_action_steps < 1:
        raise SystemExit("--n-action-steps must be at least 1")
    cfg.n_action_steps = n_action_steps
    return cfg


def check_normalization_stats(pipeline, path, kind):
    """Refuse to run a checkpoint whose normalizer carries no dataset statistics."""
    for step in getattr(pipeline, "steps", []):
        if "Normalizer" not in type(step).__name__ and "Unnormalizer" not in type(step).__name__:
            continue
        if not getattr(step, "stats", None):
            raise SystemExit(
                f"{path} has an empty {kind} normalizer — no dataset statistics.\n"
                "This is a base checkpoint meant for finetuning, not for direct rollout: "
                "state would be discretized against the wrong range and actions would come "
                "back in normalized units instead of joint radians.\n"
                "Finetune it on your exported dataset first (rd export_lerobot -> lerobot-train "
                "--policy.path=<base>), then point --policy at the finetuned checkpoint."
            )


def _dataset_features(root, slots):
    """Build LeRobot input/output features from a dataset's ``meta/info.json``.

    Only the cameras actually being fed are declared, so the policy config and
    the batch cannot disagree.
    """
    import json

    from lerobot.configs.types import FeatureType, PolicyFeature

    with open(f"{root}/meta/info.json") as f:
        feats = json.load(f)["features"]

    wanted = set(slots)
    inputs = {}
    for dataset_name in sorted(wanted):
        key = f"observation.images.{dataset_name}"
        if key not in feats:
            raise SystemExit(
                f"{root} has no feature {key!r}; it has: "
                + ", ".join(k for k in feats if k.startswith("observation.images."))
            )
        h, w, c = feats[key]["shape"]
        inputs[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))

    for key, kind in (("observation.state", FeatureType.STATE),):
        if key in feats:
            inputs[key] = PolicyFeature(type=kind, shape=tuple(feats[key]["shape"]))
    outputs = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(feats["action"]["shape"]))
    }
    return inputs, outputs


def _dataset_stats(root):
    """Load ``meta/stats.json`` as the normalization statistics."""
    import json

    with open(f"{root}/meta/stats.json") as f:
        raw = json.load(f)
    return {k: {kk: np.asarray(vv, dtype=np.float32) for kk, vv in v.items()} for k, v in raw.items()}


def numpy_of(tensor):
    """A detached float32 numpy copy of a torch tensor, or None."""
    if tensor is None:
        return None
    return np.asarray(tensor.detach().to("cpu").float().numpy(), dtype=np.float32)


def capture_io(raw, processed, chunk_normalized, chunk):
    """The arrays that actually entered and left the model, as numpy.

    ``raw`` is what the client handed in (uint8 frames, joint-space state);
    ``processed`` is what the preprocessor made of it, so the normalised state
    the policy was really conditioned on is kept alongside the raw one -- a
    correct-looking state can still normalise to nonsense.  ``chunk_normalized``
    is the model's own output before the unnormaliser, ``chunk`` the same thing
    in joint radians.  Every distance is computed from these, never from a
    separate running tally.
    """
    state = processed.get("observation.state") if hasattr(processed, "get") else None
    return {
        "input_state": np.asarray(raw["observation.state"], dtype=np.float32).reshape(-1).copy(),
        "input_state_normalized": (
            None if state is None else numpy_of(state).reshape(-1)
        ),
        "input_images": {
            key[len("observation.images."):]: value
            for key, value in raw.items()
            if key.startswith("observation.images.")
        },
        "output_chunk_normalized": numpy_of(chunk_normalized),
        "output_chunk": np.asarray(chunk, dtype=np.float32),
    }


class IORecord:
    """Everything the model saw and produced, written out for reading back after a run.

    Two streams are kept.  One row per dispatched action -- the measured state
    that was current when it was sent, the action itself, and the wall clock --
    so the executed rate and the arm's tracking are recoverable offline.  And one
    entry per inference: the frames and state the model consumed, the normalised
    state it was actually conditioned on, and the whole chunk it produced, both
    as the model emitted it and in joint radians.

    The two gaps below are derived from those stored arrays and from nothing
    else -- ``d_last`` reads the previous dispatched action back out of the step
    stream -- so ``scripts/analyze_model_io.py`` recomputes exactly the numbers
    printed live, and a disagreement means the recording is wrong rather than
    the arm.

    ``d_first``
        ``max|chunk[0] - state at plan|`` over the arm joints: how far the new
        chunk's opening target sits from the arm at the moment it is planned.  A
        plan that opens behind the arm drags it backwards, which is what a revert
        looks like from the outside.
    ``d_last``
        ``max|last action of the previous chunk - state at plan|``: the arm's
        tracking error when the plan was made, i.e. how much of the previous
        chunk the arm had not executed yet.  In the training data
        ``|action - state|`` stayed under 0.094 rad (p50 0.031), so a
        ``d_last`` far above that means the arm is chasing targets it never
        reached and every plan is being made from a stale pose.

    Grippers are excluded from both gaps -- index 6 of each arm is a linear
    position, not radians -- but kept in every saved vector.
    """

    MAX_IMAGE_PAIRS = 500

    def __init__(self, out_dir, cameras, task, save_images=True):
        # out_dir None keeps the run in memory: --debug prints from the same
        # arrays a --record-dir run would write.
        self.dir = None if out_dir is None else Path(out_dir)
        self.image_dir = None if self.dir is None else self.dir / "inputs"
        self.save_images = save_images and self.dir is not None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.save_images:
                self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = dict(cameras)
        self.task = task
        self.t0 = time.perf_counter()
        self.steps = []
        self.plans = []
        self._pairs = 0

    @staticmethod
    def joint_gap(a, b):
        """Largest arm-joint difference between two command-space vectors."""
        if a is None or b is None:
            return float("nan")
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        n = min(a.shape[0], b.shape[0])
        joints = [i for i in range(n) if i % DOF != 6]
        return float(np.abs(a[joints] - b[joints]).max())

    def step(self, index, state, action):
        """Record one dispatched action and the state it was sent from."""
        self.steps.append(
            (
                int(index),
                time.perf_counter() - self.t0,
                np.asarray(state, dtype=np.float32).reshape(-1).copy(),
                np.asarray(action, dtype=np.float32).reshape(-1).copy(),
            )
        )

    def plan(self, index, capture, camera_ages, latency):
        """Record one inference from the arrays it actually consumed and produced.

        ``capture`` comes from :func:`capture_io`.  The previous chunk's last
        action is read back out of the step stream rather than tracked
        separately, so ``d_last`` measures what was really dispatched.
        """
        state = np.asarray(capture["input_state"], dtype=np.float32).reshape(-1)
        chunk = np.asarray(capture["output_chunk"], dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        prev_last = self.steps[-1][3] if self.steps else None
        entry = {
            "plan": len(self.plans),
            "step": int(index),
            "t": time.perf_counter() - self.t0,
            "latency_s": float(latency),
            "state": state.copy(),
            "state_normalized": capture.get("input_state_normalized"),
            "chunk": chunk.copy(),
            "chunk_normalized": capture.get("output_chunk_normalized"),
            "prev_last": None if prev_last is None else prev_last.copy(),
            "d_first": self.joint_gap(chunk[0], state),
            "d_last": self.joint_gap(prev_last, state),
            "camera_age_ms": dict(camera_ages or {}),
            "images": {},
        }
        if self.save_images and self._pairs < self.MAX_IMAGE_PAIRS:
            entry["images"] = self._write_images(entry["plan"], capture["input_images"])
            self._pairs += 1
            if self._pairs == self.MAX_IMAGE_PAIRS:
                print(f"  (image cap reached: {self.MAX_IMAGE_PAIRS} plans saved, "
                      "later plans keep numbers only)", flush=True)
        self.plans.append(entry)
        return entry

    def _write_images(self, plan, images):
        """Save the frames the model was given, under the slot name it read them as.

        These are the arrays out of the observation dict, not a re-read of the
        camera, so the PNG on disk is the input.  Lossless, so it can be fed back
        through the policy to reproduce the plan.
        """
        from PIL import Image

        written = {}
        for slot, image in images.items():
            array = np.asarray(image)
            if array.dtype != np.uint8:
                array = array.clip(0, 255).astype(np.uint8)
            path = self.image_dir / f"plan_{plan:04d}_{slot}.png"
            Image.fromarray(array).save(path)
            written[slot] = path.name
        return written

    @staticmethod
    def line(entry):
        """The boundary summary: where the plan opens, and where the arm is."""
        delta = entry["chunk"][0] - entry["state"]
        return (
            f"  plan {entry['plan']:3d} @ step {entry['step']:5d}: "
            f"d_first={entry['d_first']:7.4f}  d_last={entry['d_last']:7.4f}  "
            f"infer={entry['latency_s'] * 1e3:5.0f}ms"
            + f"  [{TRAIN_REFERENCE}]\n"
            f"    first action - state: "
            + np.array2string(delta, precision=3, suppress_small=True,
                              max_line_width=250, separator=" ")
        )

    def save(self):
        """Write the record; returns the directory or None when there is nothing to write."""
        if self.dir is None or (not self.steps and not self.plans):
            return None
        arrays = {}
        if self.steps:
            arrays["step_index"] = np.array([s[0] for s in self.steps], dtype=np.int32)
            arrays["step_t"] = np.array([s[1] for s in self.steps], dtype=np.float32)
            arrays["step_state"] = np.stack([s[2] for s in self.steps])
            arrays["step_action"] = np.stack([s[3] for s in self.steps])
        if self.plans:
            arrays["plan_step"] = np.array([e["step"] for e in self.plans], dtype=np.int32)
            arrays["plan_t"] = np.array([e["t"] for e in self.plans], dtype=np.float32)
            arrays["plan_latency_s"] = np.array(
                [e["latency_s"] for e in self.plans], dtype=np.float32
            )
            arrays["plan_state"] = np.stack([e["state"] for e in self.plans])
            arrays["plan_chunk"] = np.stack([e["chunk"] for e in self.plans])
            for key, field in (
                ("plan_state_normalized", "state_normalized"),
                ("plan_chunk_normalized", "chunk_normalized"),
            ):
                values = [e[field] for e in self.plans]
                if all(v is not None for v in values) and len({np.shape(v) for v in values}) == 1:
                    arrays[key] = np.stack(values)
            width = arrays["plan_state"].shape[1]
            arrays["plan_prev_last"] = np.stack(
                [
                    np.full(width, np.nan, dtype=np.float32) if e["prev_last"] is None
                    else e["prev_last"]
                    for e in self.plans
                ]
            )
            arrays["plan_d_first"] = np.array(
                [e["d_first"] for e in self.plans], dtype=np.float32
            )
            arrays["plan_d_last"] = np.array(
                [e["d_last"] for e in self.plans], dtype=np.float32
            )
        np.savez_compressed(self.dir / "model_io.npz", **arrays)

        d_first = [e["d_first"] for e in self.plans]
        d_last = [e["d_last"] for e in self.plans if not math.isnan(e["d_last"])]
        elapsed = self.steps[-1][1] if self.steps else 0.0
        summary = {
            "task": self.task,
            "cameras": dict(self.cameras),
            "actions_dispatched": len(self.steps),
            "inferences": len(self.plans),
            "elapsed_s": round(elapsed, 3),
            "dispatch_hz": round(len(self.steps) / elapsed, 2) if elapsed > 0 else None,
            "d_first": {
                "median": float(np.median(d_first)) if d_first else None,
                "max": float(np.max(d_first)) if d_first else None,
            },
            "d_last": {
                "median": float(np.median(d_last)) if d_last else None,
                "max": float(np.max(d_last)) if d_last else None,
            },
            "training_reference": dict(TRAIN_ACTION_STATE),
            "npz_arrays": {
                "step_state": "measured state when each action was dispatched",
                "step_action": "the action dispatched, joint radians",
                "plan_state": "state the model was given at each inference",
                "plan_state_normalized": "the same state after the preprocessor",
                "plan_chunk": "the full chunk the model produced, joint radians",
                "plan_chunk_normalized": "the same chunk as the model emitted it",
                "plan_prev_last": "last action dispatched before the inference",
                "plan_d_first": "recomputable: |plan_chunk[:,0] - plan_state|",
                "plan_d_last": "recomputable: |plan_prev_last - plan_state|",
            },
            "plans": [
                {
                    "plan": e["plan"],
                    "step": e["step"],
                    "t": round(e["t"], 3),
                    "latency_s": round(e["latency_s"], 3),
                    "d_first": round(e["d_first"], 5),
                    "d_last": None if math.isnan(e["d_last"]) else round(e["d_last"], 5),
                    "camera_age_ms": {
                        k: round(v, 1) for k, v in e["camera_age_ms"].items()
                    },
                    "images": e["images"],
                }
                for e in self.plans
            ],
        }
        (self.dir / "model_io.json").write_text(json.dumps(summary, indent=2) + "\n")
        return self.dir


def _fit_prefix(prev_actions, horizon):
    """Pad or truncate the leftover prefix to the fixed execution horizon."""
    steps, action_dim = prev_actions.shape
    if steps == horizon:
        return prev_actions
    if steps > horizon:
        return prev_actions[:horizon]
    import torch

    padded = torch.zeros(
        (horizon, action_dim), dtype=prev_actions.dtype, device=prev_actions.device
    )
    padded[:steps] = prev_actions
    return padded




class YamLeRobotPolicy:
    """A LeRobot checkpoint, loaded once and asked for actions in joint space.

    Owns the model, both processor pipelines and the config, so a caller never
    has to carry the four of them around together -- which is what let the
    client drift into calling the pipelines by hand.

    Since LeRobot 0.6 normalisation and language tokenisation live in the
    processors, so inference is ``preprocessor -> model -> postprocessor``;
    calling the model on a raw batch skips normalisation and leaves a VLA with no
    language tokens at all.  ``prepare`` is the one place that ordering lives.
    """

    def __init__(self, path, *, task, action_dim, device="cuda", dataset_meta=None,
                 slots=(), n_action_steps=None, revision=None, rtc=None,
                 robot_type="yam", verbose=True):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string -- a VLA conditions on it.")
        self.task = task
        self.action_dim = int(action_dim)
        self.device = device
        self.robot_type = robot_type
        self.model, self.preprocessor, self.postprocessor, self.config, self.path = _load(
            path, device, dataset_meta=dataset_meta, slots=slots,
            n_action_steps=n_action_steps, revision=revision, rtc=rtc, verbose=verbose,
        )
        check_normalization_stats(self.preprocessor, self.path, "input")
        check_normalization_stats(self.postprocessor, self.path, "output")
        self.model.reset()

    @property
    def n_action_steps(self):
        """How many actions are executed per inference."""
        return int(getattr(self.config, "n_action_steps", 1) or 1)

    @property
    def chunk_size(self):
        """How many actions the model predicts per inference."""
        return getattr(self.config, "chunk_size", None)

    @property
    def type(self):
        return getattr(self.config, "type", "?")

    def reset(self):
        """Drop the queued chunk so the next call runs a fresh inference."""
        self.model.reset()

    def prepare(self, state, images):
        """Pack one observation and run it through the preprocessor.

        Returns the raw frame -- what a recorder saves -- and the processed
        batch, which is what the model actually reads.
        """
        import torch

        from lerobot.policies.utils import prepare_observation_for_inference

        raw = build_frame(state, images, self.task)
        batch = prepare_observation_for_inference(
            dict(raw), torch.device(self.device), self.task, self.robot_type
        )
        batch["task"] = [self.task]
        return raw, self.preprocessor(batch)

    def act(self, state, images, capture=None):
        """One action, from the queue the model refills whenever it empties.

        With ``capture``, the whole chunk is snapshotted out of that queue at the
        moment it is refilled -- the model's own output, not a second forward
        pass, so recording costs nothing and cannot disagree with what ran.
        """
        import torch

        raw, batch = self.prepare(state, images)
        with torch.inference_mode():
            popped = self.model.select_action(batch)
            queued = None
            if capture is not None:
                queued = [popped.detach().clone()] + [
                    q.detach().clone() for q in getattr(self.model, "_action_queue", [])
                ]
            action = self.postprocessor(popped)
            if queued is not None:
                chunk_normalized = torch.cat([q.reshape(1, -1) for q in queued], dim=0)
                chunk = self.postprocessor(chunk_normalized)
                capture.update(capture_io(raw, batch, chunk_normalized, numpy_of(chunk)))
        return extract_action(numpy_of(action), self.action_dim)

    def chunk(self, state, images, capture=None):
        """The whole chunk at once, ``(T, action_dim)``, for a caller that executes it."""
        import torch

        raw, batch = self.prepare(state, images)
        with torch.no_grad():
            raw_chunk = self.model.predict_action_chunk(batch)
            chunk = self.postprocessor(raw_chunk).squeeze(0)
        chunk = numpy_of(chunk)
        if chunk.shape[-1] != self.action_dim:
            extract_action(chunk[0], self.action_dim)  # raises with the same explanation
        if capture is not None:
            capture.update(capture_io(raw, batch, raw_chunk.squeeze(0), chunk))
        return chunk


def _load(path, device, *, dataset_meta, slots, n_action_steps, revision, rtc, verbose):
    """Load a LeRobot policy together with its pre/post-processor pipelines.

    A finetuned checkpoint carries its own feature shapes and statistics, so the
    processors are loaded straight from it.  ``dataset_meta`` overrides both from
    a dataset on disk, which is what makes a *base* checkpoint runnable for
    bring-up: it reshapes the padded action space to the robot's real width and
    supplies the statistics the base checkpoint ships without.  The resulting
    actions are correctly scaled but carry no learned behaviour.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = _resolve_revision(path, revision)
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if dataset_meta:
        cfg.input_features, cfg.output_features = _dataset_features(dataset_meta, slots)

    # Processors first: their statistics reveal the robot's true action width,
    # which the policy needs before it decides how far to unpad its output.
    overrides = {"device_processor": {"device": device}}
    if dataset_meta:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            dataset_stats=_dataset_stats(dataset_meta),
            preprocessor_overrides=overrides,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path, preprocessor_overrides=overrides
        )
        cfg = _align_action_dim(cfg, postprocessor, path)

    cfg = _set_action_steps(cfg, n_action_steps, path)
    cfg = enable_rtc(cfg, rtc, path)
    model = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    model.to(device)
    model.eval()
    if verbose:
        print(f"  type={cfg.type}  n_action_steps={getattr(cfg, 'n_action_steps', 1)}"
              f"  chunk_size={getattr(cfg, 'chunk_size', '?')}")
    return model, preprocessor, postprocessor, cfg, path


class RTCDriver:
    """Asynchronous chunk production — Real-Time Chunking (Black et al., 2506.07339).

    Ordinary chunking stalls: the policy consumes a whole chunk, blocks for an
    inference, then jumps onto a chunk that was conditioned on an observation
    now several steps stale.  That shows up as the arm freezing and snapping
    back.  RTC removes the stall by generating the *next* chunk on a background
    thread while the current one is still executing, and by conditioning that
    generation on the part of the current chunk that will still be unexecuted
    when it lands — an inpainting problem, solved with prefix attention so the
    new chunk agrees with the old one where they overlap and is free to diverge
    afterwards.

    The math lives in LeRobot (``RTCProcessor.denoise_step``, reached through
    ``predict_action_chunk(inference_delay=..., prev_chunk_left_over=...)``) and
    the queue bookkeeping in ``ActionQueue``.  This class is only the driver:
    LeRobot's own driver is bound to its robot-wrapper framework, so the loop is
    reproduced here against the chiral observation stream.

    ``inference_delay`` is how many control steps the pending inference is
    expected to take, predicted from observed latency; ``ActionQueue.merge``
    then discards exactly that many actions from the front of the new chunk,
    since the robot consumed them while the model was thinking.
    """

    def __init__(self, policy, fps, queue_threshold=None):
        from lerobot.policies.rtc import ActionQueue, LatencyTracker

        rtc_config = policy.config.rtc_config
        self._policy = policy
        self._fps = fps
        self._horizon = rtc_config.execution_horizon
        # How full the queue may be before the next inference starts.  This sets
        # the replan interval, and it is the parameter that decides whether the
        # arm acts on fresh observations: roughly (chunk_length - threshold)
        # actions are executed between replans.  Setting it too low leaves the
        # policy running a stale plan for seconds, which reads as the arm moving
        # correctly and then reverting when a fresh chunk finally disagrees.
        self._threshold = queue_threshold if queue_threshold is not None else 40
        self._queue = ActionQueue(rtc_config)
        self._latency = LatencyTracker()
        self._obs = None
        self._obs_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.inferences = 0
        self.stalls = 0
        self.served = 0
        self.replan_gaps = []
        self.last_latency = None
        self.error = None
        self._served_at_last_merge = 0

    def submit(self, state, images):
        """Hand the newest observation to the producer thread."""
        with self._obs_lock:
            self._obs = (np.asarray(state, dtype=np.float32).copy(), dict(images))

    def get(self):
        """Pop the next action, or None when the queue has run dry."""
        action = self._queue.get()
        if action is None:
            self.stalls += 1
            return None
        self.served += 1
        return np.asarray(action.to("cpu").float().numpy(), dtype=np.float32)

    def qsize(self):
        return self._queue.qsize()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtc-producer")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def wait_for_first(self, timeout=120.0):
        """Block until the first chunk lands, so the caller can gate on it."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.error is not None:
                raise SystemExit(f"RTC producer failed: {self.error}")
            if self._queue.qsize() > 0:
                return True
            time.sleep(0.02)
        return False

    def _loop(self):
        step_seconds = 1.0 / self._fps
        while not self._stop.is_set():
            with self._obs_lock:
                observation = self._obs
            if observation is None or self._queue.qsize() > self._threshold:
                time.sleep(0.005)
                continue
            try:
                started = time.perf_counter()
                index_before = self._queue.get_action_index()
                prefix = self._queue.get_left_over()
                # Predict the delay from the worst latency seen so far, so the
                # chunk is generated for where the robot *will* be, not where it is.
                observed = self._latency.max()
                delay = math.ceil(observed / step_seconds) if observed else 0
                if prefix is not None:
                    prefix = _fit_prefix(prefix, self._horizon)

                _, batch = self._policy.prepare(*observation)

                # Deliberately NOT torch.inference_mode(): RTC's guidance term is
                # a torch.autograd.grad through the denoiser, taken under an inner
                # torch.enable_grad().  no_grad (already on predict_action_chunk)
                # can be locally overridden that way; inference_mode cannot — its
                # tensors are permanently barred from autograd, and the second
                # chunk onward dies with "does not have a grad_fn".
                chunk = self._policy.model.predict_action_chunk(
                    batch, inference_delay=delay, prev_chunk_left_over=prefix
                )
                original = chunk.squeeze(0).detach().clone()
                processed = self._policy.postprocessor(chunk).squeeze(0).detach()

                latency = time.perf_counter() - started
                self._latency.add(latency)
                self.last_latency = latency
                self.inferences += 1
                self._queue.merge(
                    original, processed, math.ceil(latency / step_seconds), index_before
                )
                self.replan_gaps.append(self.served - self._served_at_last_merge)
                self._served_at_last_merge = self.served
            except Exception as e:  # surface to the control loop rather than dying quietly
                self.error = f"{type(e).__name__}: {e}"
                return


class SequentialChunker:
    """Execute one chunk to completion, let the arm settle, then plan the next.

    A diagnostic rather than a way to run: it strips out every timing effect —
    no blending, no inference-delay accounting, no plan executed against a moved
    robot.  Each chunk is generated from an observation taken while the arm is
    stationary at the end of the previous chunk, which is the closest inference
    ever gets to the conditions a quasi-static demonstration was recorded under.

    If the arm still reverts under this, chunking is not the cause and the
    policy is.  If it stops reverting, the fault is in the timing path.

    The settle wait is what makes it a fair test.  Dispatching the last action
    of a chunk is not the same as reaching it — the server needs a control
    period plus tracking time — so observing immediately would feed the next
    chunk a state the arm is still moving through.
    """

    def __init__(self, policy, steps, settle_tol, settle_timeout):
        self._policy = policy
        self._steps = steps
        self._settle_tol = settle_tol
        self._settle_timeout = settle_timeout
        self._chunk = None
        self._index = 0
        self.chunks = 0
        self.settle_gaps = []
        self.settle_timeouts = 0
        self.last_latency = None

    def needs_chunk(self):
        return self._chunk is None or self._index >= len(self._chunk)

    def settle(self, read_state, target):
        """Block until the arm reaches *target*, or the timeout expires.

        Returns the residual gap, which is also the tracking measurement: in
        training ``|action - state|`` stayed under 0.094 rad, so a residual
        far above that means the arm never arrived, not that the policy is wrong.
        """
        deadline = time.perf_counter() + self._settle_timeout
        gap = float("inf")
        while time.perf_counter() < deadline:
            state = read_state()
            if state is None:
                time.sleep(0.01)
                continue
            n = min(len(state), len(target))
            joints = [i for i in range(n) if i % DOF != 6]  # skip grippers
            gap = float(np.abs(np.asarray(state)[joints] - np.asarray(target)[joints]).max())
            if gap <= self._settle_tol:
                return gap, True
            time.sleep(0.02)
        self.settle_timeouts += 1
        return gap, False

    def plan(self, state, images, capture=None):
        started = time.perf_counter()
        chunk = self._policy.chunk(state, images, capture=capture)
        self.last_latency = time.perf_counter() - started
        self._chunk = chunk[: self._steps]
        self._index = 0
        self.chunks += 1
        return len(self._chunk)

    def pop(self):
        action = self._chunk[self._index]
        self._index += 1
        return action

    def remaining(self):
        return 0 if self._chunk is None else len(self._chunk) - self._index


def load_action_normalizers(path, device, revision=None):
    """Borrow just the action normalizer/unnormalizer from a checkpoint.

    Builds the processor pipelines without constructing the policy, so the
    backbone is never loaded and this is cheap enough to run before a replay.

    Returns ``(normalize, unnormalize)``.  The forward direction is the step
    training applied to the dataset's actions; the reverse is the *whole*
    postprocessor pipeline, which is exactly what a policy's raw output passes
    through at inference (``_generate_chunk`` calls it the same way).
    """
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    path = _resolve_revision(path, revision)
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    check_normalization_stats(preprocessor, path, "input")
    check_normalization_stats(postprocessor, path, "output")

    found = [s for s in preprocessor.steps if type(s).__name__ == "NormalizerProcessorStep"]
    if not found:
        raise SystemExit(f"{path} has no NormalizerProcessorStep to borrow")
    # No public single-action entry point exists; _normalize_action is the same
    # method the pipeline's __call__ dispatches to for TransitionKey.ACTION.
    normalizer = found[0]

    def normalize(action):
        tensor = torch.as_tensor(np.asarray(action, dtype=np.float32)).unsqueeze(0)
        return normalizer._normalize_action(tensor, inverse=False)

    def unnormalize(tensor):
        out = torch.as_tensor(postprocessor(tensor)).squeeze(0)
        return np.asarray(out.to("cpu").float().numpy(), dtype=np.float32)

    return normalize, unnormalize
