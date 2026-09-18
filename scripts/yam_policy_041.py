"""The AIDA simulator's PI adapter, behind the interface lerobot_client.py uses.

``yam_lerobot_policy.py`` drives a checkpoint through the *installed* LeRobot
0.6.1.  This module drives the same checkpoint through the vendored LeRobot
0.4.1 subset in the AIDA repo -- ``src/sim/_vendor/lerobot_pi/`` -- which is what
``scripts/sim/run_lerobot_policy_yam.py`` runs in Isaac Lab.  Same weights, same
frame packing, same action extraction, so a robot run and a simulator rollout
differ in the robot and not in the runtime.

Why two runtimes exist at all: Isaac pins Python 3.11 and LeRobot 0.6 requires
3.12, so the simulator cannot use the installed package.  Both raiden's venv and
AIDA's venv are 3.11, so the real robot can go the other way and meet the
simulator on 0.4.1.

Requirements, on top of raiden's venv:

* the AIDA checkout, for the vendored runtime (``--aida-root``);
* ``torch`` (already there);
* the patched transformers branch.  Vendored ``modeling_pi05.py`` raises
  ``An incorrect transformer version is used`` on stock transformers, because
  PI0/PI0.5 need LeRobot's OpenPI attention replacement::

      .venv/bin/pip install "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

RTC is not available here: ``lerobot.policies.rtc`` arrived after 0.4.1, so
``--rtc`` needs the 0.6.1 runtime.
"""

import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_AIDA_ROOT = "/home/zyang1/git/AIDA"
ACTION_DIM = 7  # the vendored adapter's own convention: 6 joints + gripper


class YamPi041Policy:
    """A vendored-0.4.1 checkpoint, answering the same calls as YamLeRobotPolicy.

    ``images`` is keyed by the model's image slots, as on the 0.6.1 side; the
    slots are passed through to the AIDA adapter as an identity map so no camera
    naming is invented here.
    """

    def __init__(self, path, *, task, action_dim, device="cuda", slots=(),
                 dataset_meta=None, n_action_steps=None, chunk_size=None,
                 family="pi05", cache_dir=None, aida_root=DEFAULT_AIDA_ROOT, verbose=True):
        if int(action_dim) != ACTION_DIM:
            raise SystemExit(
                f"the 0.4.1 runtime speaks a {ACTION_DIM}-D YAM command; {action_dim}-D was "
                "asked for. Use --arm left or --arm right, or run --runtime 0.6.1."
            )
        src = Path(aida_root).expanduser() / "src"
        if not (src / "sim" / "policies" / "yam_lerobot.py").exists():
            raise SystemExit(
                f"no vendored PI runtime under {src} — point --aida-root at the AIDA checkout"
            )
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from sim.policies.yam_lerobot import YamPiPolicy
        except ImportError as exc:  # pragma: no cover - depends on the venv
            raise SystemExit(f"cannot import the vendored runtime from {src}: {exc}")

        self.task = task
        self.action_dim = ACTION_DIM
        self.device = device
        self.path = path
        stats = None
        if dataset_meta:
            with open(Path(dataset_meta) / "meta" / "stats.json") as f:
                stats = json.load(f)
        self._policy = YamPiPolicy(
            model_id=path,
            family=family,
            device=device,
            cache_dir=cache_dir,
            local_files_only=False,
            task=task,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            dataset_stats=stats,
            image_slots={slot: slot for slot in slots},
        )
        self.normalization = self._policy.normalization
        if verbose:
            print(f"  runtime=vendored 0.4.1 ({src})  family={family}"
                  f"  n_action_steps={self.n_action_steps}  chunk_size={self.chunk_size}"
                  f"  normalization={self.normalization}")

    # ---- the interface lerobot_client.py drives ------------------------------

    @property
    def n_action_steps(self):
        return self._policy.n_action_steps

    @property
    def chunk_size(self):
        return self._policy.chunk_size

    @property
    def type(self):
        return f"{getattr(self._policy.model.config, 'type', 'pi')} (vendored 0.4.1)"

    @property
    def model(self):
        return self._policy.model

    def reset(self):
        self._policy.reset()

    def act(self, state, images, capture=None):
        return self._policy.act(self._observation(state, images), capture=capture)

    def chunk(self, state, images, capture=None):
        return self._policy.chunk(self._observation(state, images), capture=capture)

    def prepare(self, state, images):
        return self._policy.prepare(self._observation(state, images))

    @staticmethod
    def _observation(state, images):
        """The adapter's observation: the state, plus one entry per camera slot."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"state must have shape ({ACTION_DIM},), got {state.shape}.")
        return {"state": state, **images}
