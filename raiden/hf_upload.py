"""Push an exported LeRobot dataset to the Hugging Face Hub.

Used by ``rd export_lerobot --upload``.  Upload is a separate step from the
export itself: it diffs the local dataset directory against the files already in
the repo and commits only what is missing or has changed, so re-running after
converting a few new episodes uploads only those episodes.

The access token is read from a local ``.env`` file (``HF_TOKEN=hf_...``) so it
never has to appear on the command line or in shell history.  Datasets are
created **public** by default — the Hub gives public datasets far more generous
storage than private ones, which matters at the ~1 TB scale Raiden targets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Environment variable names checked for a token, in order.  ``HF_TOKEN`` is
#: the name huggingface_hub itself uses; the rest are common aliases.
TOKEN_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)

#: Directories inside the dataset root that are never uploaded.
_SKIP_DIRS = {".git", ".cache", "__pycache__"}

#: Roughly how much to put in a single commit.  Many small commits are slower;
#: one huge commit means an interrupted upload has to be re-diffed from scratch.
_MAX_FILES_PER_COMMIT = 64
_MAX_BYTES_PER_COMMIT = 8 * 1024**3


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a dict.

    Deliberately minimal (no ``python-dotenv`` dependency): ``KEY=value`` lines,
    an optional ``export`` prefix, ``#`` comments, and single or double quotes.
    """
    out: Dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:  # trailing comment on an unquoted value
            value = value.split(" #", 1)[0].strip()
        if key:
            out[key] = value
    return out


def default_env_files() -> List[Path]:
    return [Path(".env"), Path.home() / ".config" / "raiden" / ".env"]


def resolve_hf_token(env_file: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Find a Hugging Face token.  Returns ``(token, where_it_came_from)``.

    Search order: the given (or default) ``.env`` file, then the process
    environment, then a token stored by ``huggingface-cli login``.  The ``.env``
    file wins so that a project-local token overrides a stale global login.
    """
    candidates = [Path(env_file)] if env_file else default_env_files()
    for path in candidates:
        if not path.is_file():
            if env_file:
                raise SystemExit(f"--env-file not found: {path}")
            continue
        values = parse_env_file(path)
        for key in TOKEN_KEYS:
            if values.get(key):
                return values[key], f"{path} ({key})"

    for key in TOKEN_KEYS:
        if os.environ.get(key):
            return os.environ[key], f"${key}"

    try:
        from huggingface_hub import get_token  # noqa: PLC0415

        token = get_token()
        if token:
            return token, "huggingface-cli login"
    except Exception:
        pass

    return None, ""


def _token_or_die(env_file: Optional[str], explicit: Optional[str]) -> Tuple[str, str]:
    if explicit:
        return explicit, "--hf-token"
    token, source = resolve_hf_token(env_file)
    if not token:
        searched = ", ".join(str(p) for p in ([Path(env_file)] if env_file else default_env_files()))
        raise SystemExit(
            "no Hugging Face token found.\n"
            f"  looked in: {searched}, ${'/$'.join(TOKEN_KEYS)}, huggingface-cli login\n"
            "  create a write token at https://huggingface.co/settings/tokens, then:\n"
            "      echo 'HF_TOKEN=hf_xxxxxxxx' >> .env\n"
            "  (.env is already in .gitignore)"
        )
    return token, source


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@dataclass
class HFUploadConfig:
    """Where and how to push an exported dataset."""

    #: ``user/name`` or a bare ``name`` (the authenticated user is prepended).
    repo_id: str

    #: Public by default — see the module docstring.
    private: bool = False

    #: Overrides the ``.env`` lookup.  Avoid: it lands in shell history.
    token: Optional[str] = None

    #: Explicit ``.env`` path; ``None`` searches the default locations.
    env_file: Optional[str] = None

    branch: Optional[str] = None
    commit_message: Optional[str] = None

    #: Write a ``README.md`` dataset card if the dataset has none.
    write_card: bool = True

    #: SPDX-ish license id for the card, e.g. ``apache-2.0``.  ``None`` leaves
    #: the license unset on the Hub.
    license: Optional[str] = None

    #: Re-upload every file instead of diffing against the repo first.
    force: bool = False


def _local_files(root: Path) -> List[Tuple[str, Path, int]]:
    """``(path_in_repo, local_path, size)`` for every uploadable file."""
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts[:-1]) or rel.parts[0] in _SKIP_DIRS:
            continue
        out.append((rel.as_posix(), path, path.stat().st_size))
    return out


#: Below this size a file is content-hashed rather than size-compared.  Small
#: metadata rewrites often keep the byte count identical -- ``total_frames``
#: going from 1703 to 1704 in ``meta/info.json`` is the obvious case -- so size
#: alone would leave stale metadata on the Hub.  Hashing every 200 MB video
#: instead would cost minutes per run at dataset scale, and those only ever
#: change by gaining whole new files.
_HASH_BELOW_BYTES = 8 * 1024**2


def _git_blob_sha1(path: Path) -> str:
    """Git's object id for a file: ``sha1("blob <size>\0" + content)``.

    This is what the Hub reports as ``blob_id`` for non-LFS files, so it can be
    compared without downloading anything.
    """
    import hashlib  # noqa: PLC0415

    size = path.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _remote_files(api, repo_id: str, branch: Optional[str]) -> Dict[str, Tuple[int, Optional[str]]]:
    """``path_in_repo -> (size, blob_id)`` for what is already in the repo."""
    from huggingface_hub.utils import (  # noqa: PLC0415
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        entries = api.list_repo_tree(
            repo_id, repo_type="dataset", recursive=True, revision=branch
        )
        return {
            e.path: (e.size, getattr(e, "blob_id", None))
            for e in entries
            if getattr(e, "size", None) is not None
        }
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError):
        return {}


def _needs_upload(rel: str, path: Path, size: int, remote: Dict[str, Tuple[int, Optional[str]]]) -> bool:
    entry = remote.get(rel)
    if entry is None:
        return True
    remote_size, blob_id = entry
    if remote_size != size:
        return True
    # Same size: for small files confirm the content actually matches.  Large
    # files are stored as LFS pointers whose blob_id is the pointer's hash, not
    # the content's, so there is nothing cheap to compare against.
    if size <= _HASH_BELOW_BYTES and blob_id:
        return blob_id != _git_blob_sha1(path)
    return False


def _batched(
    files: Sequence[Tuple[str, Path, int]],
) -> List[List[Tuple[str, Path, int]]]:
    """Split into commit-sized groups, capped by both file count and bytes."""
    batches: List[List[Tuple[str, Path, int]]] = []
    current: List[Tuple[str, Path, int]] = []
    nbytes = 0
    for entry in files:
        if current and (
            len(current) >= _MAX_FILES_PER_COMMIT or nbytes + entry[2] > _MAX_BYTES_PER_COMMIT
        ):
            batches.append(current)
            current, nbytes = [], 0
        current.append(entry)
        nbytes += entry[2]
    if current:
        batches.append(current)
    return batches


_CARD_MARKER = "<!-- generated by rd export_lerobot -->"


def build_dataset_card(root: Path, repo_id: str, license_id: Optional[str] = None) -> str:
    """Render a dataset card from ``meta/info.json``."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        info = {}

    front = ["---"]
    if license_id:
        front.append(f"license: {license_id}")
    front += [
        "task_categories:",
        "- robotics",
        "tags:",
        "- LeRobot",
        "- raiden",
        f"- {info.get('robot_type', 'yam')}",
        "configs:",
        "- config_name: default",
        "  data_files: data/*/*.parquet",
        "---",
        "",
    ]

    features = info.get("features", {})
    rows = [
        f"| `{key}` | {ft.get('dtype')} | {tuple(ft.get('shape', []))} |"
        for key, ft in sorted(features.items())
    ]

    body = [
        _CARD_MARKER,
        f"# {repo_id.split('/')[-1]}",
        "",
        (
            "Bimanual manipulation demonstrations recorded with "
            "[Raiden](https://github.com/TRI-ML/raiden) and exported to the "
            "[LeRobot](https://github.com/huggingface/lerobot) v3.0 format."
        ),
        "",
        "## Dataset",
        "",
        f"- **Codebase version:** {info.get('codebase_version', 'v3.0')}",
        f"- **Robot type:** {info.get('robot_type', 'yam')}",
        f"- **Episodes:** {info.get('total_episodes', '?')}",
        f"- **Frames:** {info.get('total_frames', '?')}",
        f"- **Tasks:** {info.get('total_tasks', '?')}",
        f"- **FPS:** {info.get('fps', '?')}",
        "",
        "## Features",
        "",
        "| Key | dtype | Shape |",
        "|---|---|---|",
        *rows,
        "",
        "Camera intrinsics are stored per frame as `observation.intrinsics.<camera>`",
        "(flattened 3x3), and wrist-camera extrinsics as `observation.extrinsics.<camera>`.",
        "Depth is 12-bit logarithmically quantized in the video stream; a lossless",
        "first frame of each episode (RGB PNG + uint16 depth PNG + calibration JSON)",
        "is archived under `meta/keyframes/` for pose estimation.",
        "",
        "## Usage",
        "",
        "```python",
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        "",
        f'dataset = LeRobotDataset("{repo_id}")',
        "print(dataset[0].keys())",
        "```",
        "",
    ]
    return "\n".join(front + body)


def _maybe_write_card(root: Path, repo_id: str, license_id: Optional[str]) -> None:
    """Write our dataset card, unless the dataset already carries a hand-written one.

    ``repo_id`` must be the *resolved* ``user/name`` — the card embeds it in the
    usage snippet, and a card that changes between runs would be re-uploaded
    every time.
    """
    card = root / "README.md"
    if card.exists():
        # Regenerate only our own card, so hand-written notes survive.
        try:
            if _CARD_MARKER not in card.read_text():
                return
        except OSError:
            return
    text = build_dataset_card(root, repo_id, license_id)
    if not card.exists() or card.read_text() != text:
        card.write_text(text)


def _writable_namespaces(me: Dict) -> Optional[set]:
    """Namespaces this token may write to, or ``None`` if it grants blanket write.

    A classic ``write`` token can push anywhere its user can.  A fine-grained
    token carries an explicit per-entity permission list, so a read-only one can
    be rejected up front instead of 403-ing after gigabytes have been hashed.
    """
    auth = (me.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role")
    if role == "read":
        return set()
    if role != "fineGrained":
        # "write", or a shape we do not recognise: let the Hub be the judge.
        return None

    grained = auth.get("fineGrained") or {}
    writable = set()
    for scope in grained.get("scoped") or []:
        perms = set(scope.get("permissions") or [])
        if perms & {"repo.write", "repo.content.write"}:
            name = (scope.get("entity") or {}).get("name")
            if name:
                writable.add(name)
    if any(p in {"repo.write", "repo.content.write"} for p in grained.get("global") or []):
        return None
    return writable


def resolve_repo_id(api, repo_id: str) -> str:
    """Qualify a bare name with the authenticated user, and sanity-check it."""
    from huggingface_hub.utils import HfHubHTTPError  # noqa: PLC0415

    try:
        me = api.whoami()
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"Hugging Face rejected the token: {exc}\n"
            "  check it is a *write* token from https://huggingface.co/settings/tokens"
        ) from exc

    user = me.get("name", "")
    orgs = {o.get("name") for o in me.get("orgs", []) if o.get("name")}
    namespace = repo_id.split("/", 1)[0] if "/" in repo_id else user
    name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id

    if namespace != user and namespace not in orgs:
        owned = ", ".join(sorted({user} | orgs)) or user
        raise SystemExit(
            f"cannot push to namespace {namespace!r} — this token belongs to {user!r}.\n"
            f"  namespaces you can write to: {owned}\n"
            f"  pass e.g. --hf-repo-id {user}/{name}"
        )

    writable = _writable_namespaces(me)
    if writable is not None and namespace not in writable:
        raise SystemExit(
            f"this token cannot write to {namespace!r} — it looks read-only.\n"
            "  create a token with write access at "
            "https://huggingface.co/settings/tokens\n"
            "  (fine-grained tokens need the 'Write access to contents' permission)"
        )
    return f"{namespace}/{name}"


def find_datasets(output_dir: str, tasks: Optional[Sequence[str]] = None) -> List[Path]:
    """Locate already-exported LeRobot datasets under ``output_dir``.

    Uploading is deliberately decoupled from exporting: once raw recordings have
    been deleted the export step can no longer find them, but the dataset is
    still there and may still need to reach the Hub.  A directory counts as a
    dataset if it holds ``meta/info.json``.
    """
    base = Path(output_dir)
    if not base.is_dir():
        raise SystemExit(f"output directory not found: {base}")

    if tasks:
        found = []
        for name in tasks:
            candidate = Path(name) if Path(name).is_dir() else base / name
            if not (candidate / "meta" / "info.json").is_file():
                available = ", ".join(sorted(d.name for d in _dataset_dirs(base))) or "(none)"
                raise SystemExit(
                    f"no exported dataset for {name!r} (looked in {candidate}).\n"
                    f"  exported datasets under {base}: {available}\n"
                    "  drop --upload-only to export it first"
                )
            found.append(candidate)
        return found

    # The output dir may itself be a single dataset rather than a parent of many.
    if (base / "meta" / "info.json").is_file():
        return [base]

    datasets = _dataset_dirs(base)
    if not datasets:
        raise SystemExit(f"no exported LeRobot datasets found under {base}")
    return datasets


def _dataset_dirs(base: Path) -> List[Path]:
    return sorted(d for d in base.iterdir() if (d / "meta" / "info.json").is_file())


def _hub_api(cfg: HFUploadConfig):
    """Authenticated ``HfApi``, plus a human-readable note on the token source."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install
        raise SystemExit(
            "--upload needs the huggingface_hub package:\n"
            "      uv pip install 'huggingface_hub>=0.30'"
        ) from exc

    token, source = _token_or_die(cfg.env_file, cfg.token)
    return HfApi(token=token), source


def check_upload_credentials(cfg: HFUploadConfig) -> str:
    """Validate the token and namespace before a long export starts.

    Returns the resolved ``user/name``.  Exporting a task can take tens of
    minutes; discovering a missing or read-only token only at the upload step
    would waste all of it.
    """
    api, source = _hub_api(cfg)
    repo_id = resolve_repo_id(api, cfg.repo_id)
    print(f"Hugging Face: token from {source}, will push to {repo_id}")
    return repo_id


def upload_dataset(root: Path, cfg: HFUploadConfig) -> str:
    """Create/update the Hub dataset repo, uploading only changed files.

    Returns the dataset URL.
    """
    from huggingface_hub import CommitOperationAdd  # noqa: PLC0415

    root = Path(root)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"not a LeRobot dataset (no meta/info.json): {root}")

    api, source = _hub_api(cfg)
    repo_id = resolve_repo_id(api, cfg.repo_id)
    visibility = "private" if cfg.private else "public"
    print(f"\nPushing to https://huggingface.co/datasets/{repo_id} ({visibility})")
    print(f"  token from {source}")

    existed = api.repo_exists(repo_id, repo_type="dataset")
    api.create_repo(repo_id, repo_type="dataset", private=cfg.private, exist_ok=True)
    if existed:
        # create_repo ignores `private` for an existing repo, so say so rather
        # than letting the caller believe a private dataset just went public.
        current = api.repo_info(repo_id, repo_type="dataset").private
        if bool(current) != cfg.private:
            print(
                f"  note: repo already exists and is {'private' if current else 'public'}; "
                "visibility unchanged (change it in the repo settings on the Hub)"
            )
    if cfg.branch:
        api.create_branch(
            repo_id, repo_type="dataset", branch=cfg.branch, exist_ok=True
        )

    if cfg.write_card:
        _maybe_write_card(root, repo_id, cfg.license)

    files = _local_files(root)
    if cfg.force:
        pending = files
    else:
        remote = _remote_files(api, repo_id, cfg.branch)
        pending = [f for f in files if _needs_upload(f[0], f[1], f[2], remote)]
        skipped = len(files) - len(pending)
        if skipped:
            print(f"  {skipped} file(s) already on the Hub, unchanged")

    if not pending:
        print(f"✓ Hub dataset already up to date ({len(files)} files)")
        return f"https://huggingface.co/datasets/{repo_id}"

    # Commit payload before metadata: if the upload is interrupted, the repo
    # never advertises episodes whose videos have not landed yet.
    pending.sort(key=lambda f: (f[0].startswith("meta/"), f[0]))

    total_mb = sum(f[2] for f in pending) / 1e6
    batches = _batched(pending)
    print(f"  uploading {len(pending)} file(s), {total_mb:.1f} MB, in {len(batches)} commit(s)")

    base_message = cfg.commit_message or "Upload LeRobot dataset via rd export_lerobot"
    for i, batch in enumerate(batches, 1):
        message = base_message if len(batches) == 1 else f"{base_message} ({i}/{len(batches)})"
        api.create_commit(
            repo_id,
            repo_type="dataset",
            revision=cfg.branch,
            operations=[
                CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path))
                for rel, path, _ in batch
            ],
            commit_message=message,
        )
        if len(batches) > 1:
            print(f"  commit {i}/{len(batches)} done ({len(batch)} files)")

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"✓ pushed {len(pending)} file(s) ({total_mb:.1f} MB) to {url}")
    return url
