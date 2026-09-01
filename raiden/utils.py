"""Shared utilities for Raiden CLI."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import iterfzf


def fzf_select(
    items: List[str],
    prompt: str,
    multi: bool = False,
    header: Optional[str] = None,
) -> List[str]:
    """Pipe *items* to fzf and return the selected entries.

    Uses ``--multi`` when *multi=True* (Tab to toggle items).
    Exits cleanly if the user cancels (Esc / Ctrl-C).
    """
    extra = ["--height=40%", "--layout=reverse", "--border"]
    bind: dict = {}
    color: dict = {}
    hdr = ""

    if multi:
        extra += ["--marker=● ", "--header-first"]
        bind["tab"] = "toggle"
        color["marker"] = "#ffffff"
        hdr = header or "Tab: toggle  |  Enter: confirm  |  Esc: cancel"
    elif header:
        extra.append("--header-first")
        hdr = header

    try:
        result = iterfzf.iterfzf(
            items,
            prompt=prompt,
            multi=multi,
            bind=bind or None,
            color=color or None,
            header=hdr,
            __extra__=extra,
        )
    except KeyboardInterrupt:
        sys.exit(0)
    if result is None:
        sys.exit(0)
    # multi=False returns a str; multi=True returns a list[str]
    return result if isinstance(result, list) else [result]


def select_recording(data_dir: str = "data/raw") -> Optional[Path]:
    """Interactively select a single recording episode using fzf.

    Returns the Path to the selected episode directory, or None if the user
    cancels.  An episode directory is a timestamped subdirectory of a task
    folder that contains a ``cameras/`` subdirectory.
    """
    base = Path(data_dir)
    if not base.exists():
        print(f"No recordings found in {base}")
        sys.exit(1)

    episodes: dict[str, Path] = {}
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "cameras").exists():
                label = f"{task_dir.name} / {ep_dir.name}"
                episodes[label] = ep_dir

    if not episodes:
        print(f"No recordings found in {base}")
        sys.exit(1)

    selected = fzf_select(list(episodes), prompt="Select recording> ")
    if not selected:
        return None
    return episodes[selected[0]]


def select_processed_recording(data_dir: str = "data/processed") -> Optional[Path]:
    """Interactively select a single converted episode using fzf.

    Returns the Path to the selected episode directory, or None if the user
    cancels.  An episode directory contains a ``metadata.json`` file.
    """
    base = Path(data_dir)
    if not base.exists():
        print(f"No processed recordings found in {base}")
        sys.exit(1)

    episodes: dict[str, Path] = {}
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "metadata.json").exists():
                label = f"{task_dir.name} / {ep_dir.name}"
                episodes[label] = ep_dir

    if not episodes:
        print(f"No processed recordings found in {base}")
        sys.exit(1)

    selected = fzf_select(list(episodes), prompt="Select recording> ")
    if not selected:
        return None
    return episodes[selected[0]]


def demonstration_status(rec_dir: Path) -> str:
    """Return the verdict for a raw recording: success / failure / pending / unknown.

    ``metadata.json`` is the authority because it travels with the data.  The
    demonstrations DB is only a fallback, for recordings made before the status
    field was written into metadata; it is keyed by the exact path string that
    was current at record time, so both the given path and its
    relative-to-cwd form are tried.

    Returns ``"unknown"`` when neither source has a verdict, which callers
    should treat as "keep" so that pre-DB recordings are not silently dropped.
    """
    meta_file = rec_dir / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                status = json.load(f).get("status")
            if status:
                return str(status)
        except (OSError, json.JSONDecodeError):
            pass

    candidates = [str(rec_dir)]
    try:
        candidates.append(str(rec_dir.resolve().relative_to(Path.cwd())))
    except ValueError:
        pass

    try:
        from raiden.db.database import get_db

        db = get_db()
        for path_str in candidates:
            demo = db.get_demonstration_by_raw_path(path_str)
            if demo is not None:
                return str(demo.get("status") or "pending")
    except Exception:
        pass

    return "unknown"
