"""Install the upstream pen clip catalog without checkpoints or other assets.

Run: python -m dex_supermimic.tasks.pen_spin.tooling.download [--destination DIR]
The downloaded archive is temporary; clip hashes are verified before success.
"""

import argparse
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from .clip_store import discover_clip_groups

URL = "https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.10.22/wuji-mjlab-v2026.10.22-assets.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data/rawdata/clips")
    args = parser.parse_args()
    root = args.destination.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        groups = discover_clip_groups(root, verify_hashes=True)
        print(f"Existing verified catalog: {sum(len(g.clips) for g in groups)} clips at {root}")
        return
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "assets.zip"
        print("Downloading upstream release assets (~247 MB)...", flush=True)
        urlretrieve(URL, archive)
        with ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                marker = "/pen_spin/data/rawdata/clips/"
                if marker not in entry.filename or entry.is_dir():
                    continue
                relative = Path(entry.filename.split(marker, 1)[1])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(entry))
    groups = discover_clip_groups(root, verify_hashes=True)
    print(f"Installed {sum(len(g.clips) for g in groups)} verified clips at {root}")


if __name__ == "__main__":
    main()
