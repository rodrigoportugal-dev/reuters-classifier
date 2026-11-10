"""Download the Reuters-21578 corpus (UCI KDD Archive) into data/reuters21578/.

    python scripts/download_data.py
"""

import io
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz"
DEST = Path(__file__).resolve().parent.parent / "data" / "reuters21578"


def download(dest=DEST):
    dest = Path(dest)
    if len(list(dest.glob("*.sgm"))) >= 22:
        print(f"Dataset already in {dest}")
        return dest
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        data = resp.read()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".sgm")]
        for m in members:
            m.name = Path(m.name).name  # flatten, never write outside dest
        tar.extractall(dest, members=members, filter="data")
    print(f"Extracted {len(members)} files to {dest}")
    return dest


if __name__ == "__main__":
    download(*sys.argv[1:2])
