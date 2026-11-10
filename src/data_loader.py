"""Reader for the Reuters-21578 corpus (.sgm files).

* Train / test split taken from the LEWISSPLIT attribute ("TRAIN" / "TEST").
  Documents marked "NOT-USED" are skipped.
* Only the FIRST topic of each document is used as its label, as required by
  the project specification (<D>money-fx</D><D>dlr</D> -> "money-fx").
* Documents without topics or without text are skipped.
"""

import glob
import os

from bs4 import BeautifulSoup


def load_reuters_dataset(data_dir):
    """Return (train_texts, train_labels, test_texts, test_labels)."""
    sgm_files = sorted(glob.glob(os.path.join(data_dir, "*.sgm")))
    if not sgm_files:
        raise FileNotFoundError(
            f"No .sgm files found in '{data_dir}'. Run: python scripts/download_data.py"
        )

    split = {"TRAIN": ([], []), "TEST": ([], [])}
    for path in sgm_files:
        # The corpus contains some non-UTF-8 bytes; latin-1 reads every file.
        with open(path, encoding="latin-1") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for doc in soup.find_all("reuters"):
            part = doc.get("lewissplit", "").upper()
            if part not in split:
                continue
            topics = doc.find("topics")
            first = topics.find("d") if topics else None
            text_node = doc.find("text")
            if first is None or text_node is None:
                continue

            title = text_node.find("title")
            body = text_node.find("body")
            text = " ".join(n.get_text(" ", strip=True) for n in (title, body) if n)
            if not text:
                continue

            texts, labels = split[part]
            texts.append(text)
            labels.append(first.get_text(strip=True))

    return (*split["TRAIN"], *split["TEST"])
