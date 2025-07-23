import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
try:
    from hdbscan import HDBSCAN  # type: ignore
except Exception:  # pragma: no cover - optional
    HDBSCAN = None  # type: ignore

from train.intention_embedding import IntentionEncoder


def _load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _cluster_intentions(
    intentions: List[str],
    method: str,
    clusters: int | None,
    min_samples: int,
) -> Dict[str, str]:
    unique = sorted(set(intentions))
    encoder = IntentionEncoder()
    encoder.fit(unique)
    vecs = encoder.encode_batch(unique).cpu().numpy()

    if method == "kmeans":
        if not clusters:
            clusters = max(1, len(unique) // 2)
        model = KMeans(n_clusters=clusters, random_state=0)
        labels = model.fit_predict(vecs)
    elif method == "dbscan":
        model = DBSCAN(min_samples=min_samples)
        labels = model.fit_predict(vecs)
    elif method == "hdbscan":
        if HDBSCAN is None:
            raise RuntimeError("hdbscan package not available")
        model = HDBSCAN(min_cluster_size=clusters or 2, min_samples=min_samples)
        labels = model.fit_predict(vecs)
    else:
        raise ValueError(f"Unknown method: {method}")

    freq = Counter(intentions)
    mapping: Dict[str, str] = {}
    for lab in set(labels):
        members = [u for u, l in zip(unique, labels) if l == lab]
        if lab == -1 or len(members) == 1:
            for m in members:
                mapping[m] = m
            continue
        canonical = max(members, key=lambda m: freq[m])
        for m in members:
            mapping[m] = canonical
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster and merge intentions in a dataset")
    parser.add_argument("--input", required=True, help="Input NPZ dataset")
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--method", choices=["kmeans", "dbscan", "hdbscan"], default="kmeans")
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters or min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=2, help="min_samples for DBSCAN/HDBSCAN")
    parser.add_argument("--mapping", type=str, default=None, help="Optional JSON mapping output")
    args = parser.parse_args()

    data = _load_dataset(args.input)
    intentions = [str(i) for i in data.get("intentions", [])]
    if not intentions:
        raise KeyError("Dataset does not contain 'intentions'")

    mapping = _cluster_intentions(intentions, args.method, args.clusters, args.min_samples)
    new_intentions = [mapping[i] for i in intentions]
    data["intentions"] = np.array(new_intentions, dtype=object)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **data)

    if args.mapping:
        os.makedirs(os.path.dirname(args.mapping), exist_ok=True)
        with open(args.mapping, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
