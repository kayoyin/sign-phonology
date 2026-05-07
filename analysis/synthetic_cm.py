"""Cosine-similarity confusion matrices over HCS phonological parameters.

Given the JSON of model features extracted on the Handshapes-in-Context Stimuli
(HCS) data, this script groups by each phonological parameter (handshape,
location, movement, orientation) and computes the pairwise cosine similarity
between the mean feature vectors of each parameter value. The resulting CSVs
feed into the paper's Table 3 (correlations) and Figure 4 (dendrograms).

HCS filenames look like ``<HS>_<Location>_<Movement>_<Orientation><tag>.mp4``.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PARAMETERS = ("handshape", "location", "movement", "orientation")


def cosine(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def parse_filename(filename):
    """HCS basename -> (handshape, location, movement, orientation)."""
    base = os.path.basename(filename).split(".")[0]
    parts = base.split("_")
    handshape = "_".join(parts[:-3])
    location, movement, orientation = parts[-3], parts[-2], parts[-1]
    orientation = re.sub(r"[^a-zA-Z]", "", orientation)
    return handshape, location, movement, orientation


def load_features(features_json):
    with open(features_json) as fh:
        records = json.load(fh)
    df = pd.DataFrame(records)
    df = df.apply(lambda col: col.map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x))
    parsed = df["filename"].apply(parse_filename).tolist()
    df["handshape"] = [p[0] for p in parsed]
    df["location"] = [p[1] for p in parsed]
    df["movement"] = [p[2] for p in parsed]
    df["orientation"] = [p[3] for p in parsed]
    return df


def parameter_confusion(df, parameter):
    """Pairwise cosine similarity between mean feature vectors of each parameter value."""
    values = sorted(df[parameter].unique())
    means = {
        v: np.mean(np.array(df.loc[df[parameter] == v, "features"].tolist()), axis=0)
        for v in values
    }
    matrix = pd.DataFrame(index=values, columns=values, dtype=float)
    for a in values:
        for b in values:
            matrix.loc[a, b] = 1.0 if a == b else cosine(means[a], means[b])
    return matrix


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True,
                   help="Feature JSON produced by extract_features.py on HCS clips.")
    p.add_argument("--output_dir", required=True,
                   help="Directory where one CSV per phonological parameter is written.")
    p.add_argument("--model_tag", required=True,
                   help="Short tag (e.g. 'i3d_asl') used in the output filenames.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_features(args.features)
    for param in PARAMETERS:
        cm = parameter_confusion(df, param)
        out = os.path.join(args.output_dir, f"{args.model_tag}_cossim_{param}.csv")
        cm.to_csv(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
