# Analysis

These scripts consume the JSON feature files produced by
[`i3d/extract_features.py`](../i3d/extract_features.py) and
[`stgcn/extract_features.py`](../stgcn/extract_features.py), and produce the
tables and intermediate CSVs used in the paper.

## Files

- **`minimal_pair_ttest.py`** — for each minimal pair (s1, s2), build two
  cosine-similarity distributions across signer pairs (intra-sign and
  inter-sign) and run an independent-samples t-test. Also samples random
  non-minimal gloss pairs as a control. Produces Tables 1 and 2.
- **`handshape_distance.py`** — articulatory reference metric (HD): mean
  absolute joint-angle difference between handshapes, computed at 15 hand
  joints from MediaPipe landmarks. Sweeps every (location, movement,
  orientation) combination in the HCS data. Used as the geometric reference
  in Table 3 and as the geometric dendrogram in Figure 4.
- **`synthetic_cm.py`** — for each phonological parameter (handshape,
  location, movement, orientation) in the HCS data, compute the cosine
  similarity between the mean feature vectors of every parameter value.
  Used to compute Pearson correlations in Table 3 and the model dendrograms
  in Figure 4.

## Notes on conventions

The t-statistic in `ttest_results.csv` is the *signed* output of
`scipy.stats.ttest_ind(D_diff, D_same)`. A phonologically sensitive model
satisfies `D_same > D_diff`, so its t-statistic is **negative**. The
notebook that produces the published tables takes absolute values and
counts "wins" by lower-t comparisons.
