"""Shared graph topology for ST-GCN.

The 27 nodes match ``KEYPOINT_INDICES`` in ``dataset.py``: body (0-6), left
hand (7-16), right hand (17-26). Edges connect each finger to its proximal
joint and the wrist to the corresponding shoulder.
"""

GRAPH_ARGS = {
    "num_nodes": 27,
    "center": 0,
    "inward_edges": [
        [2, 0], [1, 0], [0, 3], [0, 4], [3, 5],
        [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
        [9, 10], [7, 11], [11, 12], [7, 13], [13, 14],
        [7, 15], [15, 16], [17, 18], [17, 19], [19, 20],
        [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26],
    ],
}

N_FEATURES = 256  # ST-GCN encoder output dimension; matches the FC head input.
