import numpy as np
import networkx as nx
from scipy.ndimage import convolve, gaussian_filter1d
from skimage.morphology import skeletonize

DEFAULT_SIGMA = 2

neighbour_kernel = 2 ** np.array([
    [4, 5, 6],
    [3, 0, 7],
    [2, 1, 0]
])
neighbour_kernel[1, 1] = 0

int_reprs = np.zeros((256, 8), dtype=np.uint8)
for i in range(255):
    int_reprs[i] = [int(c) for c in np.binary_repr(i, 8)]
int_reprs *= np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)

neighbour_locs = np.array([
    (0, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1)
])


def im_to_graph(skeletonized: np.ndarray) -> nx.Graph:
    convolved = (
            convolve(skeletonized.astype(np.uint8), neighbour_kernel, mode="constant", cval=0, origin=[0, 0]) * skeletonized
    ).astype(np.uint8)
    ys, xs = convolved.nonzero()  # n length

    location_bits = int_reprs[convolved[ys, xs]]  # n by 8
    diffs = neighbour_locs[location_bits]  # n by 8 by 2
    g = nx.Graph()

    for yx, this_diff in zip(zip(ys, xs), diffs):
        nonself = this_diff[np.abs(this_diff).sum(axis=1) > 0]
        partners = nonself + yx
        for partner in partners:
            g.add_edge(
                yx, tuple(partner),
                weight=np.linalg.norm(partner - yx)
            )

    return g


def partition_tree(g: nx.Graph):
    ends = [coord for coord, deg in g.degree().items() if deg == 1]
    if len(g) < 2:
        raise ValueError("Graph is cyclic")

    root, *leaves = ends

    paths = nx.single_source_shortest_path(g, root)
    visited = set()
    for leaf in leaves:
        path = []
        for node in paths[leaf]:
            path.append(node)
            if node in visited:
                break
        yield path
        visited.update(path)


def partition_forest(g: nx.Graph):
    for nodes in nx.connected_components(g):
        subgraph = g.subgraph(nodes)
        yield from partition_tree(subgraph)


def im_to_area_smooth(binary_img: np.ndarray, sigma: float=DEFAULT_SIGMA) -> float:
    skeletonised = skeletonize(binary_img)
    if skeletonised.sum() == 0:
        return 0
    g = im_to_graph(skeletonised)
    length = 0
    for linestring in partition_forest(g):
        smoothed = gaussian_filter1d(np.asarray(linestring, dtype=float), sigma=sigma, axis=0)
        length += np.linalg.norm(np.diff(smoothed, axis=0), axis=1).sum()
    return length


kernel = np.sqrt(np.array([
    [2, 1, 2],
    [1, 0, 1],
    [2, 1, 2]
])) / 2
origin = (0, 0)


def im_to_area_fast(binary_img: np.ndarray, **kwargs) -> float:
    skeletonised = skeletonize(binary_img)
    return convolve(
        skeletonised.astype(float),
        kernel,
        mode="constant",
        cval=0,
        origin=origin,
    )[skeletonised].sum()

