import networkx
import numpy as np

def _ensure_nonempty(labels, Y, k, fallback_axis=None, rng=None):
    """Ensure no empty clusters; for k=2 use median cut on fallback_axis if needed,
    else move the farthest points into empty clusters."""
    n = Y.shape[0]
    uniq = np.unique(labels)
    if len(uniq) == k and all((labels == j).any() for j in range(k)):
        return labels  # already OK

    # k=2: deterministic fallback via Fiedler vector median cut
    if k == 2 and fallback_axis is not None:
        med = np.median(fallback_axis)
        left = fallback_axis <= med
        right = ~left
        # If all equal to median, random tie-break:
        if left.sum() == 0 or right.sum() == 0:
            if rng is None:
                rng = np.random.default_rng(0)
            perm = rng.permutation(n)
            half = n // 2
            left = np.zeros(n, dtype=bool); left[perm[:half]] = True
            right = ~left
        labels = np.where(left, 0, 1)
        return labels

    # General k: seed any empty cluster with farthest points from their current centers
    # (cheap heuristic repair)
    k_present = [j for j in range(k) if (labels == j).any()]
    k_missing = [j for j in range(k) if not (labels == j).any()]
    if len(k_missing) == 0:
        return labels

    # Compute current centers for present clusters
    centers = []
    for j in range(k):
        mask = (labels == j)
        if mask.any():
            centers.append(Y[mask].mean(axis=0))
        else:
            centers.append(None)
    centers = np.array([c if c is not None else np.zeros(Y.shape[1]) for c in centers])

    # Select candidates from the largest present cluster
    sizes = [(j, (labels == j).sum()) for j in k_present]
    largest = max(sizes, key=lambda x: x[1])[0]
    candidates = np.where(labels == largest)[0]
    if rng is None:
        rng = np.random.default_rng(0)

    # Greedy: move the farthest candidates into each missing cluster
    for j in k_missing:
        d2 = ((Y[candidates] - centers[largest]) ** 2).sum(axis=1)
        idx = candidates[d2.argmax()]
        labels[idx] = j
        # update center of largest (roughly)
        candidates = candidates[candidates != idx]
        if candidates.size == 0:
            break
    return labels

def spectral_clustering(
    graph: networkx.MultiGraph,
    num_cluster: int = 2,
    normalized: bool = False,
    random_state: int = 0
):
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return tuple(() for _ in range(num_cluster))
    if num_cluster <= 1 or n == 1:
        return (tuple(nodes),) + tuple(() for _ in range(max(0, num_cluster - 1)))

    k = min(num_cluster, n)

    # Connected-components short-circuit (still useful)
    comps = [list(c) for c in networkx.connected_components(networkx.Graph(graph))]
    if len(comps) >= k:
        comps = sorted(comps, key=len, reverse=True)
        clusters = comps[:k-1] + [[u for comp in comps[k-1:] for u in comp]]
        out = tuple(tuple(c) for c in clusters)
        if num_cluster > k:
            out = out + tuple(() for _ in range(num_cluster - k))
        return out

    # Laplacian
    L = (networkx.normalized_laplacian_matrix if normalized
         else networkx.laplacian_matrix)(graph, weight='weight').astype(float)

    m = min(k, max(1, n - 1))
    try:
        from scipy.sparse.linalg import eigsh
        vals, vecs = eigsh(L, k=m, which='SM')
    except Exception:
        vals_all, vecs_all = np.linalg.eigh(L.toarray())
        vals, vecs = vals_all[:m], vecs_all[:, :m]

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # Embedding (use eigenvectors 2..k)
    if vecs.shape[1] == 1:
        embed = vecs
        fiedler = vecs[:, 0]
    else:
        embed = vecs[:, 1:min(k, vecs.shape[1])]
        fiedler = vecs[:, 1]

    # Row-normalize (Ng–Jordan–Weiss)
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y = embed / norms

    # Tiny jitter to break exact duplicates
    rng = np.random.default_rng(random_state)
    if Y.size > 0:
        Y = Y + 1e-10 * rng.normal(size=Y.shape)

    # K-means with retries
    labels = None
    tried = 0
    try:
        from sklearn.cluster import KMeans
        while tried < 4:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state + tried)
            labels = km.fit_predict(Y)
            if len(np.unique(labels)) == k and all((labels == j).any() for j in range(k)):
                break
            tried += 1
    except Exception:
        tried = 4  # go to fallback

    if labels is None or len(np.unique(labels)) < k or any(not (labels == j).any() for j in range(k)):
        labels = _ensure_nonempty(labels if labels is not None else np.zeros(n, dtype=int),
                                  Y, k, fallback_axis=fiedler, rng=rng)

    # Collect nodes
    buckets = [[] for _ in range(k)]
    for idx, lbl in enumerate(labels):
        buckets[lbl].append(nodes[idx])

    if num_cluster > k:
        buckets += [[] for _ in range(num_cluster - k)]

    return tuple(tuple(c) for c in buckets)



def spectral_balance(graph: networkx.MultiGraph, normalized: bool = False) -> float:
    """
    Return \\varepsilon(graph), the balance of the spectral cut of the graph
    :param graph: a networkx.MultiGraph object
    :param normalized: Bool indicating whether to use normalized or un-normalized Laplacian, default is False
    :return: min(|S|,|\\bar{S}|)/n(G) where (S,\\bar{S}) is the spectral cut
    """
    # FIX: use keyword args so normalized isn't misinterpreted as num_cluster
    num_nodes_cut = len(spectral_clustering(graph, num_cluster=2, normalized=normalized)[0])
    num_nodes_graph = graph.number_of_nodes()
    return float(min(num_nodes_cut, num_nodes_graph - num_nodes_cut) / num_nodes_graph)


def induced_subgraph(graph: networkx.MultiGraph, node_list: list, preserve_degrees: bool = False) -> networkx.MultiGraph:
    induced = networkx.MultiGraph()
    induced.add_nodes_from(node_list)
    edges = [(i, j, key) for (i, j, key) in graph.edges if i in node_list and j in node_list]
    induced.add_edges_from(edges)
    if preserve_degrees:
        for v in node_list:
            for _ in range(len(graph[v]) - len(induced[v])):
                induced.add_edge(v, v)
    return induced


def hierarchical_spectral_clustering_order(
        graph: networkx.MultiGraph,
        is_root: bool = False,
        normalized: bool = False
):
    """
    Return the nested tuple representation of the contraction tree obtained by hierarchical spectral clustering
    :param graph: a networkx.MultiGraph object
    :param is_root: Bool indicating whether the clustering is at the root node of the contraction tree
    :param normalized: Bool indicating whether to use normalized or un-normalized Laplacian, default is False
    :return: a nested tuple representation of the contraction tree. For example, if base graph had nodes 1,2,3,4 and the
    contraction order was to contract 1&2, 3&4 and then merge these nodes, then this function outputs ((1,2),(3,4))
    """
    # base cases
    n = graph.number_of_nodes()
    if n == 0:
        return tuple()  # nothing below this branch
    if n == 1:
        return list(graph)[0]

    if is_root:
        clusters = spectral_clustering(graph, num_cluster=3, normalized=normalized)
        sets = [list(c) for c in clusters if len(c) > 0]  # drop empties
        if len(sets) < 3:
            # degrade gracefully: if <3 non-empty, just proceed with however many we have
            while len(sets) < 3:
                sets.append([])
        set_A, set_B, set_C = sets[0], sets[1], sets[2]
        graph_A = induced_subgraph(graph, set_A, preserve_degrees=normalized)
        graph_B = induced_subgraph(graph, set_B, preserve_degrees=normalized)
        graph_C = induced_subgraph(graph, set_C, preserve_degrees=normalized)
        return (
            (hierarchical_spectral_clustering_order(graph_A, normalized=normalized),
             hierarchical_spectral_clustering_order(graph_B, normalized=normalized)),
            hierarchical_spectral_clustering_order(graph_C, normalized=normalized)
        )

    # non-root: 2-way split
    clusters = spectral_clustering(graph, num_cluster=2, normalized=normalized)
    sets = [list(c) for c in clusters if len(c) > 0]  # drop empties
    if len(sets) == 1:
        # nothing to split—return the order of the sole subtree
        return hierarchical_spectral_clustering_order(
            induced_subgraph(graph, sets[0], preserve_degrees=normalized),
            normalized=normalized
        )
    set_A, set_B = sets[0], sets[1]
    graph_A = induced_subgraph(graph, set_A, preserve_degrees=normalized)
    graph_B = induced_subgraph(graph, set_B, preserve_degrees=normalized)
    return (
        hierarchical_spectral_clustering_order(graph_A, normalized=normalized),
        hierarchical_spectral_clustering_order(graph_B, normalized=normalized)
    )
