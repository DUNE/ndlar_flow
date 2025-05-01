import numpy as np
from itertools import combinations, chain
from collections import defaultdict

def all_array_sums(arrays):
    """
    Sums all combinations of arrays in the input list list
    """

    result = []
    n = len(arrays)
    
    for r in range(1, n + 1):
        for combo in combinations(arrays, r):
            summed = np.sum(combo, axis=0)
            result.append(summed)
    
    return result

# Fuction to cluster together light hits based on timestamps 
def cluster_neighboring_timestamps(arr, threshold=3):
    """
    Cluster timestamps that are close within a certain threshold

    Parameters:
    ----------
    arr : array-like
        The data values to bin.
    threshold : float
        Threshold.

    Returns:
    -------
    clustered_sorted_indices : np.ndarray
        2D array, first dimension is the cluster index, and second
        dimension contains the indices of the timestamps in arr
    """
    arr = np.asarray(arr)
    sorted_indices = np.argsort(arr)
    sorted_vals = arr[sorted_indices]

    # Find cluster boundaries
    diff = np.diff(sorted_vals)
    boundaries = np.where(diff > threshold)[0] + 1

    # Split sorted indices at boundaries
    clustered_sorted_indices = np.split(sorted_indices, boundaries)

    return clustered_sorted_indices


# functions to merge clusters based on conditions specified in should_merge
def should_merge(timing1, timing2, tol):
    if np.abs(timing1 - timing2) < tol:
        return True
    return False
    
def merge_clusters(clusters, timings, tol=50):
    n = len(clusters)
    parent = list(range(n))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(timings[i], timings[j], tol):
                union(i, j)

    # Group points by root parent
    merged = defaultdict(list)
    for i in range(n):
        root = find(i)
        merged[root].append(clusters[i])

    return [np.vstack(group) for group in merged.values()]
