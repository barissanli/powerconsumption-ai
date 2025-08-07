#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo of the major algorithm families in Python.
Author: ChatGPT 2025
"""

from __future__ import annotations
import heapq
import math
from functools import lru_cache
from typing import List, Tuple, Dict


# ------------------------------------------------------------------
# 1. Sorting – Quick‑sort (in‑place, recursive)
# ------------------------------------------------------------------
def quick_sort(arr: List[int], low: int = 0, high: int | None = None) -> None:
    """
    Sorts the list `arr` in place using the classic quick‑sort algorithm.
    Complexity: O(n log n) average, O(n²) worst case (rare with random pivot).
    """
    if high is None:
        high = len(arr) - 1

    def partition(lo: int, hi: int) -> int:
        pivot = arr[hi]          # take last element as pivot
        i = lo - 1                # place for smaller elements
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        return i + 1

    if low < high:
        p = partition(low, high)
        quick_sort(arr, low, p - 1)
        quick_sort(arr, p + 1, high)


# ------------------------------------------------------------------
# 2. Searching – Binary search on a sorted array
# ------------------------------------------------------------------
def binary_search(sorted_arr: List[int], target: int) -> int:
    """
    Returns the index of `target` in `sorted_arr`, or -1 if not found.
    Complexity: O(log n).
    """
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return mid
        elif sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# ------------------------------------------------------------------
# 3. Graph – Dijkstra’s shortest‑path algorithm (using a priority queue)
# ------------------------------------------------------------------
def dijkstra(adj: Dict[int, List[Tuple[int, float]]], start: int) -> Tuple[Dict[int, float], Dict[int, int | None]]:
    """
    `adj` is an adjacency list: node -> [(neighbor, weight), ...]
    Returns two dicts:
        distances[node] = shortest distance from start to node
        predecessors[node] = previous node on that shortest path (None for start)
    Complexity: O((V+E) log V).
    """
    dist = {node: math.inf for node in adj}
    prev = {node: None for node in adj}
    dist[start] = 0.0
    heap = [(0.0, start)]  # (distance, vertex)

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue          # stale entry

        for v, w in adj[u]:
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    return dist, prev


# ------------------------------------------------------------------
# 4. Dynamic programming – 0/1 Knapsack
# ------------------------------------------------------------------
def knapsack(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    Returns the maximum total value that fits in the given capacity and
    a list of item indices chosen to achieve that value.
    Complexity: O(n * W).
    """
    n = len(values)
    # dp[w] – best value for weight w
    dp = [0] * (capacity + 1)
    keep = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        wt, val = weights[i], values[i]
        for w in range(capacity, wt - 1, -1):
            if dp[w - wt] + val > dp[w]:
                dp[w] = dp[w - wt] + val
                keep[i][w] = True

    # backtrack to find chosen items
    res_items = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            res_items.append(i)
            w -= weights[i]

    return dp[capacity], list(reversed(res_items))


# ------------------------------------------------------------------
# 5. Recursion & divide‑conquer – Fibonacci (naïve, memoised, matrix)
# ------------------------------------------------------------------
def fib_naive(n: int) -> int:
    """Naïve recursive Fibonacci – exponential time."""
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


@lru_cache(maxsize=None)
def fib_memoised(n: int) -> int:
    """Memoised recursion – linear time, constant space after cache."""
    if n <= 1:
        return n
    return fib_memoised(n - 1) + fib_memoised(n - 2)


def fib_matrix(n: int) -> int:
    """
    Fast doubling method using matrix exponentiation.
    Complexity: O(log n).
    """
    def mul(a, b):
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0],
             a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0],
             a[1][0] * b[0][1] + a[1][1] * b[1][1]]
        ]

    def power(mat, exp):
        result = [[1, 0], [0, 1]]  # identity
        while exp:
            if exp & 1:
                result = mul(result, mat)
            mat = mul(mat, mat)
            exp >>= 1
        return result

    base = [[1, 1], [1, 0]]
    res = power(base, n)
    return res[0][1]


# ------------------------------------------------------------------
# Demo driver – shows each algorithm in action
# ------------------------------------------------------------------
def main() -> None:
    print("=== Sorting: Quick‑sort ===")
    arr = [9, 3, 5, 2, 6, 8, 7, 1, 4]
    print("Original:", arr)
    quick_sort(arr)
    print("Sorted:  ", arr, "\n")

    print("=== Searching: Binary search ===")
    target = 5
    idx = binary_search(arr, target)
    print(f"Target {target} found at index {idx}\n")

    print("=== Graph: Dijkstra's algorithm ===")
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }
    distances, prevs = dijkstra(graph, start=0)
    print("Distances:", distances)
    print("Predecessors:", prevs, "\n")

    print("=== Dynamic programming: Knapsack ===")
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    best_val, items = knapsack(values, weights, capacity)
    print(f"Max value: {best_val}")
    print("Items chosen:", items)  # indices of selected items
    print()

    print("=== Recursion & DP: Fibonacci ===")
    n = 10
    print(f"fib_naive({n}) =", fib_naive(n))
    print(f"fib_memoised({n}) =", fib_memoised(n))
    print(f"fib_matrix({n}) =", fib_matrix(n))


if __name__ == "__main__":
    main()

