# PDQ Pathfinding

[简体中文](README.md) · English

PDQ (Potential-Difference Queue) uses a deque to find shortest paths on a grid. It works when you can move one square up, down, left, or right, every move costs 1, and the map stays fixed during a search. This repository has a Python implementation and benchmarks against BFS, heap-based A*, and bucket-queue A*.

## Why a deque is enough

Let `h` be the Manhattan distance to the goal. On a four-way grid, each move changes `h` by exactly 1. Give an edge `u → v` the weight `(1 + h(v) - h(u)) / 2` and you only have two weights left: 0 and 1. When you find a shorter route to a neighbor, put it at the front of the deque for a 0 edge or at the back for a 1 edge.

Add up those weights along a path and the intermediate `h` values cancel out. You're left with half the step count plus a constant that depends only on the start and goal. For a fixed start and goal, minimizing the new cost still gives the fewest steps. That's why 0–1 BFS works here.

In the benchmark, the original PDQ and the bucket-queue A* baseline returned the same paths and expanded the same number of nodes. The landmark version builds on the existing ALT approach.

## What the benchmarks showed

Across 240 queries on 15 maps, the median paired speedup for `pdq_fast` was **1.43×** over the original PDQ, **2.52×** over heap-based A* that breaks ties in favor of higher `g`, and **1.63×** over bucket-queue A*. The timings include search and full path reconstruction. These numbers describe this set of maps, queries, and Python implementations.

On the two maze maps, the landmark version took about 11 and 12 queries to pay back its setup cost compared with `pdq_fast`. Across all 15 maps at 16 queries each, `pdq_fast` still had the lower total cost once setup was included.

See the [original report](results/report.md), the [optimization report](results_optimized/report.md), and the [per-query data](results_optimized/queries.csv). The reports are in Chinese and include cases where BFS was faster. Diagonal moves, weighted terrain, and dynamic obstacles were outside the scope of these tests.
