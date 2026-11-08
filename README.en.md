# PDQ Pathfinding

[简体中文](README.md) · English

PDQ (Potential-Difference Queue) finds shortest paths on a grid using a deque. You can move up, down, left, or right, each move costs 1, and obstacles stay put during the search. The repository includes a Python implementation and benchmarks against BFS, heap-based A*, and bucket-queue A*.

## Why a deque is enough

Think about what happens to A*'s priority when you take a step. The distance traveled, `g`, goes up by 1. If you step closer to the goal, the Manhattan distance `h` goes down by 1, so `g + h` stays the same. If you step away, `g + h` goes up by 2.

Divide that change by two and each edge `u → v` gets a weight of `(1 + h(v) - h(u)) / 2`: either 0 or 1. When a better route to a neighbor turns up, put the neighbor at the front of the deque for a 0 edge and at the back for a 1 edge. That's the rule for 0–1 BFS.

This still gives a shortest path. Add up the new weights along a route and the intermediate `h` terms cancel out. The total is half the number of steps plus a constant that depends only on the start and goal. So the route with the lowest new cost also takes the fewest steps.

## What the benchmarks showed

The original PDQ and bucket-queue A* returned the same paths and expanded the same number of nodes. The timing difference mainly comes down to how they manage the queue and its priorities. BFS still came out ahead on some maps with long detours.

Across 240 queries on 15 maps, the median speedup for `pdq_fast` was **1.43×** over the original PDQ, **2.52×** over heap-based A* with ties broken in favor of higher `g`, and **1.63×** over bucket-queue A*. Each ratio compares timings for the same query, including the time it took to reconstruct the full path.

The landmark version borrows from ALT. On the two maze maps, it took an estimated 11 and 12 queries to make up for the time spent building the landmarks, compared with just using `pdq_fast`. Across all 15 maps, with 16 queries per map, `pdq_fast` still took less time overall.

The [original report](results/report.md) and [follow-up report](results_optimized/report.md) are in Chinese; the [per-query timings](results_optimized/queries.csv) are also available. The results are for the Python implementations in this repo. They don't cover diagonal movement, weighted terrain, or obstacles that change during a search.
