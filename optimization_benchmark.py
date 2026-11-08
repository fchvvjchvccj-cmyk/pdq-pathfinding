"""在原有问题上交错复测优化版本，分别记录在线查询和静态预处理成本。"""

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import benchmark as base
from optimized import PreparedPDQ, pdq_fast, prepared_query

ROOT = Path(__file__).resolve().parent
METHODS = {
    'bfs': base.bfs,
    'astar_heap_high_g': base.ALGORITHMS['astar_heap_high_g'],
    'astar_bucket': base.ALGORITHMS['astar_bucket'],
    'pdq_original': base.ALGORITHMS['pdq'],
    'pdq_fast': pdq_fast,
    'pdq_workspace': lambda m, s, t: prepared_query(m, s, t, False),
    'pdq_landmark': prepared_query,
}


def check_optimizations():
    original = base.ALGORITHMS.copy()
    base.ALGORITHMS.update({k: v for k, v in METHODS.items() if k.startswith('pdq_')})
    try:
        count = base.correctness()
    finally:
        base.ALGORITHMS = original
    rng = random.Random(90210)
    edge_checks = 0
    for index in range(12):
        grid = base.Grid(f'heuristic-{index}', [''.join('@' if rng.random() < .27 else '.' for _ in range(19)) for _ in range(15)])
        prepared = PreparedPDQ(grid)
        for goal in rng.sample(grid.free, 3):
            if prepared.components[goal] != prepared.landmark_component:
                continue
            targets = [d[goal] for d in prepared.landmarks]
            values = {}
            for u in grid.free:
                if prepared.components[u] != prepared.landmark_component:
                    continue
                h = abs(u % grid.pitch - goal % grid.pitch) + abs(u // grid.pitch - goal // grid.pitch)
                values[u] = max([h] + [abs(d[u] - t) for d, t in zip(prepared.landmarks, targets)])
            assert values[goal] == 0
            for u, hu in values.items():
                for offset in (-1, 1, -grid.pitch, grid.pitch):
                    if u + offset in values:
                        assert abs(hu - values[u + offset]) == 1
                        edge_checks += 1
    grid = base.Grid('stamp-overflow', ['...@.', '.@.@.', '.@...', '.....'])
    prepared = PreparedPDQ(grid)
    start, goal = grid.at(0, 0), grid.at(4, 0)
    expected = base.bfs(grid, start, goal).distance
    prepared.stamp = 2**32 - 1
    held = prepared.query(start, goal)
    base.validate(grid, start, goal, held, expected)
    assert prepared.stamp == 1
    for _ in range(4):
        base.validate(grid, goal, start, prepared.query(goal, start, False), expected)
        base.validate(grid, start, goal, prepared.query(start, goal), expected)
    base.validate(grid, start, goal, held, expected)
    empty = PreparedPDQ(base.Grid('empty', ['@@']))
    assert empty.query(0, 0).distance == -1
    print(f'优化专项检查通过：{edge_checks} 条启发值边差、代数溢出、跨目标复用及返回路径独立性', flush=True)
    return count, edge_checks


def run(cases, args):
    rng = random.Random(args.seed)
    rows, preparations = [], []
    # 确认与上轮确为相同地图、起终点和最短距离，不直接复用上轮耗时。
    previous = json.loads((ROOT / 'results/raw.json').read_text(encoding='utf-8'))
    reference_rows = {(r['map'], r['query']): r for r in previous['rows'] if r['algorithm'] == 'pdq'}
    for grid, pairs in cases:
        prepared = PreparedPDQ(grid)
        grid._prepared_pdq = prepared
        preparations.append({
            'map': grid.name, 'shared_prepare_ms': prepared.shared_prepare_ms,
            'landmark_prepare_ms': prepared.prepare_ms, 'buffer_bytes': prepared.buffer_bytes,
            'landmarks': prepared.landmark_nodes, 'components': prepared.component_count,
        })
        print(f'测试 {grid.name}；工作区预处理 {prepared.shared_prepare_ms:.1f} ms，含地标 {prepared.prepare_ms:.1f} ms', flush=True)
        for query, (start, goal) in enumerate(pairs):
            reference = base.bfs(grid, start, goal)
            if args.queries == previous['metadata']['queries_per_map'] and args.seed == previous['metadata']['seed']:
                old = reference_rows[grid.name, query]
                assert (start, goal, reference.distance) == (old['start'], old['goal'], old['distance'])
            warm = {name: method(grid, start, goal) for name, method in METHODS.items()}
            for result in warm.values():
                base.validate(grid, start, goal, result, reference.distance)
            if warm['pdq_fast'].expanded:
                assert warm['pdq_fast'].path == warm['pdq_original'].path
                assert warm['pdq_fast'].expanded == warm['pdq_original'].expanded
            samples = {name: [] for name in METHODS}
            for _ in range(args.repeats):
                order = list(METHODS)
                rng.shuffle(order)
                for name in order:
                    was_enabled = gc.isenabled()
                    gc.disable()
                    try:
                        before = time.perf_counter_ns()
                        result = METHODS[name](grid, start, goal)
                        elapsed = (time.perf_counter_ns() - before) / 1e6
                    finally:
                        if was_enabled:
                            gc.enable()
                    base.validate(grid, start, goal, result, reference.distance)
                    assert result.expanded == warm[name].expanded
                    samples[name].append(elapsed)
            for name, result in warm.items():
                rows.append({
                    'map': grid.name, 'query': query, 'start': start, 'goal': goal,
                    'algorithm': name, 'distance': reference.distance,
                    'expanded': result.expanded, 'scanned': result.scanned,
                    'median_ms': statistics.median(samples[name]), 'samples_ms': samples[name],
                })
        medians = {name: statistics.median(r['median_ms'] for r in rows if r['map'] == grid.name and r['algorithm'] == name) for name in ('pdq_original', 'pdq_fast', 'pdq_landmark')}
        print('  ' + ', '.join(f'{name}: {value:.3f} ms' for name, value in medians.items()), flush=True)
    return rows, preparations


def report(rows, preparations, checks, edge_checks, args):
    output = ROOT / 'results_optimized'
    output.mkdir(exist_ok=True)
    metadata = {
        'python': sys.version, 'platform': platform.platform(), 'seed': args.seed,
        'queries_per_map': args.queries, 'repeats': args.repeats,
        'correctness_queries': checks, 'heuristic_edge_checks': edge_checks,
        'algorithms': list(METHODS), 'baseline_source': 'rerun interleaved, not historical timings',
        'timing': 'search workspace initialization or epoch update + search + complete path reconstruction',
        'static_preprocessing': 'recorded separately and added once to map batch totals',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ('benchmark.py', 'optimized.py', 'optimization_benchmark.py')},
    }
    (output / 'raw.json').write_text(json.dumps({'metadata': metadata, 'preparations': preparations, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'queries.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != 'samples_ms'])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != 'samples_ms'} for row in rows)
    problems = {}
    for row in rows:
        problems.setdefault((row['map'], row['query']), {})[row['algorithm']] = row
    aggregate = {}
    for name in METHODS:
        values = [g[name]['median_ms'] for g in problems.values()]
        ratios = [g['pdq_original']['median_ms'] / g[name]['median_ms'] for g in problems.values()]
        aggregate[name] = {
            'median_ms': statistics.median(values), 'p95_ms': base.percentile(values, .95),
            'sum_query_medians_ms': sum(values), 'paired_speedup_vs_original': statistics.median(ratios),
            'wins_vs_original': sum(r > 1 for r in ratios),
            'paired_speedup_vs_heap_high_g': statistics.median(g['astar_heap_high_g']['median_ms'] / g[name]['median_ms'] for g in problems.values()),
            'paired_speedup_vs_bucket': statistics.median(g['astar_bucket']['median_ms'] / g[name]['median_ms'] for g in problems.values()),
            'expanded': sum(g[name]['expanded'] for g in problems.values()),
        }
    summaries = []
    lines = [
        '# PDQ 优化复测', '',
        f'- Python {platform.python_version()}；{platform.platform()}。',
        f'- 种子 {args.seed}，{len(preparations)} 张地图，每图 {args.queries} 个查询，各算法各查询 {args.repeats} 次正式计时，共 {len(rows) * args.repeats} 次。',
        '- 默认参数复用上轮 240 个问题；原 PDQ、堆 A*、桶队列 A*、BFS 均重新交错计时，没有直接与历史耗时相除。',
        '- 地图准备之后分别计入数组初始化或工作区代数更新、完整搜索和完整路径还原。预热之后随机交错算法顺序，计时期间关闭循环垃圾回收。',
        '- 地标版和工作区版需要静态预处理。查询表排除这部分，后面的批次表明确加回。',
        f'- 正确性：{checks} 个小图问题及全部性能问题通过；另验证 {edge_checks} 条边的启发值差恰好为 1，并检查工作区溢出和跨目标复用。', '',
        '## 在线查询', '',
        '数值为每个问题重复计时的中位数，再在该地图所有问题间取中位数，单位毫秒。', '',
        '| 地图 | 原 PDQ | 快速 PDQ，无预处理 | 工作区 PDQ | 地标 PDQ | 地标全部预处理 ms |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for prep in preparations:
        name = prep['map']
        subset = [g for (map_name, _), g in problems.items() if map_name == name]
        medians = {method: statistics.median(g[method]['median_ms'] for g in subset) for method in METHODS}
        totals = {method: sum(g[method]['median_ms'] for g in subset) for method in METHODS}
        fast_mean = totals['pdq_fast'] / len(subset)
        landmark_mean = totals['pdq_landmark'] / len(subset)
        saving = fast_mean - landmark_mean
        break_even = math.ceil(prep['landmark_prepare_ms'] / saving) if saving > 0 else None
        total_landmark = prep['landmark_prepare_ms'] + totals['pdq_landmark']
        # 消融对照复用同一个完整 PreparedPDQ，部署成本按实际构造函数全额计入。
        total_workspace = prep['landmark_prepare_ms'] + totals['pdq_workspace']
        summaries.append({
            'map': name, 'query_median_ms': medians, 'query_totals_ms': totals,
            'preparation': prep, 'landmark_batch_ms': total_landmark, 'workspace_batch_ms': total_workspace,
            'landmark_batch_speedup_vs_fast': totals['pdq_fast'] / total_landmark,
            'estimated_landmark_break_even_queries_vs_fast': break_even,
        })
        lines.append(f'| {name} | {medians["pdq_original"]:.3f} | {medians["pdq_fast"]:.3f} | {medians["pdq_workspace"]:.3f} | {medians["pdq_landmark"]:.3f} | {prep["landmark_prepare_ms"]:.1f} |')
    lines.extend(['', '## 同一地图整批查询，包含一次预处理', '', '这些总数是逐问题中位耗时相加，再加一次实测预处理时间，不是完整批次单独计时。工作区版是复用完整 PreparedPDQ、关闭地标估值的消融对照，因此也计入完整构造成本；shared_prepare_ms 仅为其中共享阶段耗时。', '', '| 地图 | 快速 PDQ 总 ms | 工作区版总 ms | 地标版总 ms | 地标对快速版加速比 | 地标估计回本查询数 |', '|---|---:|---:|---:|---:|---:|'])
    for s in summaries:
        break_even = s['estimated_landmark_break_even_queries_vs_fast']
        lines.append(f'| {s["map"]} | {s["query_totals_ms"]["pdq_fast"]:.2f} | {s["workspace_batch_ms"]:.2f} | {s["landmark_batch_ms"]:.2f} | {s["landmark_batch_speedup_vs_fast"]:.2f}× | {break_even if break_even is not None else "本样本不回本"} |')
    fast_total = aggregate['pdq_fast']['sum_query_medians_ms']
    landmark_total = sum(s['landmark_batch_ms'] for s in summaries)
    lines.extend([
        '', f'所有地图合计：快速版 {fast_total:.2f} ms；地标版含一次预处理 {landmark_total:.2f} ms。',
        '本批次每图只有少量查询，地标版在全部地图合计的成本更高；默认建议使用无预处理快速版，固定迷宫等复杂地图上多次查询再考虑地标版。',
    ])
    lines.extend(['', '回本查询数按该地图样本的平均查询节省估算，仅适用于相同查询混合；预处理只计时一次，未进行独立重复预处理统计。', '', '## 所有查询的分布', '', '| 实现 | 查询中位数 ms | P95 ms | 对原 PDQ 配对加速比中位数 | 更快问题数 | 总扩展节点 |', '|---|---:|---:|---:|---:|---:|'])
    for name, values in aggregate.items():
        lines.append(f'| {name} | {values["median_ms"]:.3f} | {values["p95_ms"]:.3f} | {values["paired_speedup_vs_original"]:.2f}× | {values["wins_vs_original"]} | {values["expanded"]} |')
    lines.extend([
        '', '## 优化与适用条件', '',
        '- 快速版拆出专用循环，只在成功松弛时决定队首或队尾，避免通用模式判断和逐节点构建四个邻居元组。没有折线捷径时，与原 PDQ 的路径及扩展量保持一致。',
        '- 先检查水平后垂直、垂直后水平两条折线。只有全程可通行才返回，此时路径长度达到曼哈顿下界，必定最优；失败继续完整搜索。探测计入查询时间和 scanned。',
        '- 工作区版保存静态地图快照、连通分量及查询数组。不同分量直接判无解；通过代数标记复用数组，不缓存前次起终点的答案或目标启发值。',
        '- 地标版在最大连通分量内，以最远点采样选择最多四个地标并各做一次 BFS。其他分量退回曼哈顿下界。',
        '- 启发值为曼哈顿距离与各个 |d(地标,u)-d(地标,t)| 的最大值。每项均为一致下界；四方向网格是二分图，各项具有相同奇偶性，故相邻格子的最终启发值差恰为 ±1。势差边权仍为 0 或 1，双端队列仍保证最短路。',
        '- 地标下界属于已有 ALT 思路的组合应用，不宣称学术首创：https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory/ 。',
        '- PreparedPDQ 是不可变地图的独立快照，地图更新后需重建；查询工作区不支持并发，多个工作线程需各自实例。',
        '- buffer_bytes 仅记录持久数组和网格缓冲区大小，不是 Python 进程内存峰值；预处理临时数组及搜索队列不在其中。',
        '- 未做 JPS4、KIA*、带权地图、八方向、动态障碍、跨机器或完整公开数据集测试。原始 results/ 结果保留。',
    ])
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps({'aggregate': aggregate, 'maps': summaries}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    assert args.queries >= 2 and args.repeats >= 1
    checks, edge_checks = check_optimizations()
    if args.check_only:
        return
    cases = base.synthetic_cases(args.queries, args.seed) + base.public_cases(args.queries)
    assert len(cases) == 15
    rows, preparations = run(cases, args)
    report(rows, preparations, checks, edge_checks, args)


if __name__ == '__main__':
    main()
