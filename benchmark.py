"""四方向等权网格上的可复现寻路正确性与性能测试。"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import heapq
import html
import io
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INF = 10**12


@dataclass
class Grid:
    name: str
    rows: list[str]

    def __post_init__(self):
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        assert all(len(row) == self.width for row in self.rows)
        self.pitch = self.width + 2
        self.walk = bytearray((self.height + 2) * self.pitch)
        self.free = []
        for y, row in enumerate(self.rows, 1):
            for x, cell in enumerate(row, 1):
                if cell in '.GS':
                    u = y * self.pitch + x
                    self.walk[u] = 1
                    self.free.append(u)

    def at(self, x, y):
        return (y + 1) * self.pitch + x + 1


@dataclass
class Result:
    path: list[int]
    expanded: int
    scanned: int

    @property
    def distance(self):
        return len(self.path) - 1 if self.path else -1


def finish(parent, start, goal, expanded, scanned):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return Result(path, expanded, scanned)


def bfs(grid, start, goal):
    walk, pitch = grid.walk, grid.pitch
    if not (0 <= start < len(walk) and 0 <= goal < len(walk)):
        return Result([], 0, 0)
    if not walk[start] or not walk[goal]:
        return Result([], 0, 0)
    dist, parent = [INF] * len(walk), [-1] * len(walk)
    dist[start] = 0
    queue = deque([start])
    expanded = scanned = 0
    offsets = (-1, 1, -pitch, pitch)
    while queue:
        u = queue.popleft()
        expanded += 1
        if u == goal:
            return finish(parent, start, goal, expanded, scanned)
        nd = dist[u] + 1
        for offset in offsets:
            scanned += 1
            v = u + offset
            if walk[v] and dist[v] == INF:
                dist[v], parent[v] = nd, u
                queue.append(v)
    return Result([], expanded, scanned)


def informed(grid, start, goal, mode):
    """共用松弛、邻居顺序、数组与路径还原，仅替换开放集合的组织方式。"""
    walk, pitch = grid.walk, grid.pitch
    if not (0 <= start < len(walk) and 0 <= goal < len(walk)):
        return Result([], 0, 0)
    if not walk[start] or not walk[goal]:
        return Result([], 0, 0)
    dist, parent = [INF] * len(walk), [-1] * len(walk)
    dist[start] = 0
    tx, ty = goal % pitch, goal // pitch
    current_f = abs(start % pitch - tx) + abs(start // pitch - ty)
    queue = deque([(start, 0)])
    future = deque()
    heap = [(current_f, 0, 0, start)]
    serial = expanded = scanned = 0
    while heap if mode.startswith('heap') else queue or future:
        if mode.startswith('heap'):
            _, tie, _, u = heapq.heappop(heap)
            g = -tie if mode == 'heap_high_g' else tie
        else:
            if not queue:
                queue, future = future, queue
                current_f += 2
            u, g = queue.popleft()
        if g != dist[u]:
            continue
        expanded += 1
        if u == goal:
            return finish(parent, start, goal, expanded, scanned)
        x, y = u % pitch, u // pitch
        hu = abs(x - tx) + abs(y - ty) if mode != 'pdq' else 0
        neighbors = (
            (u - 1, -1 if x > tx else 1),
            (u + 1, -1 if x < tx else 1),
            (u - pitch, -1 if y > ty else 1),
            (u + pitch, -1 if y < ty else 1),
        )
        ng = g + 1
        for v, dh in neighbors:
            scanned += 1
            if not walk[v] or ng >= dist[v]:
                continue
            dist[v], parent[v] = ng, u
            if mode.startswith('heap'):
                serial += 1
                tie = -ng if mode == 'heap_high_g' else ng
                heapq.heappush(heap, (ng + hu + dh, tie, serial, v))
            elif mode == 'pdq':
                if dh == -1:
                    queue.appendleft((v, ng))
                else:
                    queue.append((v, ng))
            else:
                nf = ng + hu + dh
                if nf == current_f:
                    queue.appendleft((v, ng))
                else:
                    assert nf == current_f + 2
                    future.append((v, ng))
    return Result([], expanded, scanned)


ALGORITHMS = {
    'bfs': bfs,
    'astar_heap_high_g': lambda m, s, t: informed(m, s, t, 'heap_high_g'),
    'astar_heap_low_g': lambda m, s, t: informed(m, s, t, 'heap_low_g'),
    'astar_bucket': lambda m, s, t: informed(m, s, t, 'bucket'),
    'pdq': lambda m, s, t: informed(m, s, t, 'pdq'),
}


def validate(grid, start, goal, result, expected):
    assert result.distance == expected, (grid.name, start, goal, result.distance, expected)
    if not result.path:
        return
    assert result.path[0] == start and result.path[-1] == goal
    assert len(set(result.path)) == len(result.path)
    assert all(grid.walk[u] for u in result.path)
    assert all(abs(u - v) in (1, grid.pitch) for u, v in zip(result.path, result.path[1:]))


def correctness():
    count = 0
    # 穷举全部 3×3 障碍布局及其中所有合法起终点，包含同点和无解情形。
    for mask in range(1 << 9):
        rows = [''.join('.' if mask & (1 << (y * 3 + x)) else '@' for x in range(3)) for y in range(3)]
        grid = Grid(f'exhaustive-{mask}', rows)
        for start in grid.free:
            for goal in grid.free:
                expected = bfs(grid, start, goal).distance
                for name, algorithm in ALGORITHMS.items():
                    validate(grid, start, goal, algorithm(grid, start, goal), expected)
                count += 1
    rng = random.Random(20261103)
    for k in range(120):
        rows = [''.join('.' if rng.random() > .32 else '@' for _ in range(17)) for _ in range(13)]
        grid = Grid(f'fuzz-{k}', rows)
        for _ in range(8):
            start, goal = rng.choices(grid.free, k=2)
            expected = bfs(grid, start, goal).distance
            for algorithm in ALGORITHMS.values():
                validate(grid, start, goal, algorithm(grid, start, goal), expected)
            count += 1
    grid = Grid('invalid', ['.@', '..'])
    invalid = [(grid.at(0, 0), grid.at(1, 0)), (-1, grid.at(0, 0)), (0, 0), (10**6, 0)]
    for start, goal in invalid:
        for algorithm in ALGORITHMS.values():
            validate(grid, start, goal, algorithm(grid, start, goal), -1)
        count += 1
    print(f'正确性通过：{count} 个问题 × {len(ALGORITHMS)} 种算法', flush=True)
    return count


def download_public():
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    wanted = {
        'maze': ['maze512-1-0', 'maze512-16-0'],
        'random': ['random512-10-0', 'random512-30-0'],
        'room': ['16room_000'],
        'dao': ['arena2', 'brc000d'],
    }
    manifest = []
    for family, names in wanted.items():
        index = f'https://www.movingai.com/benchmarks/{family}/index.html'
        page = urllib.request.urlopen(index, timeout=45).read().decode()
        links = re.findall(r'<a\s+[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', page, re.I | re.S)
        for name in names:
            for suffix in ('.map', '.scen'):
                filename = name + suffix
                matches = [href for href, label in links if html.unescape(re.sub('<[^>]+>', '', label)).strip() == filename]
                assert len(matches) == 1, (filename, matches)
                url = urllib.parse.urljoin(index, html.unescape(matches[0]))
                target = data_dir / filename
                if not target.exists():
                    payload = urllib.request.urlopen(url, timeout=45).read()
                    if payload.startswith(b'PK'):
                        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                            member = next(n for n in archive.namelist() if Path(n).name in (filename, name + '.map.scen'))
                            payload = archive.read(member)
                    target.write_bytes(payload)
                manifest.append({'file': filename, 'url': url, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        print(f'公开地图已就绪：{family}', flush=True)
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def public_cases(queries):
    cases = []
    for path in sorted((ROOT / 'data').glob('*.map')):
        lines = path.read_text().splitlines()
        width, height = int(lines[2].split()[1]), int(lines[1].split()[1])
        grid = Grid(path.stem, lines[4:4 + height])
        assert grid.width == width
        scenarios = path.with_suffix('.scen').read_text().splitlines()[1:]
        # 按场景文件顺序分层抽样，保留短、中、长距离；八方向参考距离不用于四方向判定。
        indices = [round(i * (len(scenarios) - 1) / (queries - 1)) for i in range(queries)]
        pairs = []
        for i in indices:
            fields = scenarios[i].split()
            sx, sy, tx, ty = map(int, fields[4:8])
            pairs.append((grid.at(sx, sy), grid.at(tx, ty)))
        cases.append((grid, pairs))
    return cases


def synthetic_cases(queries, seed):
    rng = random.Random(seed)
    cases = []
    for size in (128, 256):
        for kind in ('open', 'random25', 'barriers', 'disconnected'):
            rows = [['.'] * size for _ in range(size)]
            if kind == 'random25':
                rows = [['@' if rng.random() < .25 else '.' for _ in range(size)] for _ in range(size)]
            elif kind == 'barriers':
                for index, x in enumerate(range(8, size - 1, 12)):
                    gap = size - 2 if index % 2 == 0 else 1
                    for y in range(size):
                        if y != gap:
                            rows[y][x] = '@'
            elif kind == 'disconnected':
                for y in range(size):
                    rows[y][size // 2] = '@'
            grid = Grid(f'{kind}-{size}', [''.join(row) for row in rows])
            if kind == 'disconnected':
                pairs = [(grid.at(1, rng.randrange(size)), grid.at(size - 2, rng.randrange(size))) for _ in range(queries)]
            else:
                pairs = [tuple(rng.sample(grid.free, 2)) for _ in range(queries)]
            cases.append((grid, pairs))
    return cases


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def benchmark(cases, repeats, seed):
    rng = random.Random(seed)
    rows = []
    for grid, pairs in cases:
        print(f'测试 {grid.name}: {grid.width}×{grid.height}, {len(pairs)} 个问题', flush=True)
        for query_id, (start, goal) in enumerate(pairs):
            reference = bfs(grid, start, goal)
            warm = {name: algorithm(grid, start, goal) for name, algorithm in ALGORITHMS.items()}
            for result in warm.values():
                validate(grid, start, goal, result, reference.distance)
            assert warm['pdq'].expanded == warm['astar_bucket'].expanded
            assert warm['pdq'].path == warm['astar_bucket'].path
            samples = {name: [] for name in ALGORITHMS}
            for _ in range(repeats):
                order = list(ALGORITHMS)
                rng.shuffle(order)
                for name in order:
                    # 计时包含数组初始化、完整搜索、父节点记录和路径还原，排除校验与地图准备。
                    gc_was_enabled = gc.isenabled()
                    gc.disable()
                    try:
                        before = time.perf_counter_ns()
                        result = ALGORITHMS[name](grid, start, goal)
                        elapsed = time.perf_counter_ns() - before
                    finally:
                        if gc_was_enabled:
                            gc.enable()
                    validate(grid, start, goal, result, reference.distance)
                    samples[name].append(elapsed / 1e6)
            for name in ALGORITHMS:
                result = warm[name]
                rows.append({
                    'map': grid.name, 'width': grid.width, 'height': grid.height,
                    'query': query_id, 'start': start, 'goal': goal,
                    'algorithm': name, 'distance': reference.distance,
                    'expanded': result.expanded, 'scanned': result.scanned,
                    'median_ms': statistics.median(samples[name]),
                    'samples_ms': samples[name],
                })
        pdq = [r['median_ms'] for r in rows if r['map'] == grid.name and r['algorithm'] == 'pdq']
        print(f'  PDQ 中位数 {statistics.median(pdq):.3f} ms', flush=True)
    return rows


def save_report(rows, checks, args):
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    metadata = {
        'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
        'logical_cpus': os.cpu_count(), 'seed': args.seed, 'queries_per_map': args.queries,
        'repeats': args.repeats, 'correctness_queries': checks, 'algorithms': list(ALGORITHMS),
        'movement': 'four-connected, unit cost', 'timing': 'initialization + search + path reconstruction',
    }
    (output / 'raw.json').write_text(json.dumps({'metadata': metadata, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'queries.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != 'samples_ms'])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != 'samples_ms'} for row in rows)
    names = list(ALGORITHMS)
    lines = [
        '# 寻路实测结果', '',
        f'- 环境：{platform.platform()}；Python {platform.python_version()}。',
        f'- 固定种子：{args.seed}；每图 {args.queries} 个问题；每问题每算法 {args.repeats} 次计时，先取中位数。',
        f'- 正确性：{checks} 个独立小图问题，加全部 {len(rows) // len(names)} 个性能问题，五种算法均与 BFS 最短距离一致。',
        '- 耗时包含查询数组初始化、搜索和完整路径还原；不含下载、地图解析、问题选择、正确性校验和结果序列化。',
        '- 表中耗时为该地图所有问题耗时中位数的中位数；加速比为逐问题 A* 高 g 优先耗时 / PDQ 耗时的中位数。',
        '- 单位：毫秒；数值越低越快。', '',
        '| 地图 | BFS | A* 高 g 优先 | A* 低 g 优先 | 桶队列 A* | PDQ | PDQ 对高 g A* 加速比 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    summaries = []
    for map_name in dict.fromkeys(r['map'] for r in rows):
        subset = [r for r in rows if r['map'] == map_name]
        times = {name: statistics.median(r['median_ms'] for r in subset if r['algorithm'] == name) for name in names}
        by_query = {(r['query'], r['algorithm']): r for r in subset}
        ratios = [by_query[q, 'astar_heap_high_g']['median_ms'] / by_query[q, 'pdq']['median_ms'] for q in range(args.queries)]
        ratio = statistics.median(ratios)
        summaries.append({'map': map_name, 'median_ms': times, 'pdq_speedup_vs_heap_high_g': ratio})
        lines.append('| ' + map_name + ' | ' + ' | '.join(f'{times[name]:.3f}' for name in names) + f' | {ratio:.2f}× |')
    by_problem = {}
    for row in rows:
        by_problem.setdefault((row['map'], row['query']), {})[row['algorithm']] = row
    heap_ratios = [g['astar_heap_high_g']['median_ms'] / g['pdq']['median_ms'] for g in by_problem.values()]
    bucket_ratios = [g['astar_bucket']['median_ms'] / g['pdq']['median_ms'] for g in by_problem.values()]
    lines.extend([
        '', '## 配对比较', '',
        f'- 共 {len(by_problem)} 个性能问题、{len(rows) * args.repeats} 次正式计时，其中 {sum(g["pdq"]["distance"] == -1 for g in by_problem.values())} 个问题无解。',
        f'- PDQ 在 {sum(r > 1 for r in heap_ratios)} 个问题上快于高 g 优先堆 A*；逐问题配对加速比中位数为 {statistics.median(heap_ratios):.3f}×，几何平均为 {math.exp(statistics.mean(math.log(r) for r in heap_ratios)):.3f}×。',
        f'- 相对桶队列 A*，逐问题配对加速比中位数为 {statistics.median(bucket_ratios):.3f}×。',
        '- 加速比超过 1 表示 PDQ 更快。这些统计仅对应本次采样及实现。',
        '', '## 全部问题的分布', '', '| 算法 | 中位数 ms | P95 ms | 总扩展节点 |', '|---|---:|---:|---:|',
    ])
    for name in names:
        subset = [r for r in rows if r['algorithm'] == name]
        values = [r['median_ms'] for r in subset]
        lines.append(f'| {name} | {statistics.median(values):.3f} | {percentile(values, .95):.3f} | {sum(r["expanded"] for r in subset)} |')
    lines.extend([
        '', '## 解释与限制', '',
        '- PDQ 与桶队列 A* 使用相同的同优先级顺序；全部性能问题的路径和扩展节点数完全一致。两者属于同一搜索排序的不同实现，不能据此宣称新算法突破。',
        '- 两个堆 A* 分别在 f 相同时优先较大的 g 或较小的 g；同优先级策略可能显著影响扩展量，主对照为较大 g 优先。',
        '- 公开场景原有参考距离采用其原始移动规则；本测试只复用起终点，重新用四方向 BFS 计算最短距离。',
        '- 本次没有实现或测试 JPS4、KIA*、C++ 实现、多代理寻路、动态障碍和非均匀权重，不能把结果外推至这些算法或条件。',
        '- 地图仅抽取七张公开地图和八张合成地图；未对完整公开测试集作统计结论。',
        '- 各方法共用带边框网格与数组布局；三个启发式队列在同一核心函数内替换，未对某一种方法单独作激进优化。',
        '- 单进程串行运行，逐问题预热，随机交错计时，计时期间关闭循环垃圾回收；未固定 CPU 频率或进程亲和性，结果存在系统噪声。',
        '', '## 数据来源', '',
        '- Moving AI Lab：https://www.movingai.com/benchmarks/grids.html 。原始数据采用 Open Data Commons Attribution License。',
        '- 文件来源与 SHA-256 见 `../data/manifest.json`。',
        '- 原始每次计时见 `raw.json`，逐问题中位数、路径长度与扩展量见 `queries.csv`。',
    ])
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print('\n'.join(lines[:12 + len(summaries)]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    assert args.queries >= 2 and args.repeats >= 1
    checks = correctness()
    if args.check_only:
        return
    if args.download:
        download_public()
    cases = synthetic_cases(args.queries, args.seed) + public_cases(args.queries)
    assert len(cases) == 15, '公开地图不完整，请先加 --download'
    rows = benchmark(cases, args.repeats, args.seed)
    save_report(rows, checks, args)


if __name__ == '__main__':
    main()
