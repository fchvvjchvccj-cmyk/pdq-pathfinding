"""保留最短路保证的 PDQ 专用循环与静态地图地标优化。"""

from array import array
from collections import deque
from time import perf_counter_ns

from benchmark import INF, Result, finish


def direct_path(walk, pitch, start, goal):
    """只接受达到曼哈顿下界的两种折线路径，失败时继续完整搜索。"""
    sx, sy = start % pitch, start // pitch
    tx, ty = goal % pitch, goal // pitch
    dx = 1 if tx > sx else -1
    dy = pitch if ty > sy else -pitch
    horizontal, vertical = abs(tx - sx), abs(ty - sy)
    scanned = 0
    for segments in (((dx, horizontal), (dy, vertical)), ((dy, vertical), (dx, horizontal))):
        path = [start]
        u = start
        blocked = False
        for offset, length in segments:
            for _ in range(length):
                u += offset
                scanned += 1
                if not walk[u]:
                    blocked = True
                    break
                path.append(u)
            if blocked:
                break
        if not blocked:
            return path, scanned
        if not horizontal or not vertical:
            break
    return None, scanned


def pdq_fast(grid, start, goal):
    """不预处理地图，去掉通用模式分派和逐扩展的邻居元组。"""
    walk, pitch = grid.walk, grid.pitch
    size = len(walk)
    if not (0 <= start < size and 0 <= goal < size) or not walk[start] or not walk[goal]:
        return Result([], 0, 0)
    path, scanned = direct_path(walk, pitch, start, goal)
    if path is not None:
        return Result(path, 0, scanned)
    dist, parent = [INF] * size, [-1] * size
    dist[start] = 0
    queue = deque([(start, 0)])
    tx, ty = goal % pitch, goal // pitch
    offsets = (-1, 1, -pitch, pitch)
    expanded = 0
    while queue:
        u, g = queue.popleft()
        if g != dist[u]:
            continue
        expanded += 1
        if u == goal:
            return finish(parent, start, goal, expanded, scanned)
        x, y = u % pitch, u // pitch
        near_x = u - 1 if x > tx else u + 1 if x < tx else -1
        near_y = u - pitch if y > ty else u + pitch if y < ty else -1
        ng = g + 1
        scanned += 4
        for offset in offsets:
            v = u + offset
            if walk[v] and ng < dist[v]:
                dist[v], parent[v] = ng, u
                if v == near_x or v == near_y:
                    queue.appendleft((v, ng))
                else:
                    queue.append((v, ng))
    return Result([], expanded, scanned)


class PreparedPDQ:
    """为固定地图建立连通分量、四个地标和可重复使用的查询工作区。"""

    def __init__(self, grid, landmark_count=4):
        before = perf_counter_ns()
        self.walk = bytearray(grid.walk)
        self.pitch = grid.pitch
        size = len(self.walk)
        self.components = array('i', [-1]) * size
        self.g = array('i', [0]) * size
        self.parent = array('i', [-1]) * size
        self.h = array('i', [0]) * size
        self.seen = array('I', [0]) * size
        self.stamp = 0
        offsets = (-1, 1, -self.pitch, self.pitch)
        largest = []
        seed = -1
        component_id = 0
        for start in grid.free:
            if self.components[start] != -1:
                continue
            queue = deque([start])
            self.components[start] = component_id
            cells = []
            while queue:
                u = queue.popleft()
                cells.append(u)
                for offset in offsets:
                    v = u + offset
                    if self.walk[v] and self.components[v] == -1:
                        self.components[v] = component_id
                        queue.append(v)
            if len(cells) > len(largest):
                largest = cells
                self.landmark_component = component_id
                seed = cells[-1]
            component_id += 1
        self.component_count = component_id
        self.landmark_component = getattr(self, 'landmark_component', -1)
        self.shared_prepare_ms = (perf_counter_ns() - before) / 1e6
        self.landmarks = []
        self.landmark_nodes = []
        minimum = array('i', [2**31 - 1]) * size
        for _ in range(min(landmark_count, len(largest))):
            distances = array('i', [-1]) * size
            distances[seed] = 0
            queue = deque([seed])
            while queue:
                u = queue.popleft()
                nd = distances[u] + 1
                for offset in offsets:
                    v = u + offset
                    if self.walk[v] and distances[v] == -1:
                        distances[v] = nd
                        queue.append(v)
            self.landmarks.append(distances)
            self.landmark_nodes.append(seed)
            farthest = -1
            for u in largest:
                if distances[u] < minimum[u]:
                    minimum[u] = distances[u]
                if minimum[u] > farthest:
                    farthest, seed = minimum[u], u
            if farthest == 0:
                break
        self.prepare_ms = (perf_counter_ns() - before) / 1e6
        buffers = [self.components, self.g, self.parent, self.h, self.seen, *self.landmarks]
        self.buffer_bytes = len(self.walk) + sum(len(a) * a.itemsize for a in buffers)

    def query(self, start, goal, use_landmarks=True):
        walk, pitch = self.walk, self.pitch
        size = len(walk)
        if not (0 <= start < size and 0 <= goal < size) or not walk[start] or not walk[goal]:
            return Result([], 0, 0)
        if self.components[start] != self.components[goal]:
            return Result([], 0, 0)
        path, scanned = direct_path(walk, pitch, start, goal)
        if path is not None:
            return Result(path, 0, scanned)
        # 每次调用更新代数，避免清空整张地图；不跨查询复用目标相关的启发值。
        self.stamp += 1
        if self.stamp >= 2**32:
            self.seen = array('I', [0]) * size
            self.stamp = 1
        stamp = self.stamp
        seen, g_values, parent, h_values = self.seen, self.g, self.parent, self.h
        tx, ty = goal % pitch, goal // pitch
        landmarks = self.landmarks if use_landmarks and self.components[start] == self.landmark_component else []
        if not landmarks:
            return self._query_manhattan(start, goal, scanned, stamp)
        targets = tuple(distances[goal] for distances in landmarks)

        if len(landmarks) == 4:
            d0, d1, d2, d3 = landmarks
            t0, t1, t2, t3 = targets

            def estimate(v):
                value = abs(v % pitch - tx) + abs(v // pitch - ty)
                bound = abs(d0[v] - t0)
                if bound > value:
                    value = bound
                bound = abs(d1[v] - t1)
                if bound > value:
                    value = bound
                bound = abs(d2[v] - t2)
                if bound > value:
                    value = bound
                bound = abs(d3[v] - t3)
                if bound > value:
                    value = bound
                return value
        else:
            def estimate(v):
                value = abs(v % pitch - tx) + abs(v // pitch - ty)
                for distances, target in zip(landmarks, targets):
                    bound = abs(distances[v] - target)
                    if bound > value:
                        value = bound
                return value

        h_start = estimate(start)
        seen[start], g_values[start], h_values[start] = stamp, 0, h_start
        queue = deque([(start, 0)])
        offsets = (-1, 1, -pitch, pitch)
        expanded = 0
        while queue:
            u, g = queue.popleft()
            if g != g_values[u]:
                continue
            expanded += 1
            if u == goal:
                return finish(parent, start, goal, expanded, scanned)
            hu = h_values[u]
            ng = g + 1
            scanned += 4
            for offset in offsets:
                v = u + offset
                if not walk[v]:
                    continue
                if seen[v] == stamp:
                    if ng >= g_values[v]:
                        continue
                    hv = h_values[v]
                else:
                    hv = estimate(v)
                    h_values[v] = hv
                    seen[v] = stamp
                g_values[v], parent[v] = ng, u
                if hv < hu:
                    queue.appendleft((v, ng))
                else:
                    queue.append((v, ng))
        return Result([], expanded, scanned)

    def _query_manhattan(self, start, goal, scanned, stamp):
        """工作区消融对照也使用专用循环，避免空地标估值函数的额外开销。"""
        walk, pitch = self.walk, self.pitch
        seen, g_values, parent = self.seen, self.g, self.parent
        seen[start], g_values[start] = stamp, 0
        queue = deque([(start, 0)])
        tx, ty = goal % pitch, goal // pitch
        offsets = (-1, 1, -pitch, pitch)
        expanded = 0
        while queue:
            u, g = queue.popleft()
            if g != g_values[u]:
                continue
            expanded += 1
            if u == goal:
                return finish(parent, start, goal, expanded, scanned)
            x, y = u % pitch, u // pitch
            near_x = u - 1 if x > tx else u + 1 if x < tx else -1
            near_y = u - pitch if y > ty else u + pitch if y < ty else -1
            ng = g + 1
            scanned += 4
            for offset in offsets:
                v = u + offset
                if walk[v] and (seen[v] != stamp or ng < g_values[v]):
                    seen[v], g_values[v], parent[v] = stamp, ng, u
                    if v == near_x or v == near_y:
                        queue.appendleft((v, ng))
                    else:
                        queue.append((v, ng))
        return Result([], expanded, scanned)


def prepared_query(grid, start, goal, use_landmarks=True):
    if not hasattr(grid, '_prepared_pdq'):
        grid._prepared_pdq = PreparedPDQ(grid)
    return grid._prepared_pdq.query(start, goal, use_landmarks)
