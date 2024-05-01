from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network

def boykov_kolmogorov(G, s, t):
    R = build_residual_network(G, 'sim')

    for u in R:
        for e in R[u].values():
            e['flow'] = 0
    INF = R.graph['inf']
    cutoff = INF

    R_succ = R.succ
    R_pred = R.pred

    def grow():
        while active:
            u = active[0]
            if u in source_tree:
                this_tree = source_tree
                other_tree = target_tree
                neighbors = R_succ
            else:
                this_tree = target_tree
                other_tree = source_tree
                neighbors = R_pred
            for v, attr in neighbors[u].items():
                if attr['capacity'] - attr['flow'] > 0:
                    if v not in this_tree:
                        if v in other_tree:
                            return (u, v) if this_tree is source_tree else (v, u)
                        this_tree[v] = u
                        dist[v] = dist[u] + 1
                        timestamp[v] = timestamp[u]
                        active.append(v)
                    elif v in this_tree and _is_closer(u, v):
                        this_tree[v] = u
                        dist[v] = dist[u] + 1
                        timestamp[v] = timestamp[u]
            active.popleft()
        return None, None

    def augment(u, v):
        attr = R_succ[u][v]
        flow = min(INF, attr['capacity'] - attr['flow'])
        path = [u]
        w = u
        while w != s:
            n = w
            w = source_tree[n]
            attr = R_pred[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        path.reverse()
        path.append(v)
        w = v
        while w != t:
            n = w
            w = target_tree[n]
            attr = R_succ[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        it = iter(path)
        u = next(it)
        these_orphans = []
        for v in it:
            R_succ[u][v]['flow'] += flow
            R_succ[v][u]['flow'] -= flow
            if R_succ[u][v]['flow'] == R_succ[u][v]['capacity']:
                if v in source_tree:
                    source_tree[v] = None
                    these_orphans.append(v)
                if u in target_tree:
                    target_tree[u] = None
                    these_orphans.append(u)
            u = v
        orphans.extend(sorted(these_orphans, key=dist.get))
        return flow

    def adopt():
        while orphans:
            u = orphans.popleft()
            if u in source_tree:
                tree = source_tree
                neighbors = R_pred
            else:
                tree = target_tree
                neighbors = R_succ
            nbrs = ((n, attr, dist[n]) for n, attr in neighbors[u].items()
                    if n in tree)
            for v, attr, d in sorted(nbrs, key = itemgetter(2)):
                if attr['capacity'] - attr['flow'] > 0:
                    if _has_valid_root(v, tree):
                        tree[u] = v
                        dist[u] = dist[v] + 1
                        timestamp[u] = time
                        break
            else:
                nbrs = ((n, attr, dist[n]) for n, attr in neighbors[u].items()
                        if n in tree)
                for v, attr, d in sorted(nbrs, key = itemgetter(2)):
                    if attr['capacity'] - attr['flow'] > 0:
                        if v not in active:
                            active.append(v)
                    if tree[v] == u:
                        tree[v] = None
                        orphans.appendleft(v)
                if u in active:
                    active.remove(u)
                del tree[u]

    def _has_valid_root(n, tree):
        path = []
        v = n
        while v is not None:
            path.append(v)
            if v == s or v == t:
                base_dist = 0
                break
            elif timestamp[v] == time:
                base_dist = dist[v]
                break
            v = tree[v]
        else:
            return False
        length = len(path)
        for i, u in enumerate(path, 1):
            dist[u] = base_dist + length - i
            timestamp[u] = time
        return True

    def _is_closer(u, v):
        return timestamp[v] <= timestamp[u] and dist[v] > dist[u] + 1

    source_tree = {s: None}
    target_tree = {t: None}
    active = deque([s, t])
    orphans = deque()
    flow_value = 0

    time = 1
    timestamp = {s: time, t: time}
    dist = {s: 0, t: 0}
    while flow_value < cutoff:
        u, v = grow()
        if u is None:
            break
        time += 1
        flow_value += augment(u, v)
        adopt()

    R.graph['trees'] = source_tree
    return R