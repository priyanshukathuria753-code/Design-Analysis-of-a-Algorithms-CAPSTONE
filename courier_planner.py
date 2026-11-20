"""
Mini-Courier Planner  -  DAA Capstone (ETCCDA302)
-------------------------------------------------
Pipeline:
  1. Load parcels + road graph from CSV files
  2. Merge Sort   -> order parcels by deadline (then priority)
  3. 0/1 Knapsack -> choose the best set of parcels for ONE van trip (DP)
  4. Dijkstra     -> shortest distance between delivery points
  5. Nearest-Neighbour TSP (approximation) -> final delivery route
  6. Performance tests on small & large inputs + runtime charts

Run:  python courier_planner.py
"""
import csv, heapq, random, time, os
import matplotlib
matplotlib.use("Agg")           # draw charts without needing a window
import matplotlib.pyplot as plt

VAN_CAPACITY = 20               # kg the van can carry in one trip
DEPOT = "Depot"
os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------
# STEP 0: DATA LOADING
# ---------------------------------------------------------------
def load_parcels(path):
    """Read parcels.csv -> list of dictionaries."""
    parcels = []
    with open(path) as f:
        for row in csv.DictReader(f):
            parcels.append({
                "id": row["parcel_id"],
                "dest": row["destination"],
                "weight": int(row["weight_kg"]),
                "priority": int(row["priority"]),      # 5 = most important
                "deadline": int(row["deadline_hr"]),   # hours left
            })
    return parcels


def load_graph(path):
    """Read graph.csv -> adjacency list {node: [(neighbour, distance), ...]}"""
    graph = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            a, b, d = row["node1"], row["node2"], float(row["distance_km"])
            graph.setdefault(a, []).append((b, d))
            graph.setdefault(b, []).append((a, d))   # roads are two-way
    return graph


# ---------------------------------------------------------------
# STEP 1: SORTING  (Merge Sort - O(n log n), stable)
# ---------------------------------------------------------------
def sort_key(p):
    # earlier deadline first; if tie, higher priority first
    return (p["deadline"], -p["priority"])


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if sort_key(left[i]) <= sort_key(right[j]):
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def bubble_sort(arr):
    """Slow O(n^2) sort - used only to COMPARE against merge sort."""
    arr = arr[:]
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if sort_key(arr[j]) > sort_key(arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# ---------------------------------------------------------------
# STEP 2: 0/1 KNAPSACK (Dynamic Programming) - O(n * W)
# value of a parcel = priority score (priority * 10) so urgent
# parcels are preferred. weight = parcel weight in kg.
# ---------------------------------------------------------------
def parcel_value(p):
    return p["priority"] * 10


def knapsack(parcels, capacity):
    n = len(parcels)
    # dp[i][w] = best value using first i parcels with weight limit w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wt, val = parcels[i - 1]["weight"], parcel_value(parcels[i - 1])
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]                       # skip parcel
            if wt <= w:                                   # or take it
                dp[i][w] = max(dp[i][w], dp[i - 1][w - wt] + val)
    # backtrack to find WHICH parcels were chosen
    chosen, w = [], capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(parcels[i - 1])
            w -= parcels[i - 1]["weight"]
    chosen.reverse()
    return dp[n][capacity], chosen, dp


# ---------------------------------------------------------------
# STEP 3: DIJKSTRA - shortest path from one source to all nodes
# ---------------------------------------------------------------
def dijkstra(graph, source):
    dist = {node: float("inf") for node in graph}
    prev = {node: None for node in graph}
    dist[source] = 0
    pq = [(0, source)]                      # min-heap of (distance, node)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                        # old, outdated entry
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, prev


def get_path(prev, target):
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    return path[::-1]


# ---------------------------------------------------------------
# STEP 4: NEAREST-NEIGHBOUR TSP (approximation)
# Always drive to the closest unvisited stop, then return to depot.
# ---------------------------------------------------------------
def nearest_neighbour_route(graph, stops, start=DEPOT):
    unvisited = set(stops)
    unvisited.discard(start)
    route, total, current = [start], 0.0, start
    full_path = [start]
    while unvisited:
        dist, prev = dijkstra(graph, current)
        nxt = min(unvisited, key=lambda s: dist[s])      # closest stop
        total += dist[nxt]
        full_path += get_path(prev, nxt)[1:]             # real roads used
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    dist, prev = dijkstra(graph, current)                # go back home
    total += dist[start]
    full_path += get_path(prev, start)[1:]
    route.append(start)
    return route, total, full_path


# ---------------------------------------------------------------
# STEP 5: PERFORMANCE TESTING
# ---------------------------------------------------------------
def random_parcels(n, nodes):
    random.seed(n)
    return [{"id": f"P{i}", "dest": random.choice(nodes),
             "weight": random.randint(1, 10), "priority": random.randint(1, 5),
             "deadline": random.randint(1, 72)} for i in range(n)]


def random_graph(v):
    """Connected random graph: a chain + extra random roads."""
    random.seed(v)
    names = [DEPOT] + [f"C{i}" for i in range(1, v)]
    g = {n: [] for n in names}
    def add(a, b, d):
        g[a].append((b, d)); g[b].append((a, d))
    for i in range(1, v):
        add(names[i - 1], names[i], random.randint(1, 20))
    for _ in range(v * 2):
        a, b = random.sample(names, 2)
        add(a, b, random.randint(1, 20))
    return g


def timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - t) * 1000      # milliseconds


def run_performance():
    sizes = [10, 50, 100, 200, 400]
    rows = []
    print("\n=== PERFORMANCE TEST (time in ms) ===")
    print(f"{'n':>5} {'Merge':>9} {'Bubble':>9} {'Knapsack':>10} {'Dijkstra':>10} {'NN-TSP':>9}")
    for n in sizes:
        g = random_graph(n)
        nodes = [x for x in g if x != DEPOT]
        parcels = random_parcels(n, nodes)
        _, t_merge = timed(merge_sort, parcels)
        _, t_bub = timed(bubble_sort, parcels)
        _, t_knap = timed(knapsack, parcels, VAN_CAPACITY * 5)
        _, t_dij = timed(dijkstra, g, DEPOT)
        stops = random.sample(nodes, min(30, len(nodes)))    # 30 stops per trip
        _, t_tsp = timed(nearest_neighbour_route, g, stops)
        rows.append((n, t_merge, t_bub, t_knap, t_dij, t_tsp))
        print(f"{n:>5} {t_merge:>9.2f} {t_bub:>9.2f} {t_knap:>10.2f} {t_dij:>10.2f} {t_tsp:>9.2f}")
    return rows


def make_charts(rows):
    n = [r[0] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(n, [r[1] for r in rows], "o-", label="Merge Sort  O(n log n)")
    plt.plot(n, [r[2] for r in rows], "s-", label="Bubble Sort  O(n^2)")
    plt.xlabel("Number of parcels (n)"); plt.ylabel("Time (ms)")
    plt.title("Sorting: Merge Sort vs Bubble Sort"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig("outputs/chart_sorting.png", dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(n, [r[1] for r in rows], "o-", label="Merge Sort")
    plt.plot(n, [r[3] for r in rows], "^-", label="Knapsack DP")
    plt.plot(n, [r[4] for r in rows], "d-", label="Dijkstra")
    plt.plot(n, [r[5] for r in rows], "x-", label="Nearest-Neighbour TSP")
    plt.xlabel("Input size (n)"); plt.ylabel("Time (ms)")
    plt.title("Runtime of Major Steps"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig("outputs/chart_all_steps.png", dpi=130, bbox_inches="tight"); plt.close()
    print("Charts saved in outputs/")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    parcels = load_parcels("data/parcels.csv")
    graph = load_graph("data/graph.csv")

    print("=" * 60)
    print(" MINI-COURIER PLANNER")
    print("=" * 60)
    print(f"Van capacity: {VAN_CAPACITY} kg | Parcels loaded: {len(parcels)}")

    # --- Step 1: sort
    sorted_p = merge_sort(parcels)
    print("\n[1] Parcels sorted by deadline (then priority):")
    print(f"{'ID':<5}{'Dest':<6}{'Wt':>3}{'Pri':>4}{'Deadline(h)':>12}")
    for p in sorted_p:
        print(f"{p['id']:<5}{p['dest']:<6}{p['weight']:>3}{p['priority']:>4}{p['deadline']:>12}")

    # --- Step 2: knapsack
    best, chosen, dp = knapsack(sorted_p, VAN_CAPACITY)
    print(f"\n[2] Knapsack result: best total value = {best}")
    print("    Selected parcels:", ", ".join(p["id"] for p in chosen))
    print("    Total weight:", sum(p["weight"] for p in chosen), "kg")
    left = [p["id"] for p in sorted_p if p not in chosen]
    print("    Left for next trip:", ", ".join(left))
    print("    DP table (last row, capacity 0..%d):" % VAN_CAPACITY, dp[-1])

    # --- Step 3: Dijkstra from depot
    dist, prev = dijkstra(graph, DEPOT)
    print("\n[3] Dijkstra shortest paths from Depot:")
    for node in sorted(dist):
        if node != DEPOT:
            print(f"    Depot -> {node}: {dist[node]:>4.0f} km via {' -> '.join(get_path(prev, node))}")

    # --- Step 4: route
    stops = sorted({p["dest"] for p in chosen})
    route, total, full = nearest_neighbour_route(graph, stops)
    print("\n[4] Nearest-Neighbour delivery route:")
    print("    Stops order :", " -> ".join(route))
    print("    Road path   :", " -> ".join(full))
    print(f"    Total route length = {total:.0f} km")

    # --- Step 5: performance
    rows = run_performance()
    make_charts(rows)


if __name__ == "__main__":
    main()
