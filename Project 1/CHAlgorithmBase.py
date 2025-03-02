import networkx as nx
import heapq
import matplotlib.pyplot as plt
from networkx.exception import NetworkXNoPath
import math

#CH_edge_diff()
 # tie breaker ordering is least number of shortcuts added

"""
def preprocess_tnr(graph, num_transit_nodes):
    # Step 1: Calculate betweenness centrality for all nodes
    centrality = nx.betweenness_centrality(graph, weight='weight')

    # Step 2: Select the top-n nodes with highest betweenness centrality as transit nodes
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    transit_nodes = [node for node, _ in sorted_nodes[:num_transit_nodes]]

    # Step 3: Compute shortest paths between all pairs of transit nodes
    distance = {}
    for u in transit_nodes:
        distance[u] = {}
        for v in transit_nodes:
            if u == v:
                distance[u][v] = 0
            else:
                distance[u][v] = nx.shortest_path_length(graph, u, v, weight='weight')

    return transit_nodes, distance


# Example usage:
# graph = nx.random_geometric_graph(100, 0.1)
# num_transit_nodes = 10  # Number of transit nodes to select
# transit_nodes, distance = preprocess_tnr(graph, num_transit_nodes)


def query_tnr(graph, source, target, transit_nodes, distance):
    # Step 1: If source and target are transit nodes, return precomputed distance
    if source in transit_nodes and target in transit_nodes:
        return distance[source][target]

    # Step 2: Otherwise, compute the shortest path from source to transit nodes
    shortest_distances_from_source = {}
    for tn in transit_nodes:
        shortest_distances_from_source[tn] = nx.shortest_path_length(graph, source, tn, weight='weight')

    # Step 3: Compute the shortest path from transit nodes to target
    shortest_distances_to_target = {}
    for tn in transit_nodes:
        shortest_distances_to_target[tn] = nx.shortest_path_length(graph, tn, target, weight='weight')

    # Step 4: Combine distances to get the shortest path from source to target via transit nodes
    min_distance = float('inf')
    for tn in transit_nodes:
        total_distance = shortest_distances_from_source[tn] + shortest_distances_to_target[tn]
        min_distance = min(min_distance, total_distance)

    return min_distance


# Example usage:
# source = 0
# target = 99
# shortest_distance = query_tnr(graph, source, target, transit_nodes, distance)
# print(f"Shortest distance from {source} to {target} is {shortest_distance}")

"""


#Credit to ChatGPT for framework, but various edits and heavily debugged by me (me=Anderson Worcester)

import networkx as nx

def contract_node(G, H, v):
    """
    Contracts node v in an undirected MultiDiGraph G.
    For every unique pair of neighbors (u, w) of v,
    if a shortcut edge u-w is needed to preserve shortest path distances,
    it is added with weight = best_weight(u,v) + best_weight(v,w).
    """
    # Get the list of neighbors (undirected, so no distinction between in/out)
    neighbors = list(H.neighbors(v))

    # For each unique pair of neighbors, consider a potential shortcut.
    neighbor_shortest_routes = []
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            w = neighbors[j]
            # Find the best (minimal) weight on the edges u-v and v-w.

            shortest_weight = nx.shortest_path_length(H, source=u, target=w, weight='weight')  #, unused_path = nx.bidirectional_dijkstra(G, u, w, weight='weight') # nx.shortest_path_length(H, source=u, target=w, weight='weight')
            neighbor_shortest_routes.append(shortest_weight)

    H.remove_node(v)

    shortcuts_added_for_node = []

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            w = neighbors[j]
            # Find the best (minimal) weight on the edges u-v and v-w.

            try:
                shortest_weight_without_v = nx.shortest_path_length(H, source=u, target=w, weight='weight')  #, unused_path = nx.bidirectional_dijkstra(G, u, w, weight='weight') # nx.shortest_path_length(H, source=u, target=w, weight='weight')
            except NetworkXNoPath:
                shortest_weight_without_v = math.inf

            shortest_possible_weight = neighbor_shortest_routes.pop(0)

            if shortest_weight_without_v > shortest_possible_weight:
                # Add a shortcut edge between u and w.
                # print("adding edge...")
                H.add_edge(u, w, weight=shortest_possible_weight, shortcut=True)
                G.add_edge(u, w, weight=shortest_possible_weight, shortcut=True)
                shortcuts_added_for_node.append((u, w, shortest_possible_weight))
    return G, shortcuts_added_for_node

def compute_potential_shortcuts(G, v):
    """
    Computes the number of shortcut edges that would be added if node v were contracted.
    This serves as a tie-breaker in the ordering.
    """
    potential_shortcuts = 0
    neighbors = list(G.neighbors(v))
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            w = neighbors[j]


            shortest_weight = nx.shortest_path_length(G, source=u, target=w, weight='weight')  #, unused_path = nx.bidirectional_dijkstra(G, u, w, weight='weight') # nx.shortest_path_length(G, source=u, target=w, weight='weight')

            node_temp = G.nodes[v]
            edges_connected = list(G.edges(v, data=True))
            G.remove_node(v)

            try:
                shortest_weight_without_v = nx.shortest_path_length(G, source=u, target=w, weight='weight')  # , unused_path = nx.bidirectional_dijkstra(G, u, w, weight='weight') #nx.shortest_path_length(G, source=u, target=w, weight='weight')
            except NetworkXNoPath:
                shortest_weight_without_v = math.inf

            G.add_node(v, **node_temp)
            for edge in edges_connected:
                G.add_edge(edge[0], edge[1], **edge[2])

            if shortest_weight_without_v > shortest_weight:
                potential_shortcuts += 1

    return potential_shortcuts


def contraction_hierarchy(G):
    """
    Constructs a contraction hierarchy on an undirected MultiDiGraph G.
    At each step, it selects the node with the lowest degree (number of incident edges).
    Ties are broken by choosing the node that would add the fewest shortcuts upon contraction.

    Returns:
        A list representing the order in which nodes were contracted.

    Note: The input graph G is modified in place.
    """
    # Make a working copy if you want to preserve the original graph.
    H = G.copy() # graph to trim down to nothing
    F = G.copy() # graph to not add shortcuts but do add node ordering
    order_counter = 0
    contraction_order = []
    all_shortcuts = []

    while H.number_of_nodes() > 0:
        best_node = None
        best_degree = float('inf')
        best_shortcuts = float('inf')

        # Evaluate each node by its degree and potential shortcuts.
        for v in list(H.nodes()):
            degree = H.degree(v)
            shortcuts = compute_potential_shortcuts(H, v)

            if degree < best_degree or (degree == best_degree and shortcuts < best_shortcuts):
                best_node = v
                best_degree = degree
                best_shortcuts = shortcuts

        contraction_order.append(best_node)
        G.nodes[best_node]['order'] = order_counter
        F.nodes[best_node]['order'] = order_counter
        order_counter += 1
        G, shortcuts_added_for_node = contract_node(G, H, best_node)
        if len(shortcuts_added_for_node) > 0:
            all_shortcuts = all_shortcuts + shortcuts_added_for_node

    return contraction_order, all_shortcuts, F


def ch_query(G, source, target):
    """
    Runs a bidirectional Dijkstra search on the original graph G using the CH ordering.
    In the forward search, from node u only neighbors v with G.nodes[u]['order'] < G.nodes[v]['order']
    are relaxed. In the backward search, only neighbors v with G.nodes[v]['order'] < G.nodes[u]['order']
    are relaxed.

    Instead of using the built-in successors method, we iterate over all neighbors.
    The function also collects all nodes that were popped from the priority queues (i.e. "explored").

    Returns:
        mu: the best (shortest) distance from source to target (or infinity if no path exists),
        explored: a list of nodes that were explored during the search.
    """
    # weight = lambda u, v, d: 1 if G.nodes[u]["order"] > G.nodes[v]["order"] else None
    # length, path = nx.bidirectional_dijkstra(G, source, target, weight="weight")

    INF = float('inf')
    # Initialize distances for forward and backward searches.
    d_f = {node: INF for node in G.nodes()}
    d_b = {node: INF for node in G.nodes()}
    d_f[source] = 0
    d_b[target] = 0

    pq_f = [(0, source)]
    pq_b = [(0, target)]

    nodes_explored = []
    mu = INF

    while pq_f or pq_b:

        # Forward search step.
        if pq_f:
            d, u = heapq.heappop(pq_f)
            nodes_explored.append(u)
            if d < d_f[u]:
                continue
            for v in G.neighbors(u):
                # Relax only if the neighbor is "upward" in CH order.
                if G.nodes[u].get('order', -1) < G.nodes[v].get('order', -1):
                    edge_weight = min(data.get('weight', 1) for data in G.get_edge_data(u, v).values())
                    newd = d_f[u] + edge_weight
                    if newd < d_f[v]:
                        d_f[v] = newd
                        heapq.heappush(pq_f, (newd, v))
                        # If u was also reached by the backward search, update mu.
                        if v in d_b and d_f[v] + d_b[v] < mu:
                            mu = d_f[v] + d_b[v]

        # Backward search step.
        if pq_b:
            d, u = heapq.heappop(pq_b)
            nodes_explored.append(u)
            if d > d_b[u]:
                continue
            for v in G.neighbors(u):
                # Just like forward search, only ascend CH order:
                if G.nodes[u].get('order', -1) < G.nodes[v].get('order', -1):
                    # Here, the edge considered is v->u (remember: the graph is undirected).
                    edge_weight = min(data.get('weight', 1) for data in G.get_edge_data(u, v).values())
                    newd = d_b[u] + edge_weight
                    if newd < d_b[v]:
                        d_b[v] = newd
                        heapq.heappush(pq_b, (newd, v))
                        if v in d_f and d_f[v] + d_b[v] < mu:
                            mu = d_f[v] + d_b[v]
    return mu, nodes_explored


# Example usage:
if __name__ == "__main__":
    # Create a simple undirected MultiDiGraph.
    G = nx.MultiDiGraph()
    G = G.to_undirected()

    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_node('D')
    G.add_node('E')
    G.add_node('F')

    G.add_edge('A', 'B', weight=2)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('C', 'F', weight=2)
    G.add_edge('D', 'F', weight=2)
    G.add_edge('B', 'D', weight=2)
    G.add_edge('E', 'C', weight=1)

    plt.figure()
    pos = nx.spring_layout(G, weight='weight', seed=2)  # shell_layout, planar_layout
    nx.draw(G, pos,
            with_labels=True)  # fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=1, bgcolor='k', node_color='r')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]}' for u, v, d in G.edges(data=True)})
    plt.savefig("dx_example Graph")

    shortest_weight_without_v = nx.shortest_path_length(G, source='E', target='D', weight='weight')

    order, edges_added, F = contraction_hierarchy(G)
    print("Contraction order:", order)
    # At this point, G has been contracted and contains any shortcut edges added.


    # Run the CH query.
    source = 'A'
    target = 'D'
    distance, explored_nodes = ch_query(G, source, target)
    print(f"Shortest distance from {source} to {target}: {distance}")
    print("Explored nodes: ", explored_nodes)

    plt.figure()
    pos = nx.spring_layout(G, weight='weight', seed=2)  # shell_layout, planar_layout
    nx.draw(G, pos,
            with_labels=True)  # fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=1, bgcolor='k', node_color='r')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]}' for u, v, d in G.edges(data=True)})
    plt.savefig("dxx_example Graph")