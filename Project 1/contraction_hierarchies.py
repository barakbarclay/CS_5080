import networkx as nx
from typing import Tuple, List, Dict
from bidirectional_dijkstra import bidirectional_dijkstra

# This code was written with assistance from Gemini and GitHub Copilot


def process_node(
    graph: nx.Graph,
    node: str,
    update_graph: bool = False,
    shortcut_graph: nx.Graph = None,
    criterion: str = "edge_difference",
) -> Tuple[int, int]:
    """Processes a node, creates shortcuts, and optionally updates the graphs.

    Args:
        graph (nx.Graph): The graph to process the node in.
        node (str): The node to process.
        update_shortcut_graph (bool): Whether to update the shortcut graph.
        shortcut_graph (nx.Graph): The shortcut graph to update if update_shortcut_graph is True.
        criterion (str): The criterion to order nodes by ("edge_difference", "shortcuts_added", or "edges_removed").

    Returns:
        Tuple[int, int]: The node's rank and the number of shortcuts added.
    """
    neighbors = list(graph.neighbors(node))
    shortcuts_added = 0

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            v = neighbors[j]
            if graph.has_edge(u, node) and graph.has_edge(node, v):
                weight = graph[u][node]["weight"] + graph[node][v]["weight"]
                if not graph.has_edge(u, v) or graph[u][v]["weight"] > weight:
                    if not graph.has_edge(u, v):
                        shortcuts_added += 1
                    if update_graph:
                        graph.add_edge(u, v, weight=weight)
                        if shortcut_graph is not None:
                            shortcut_graph.add_edge(u, v, weight=weight)

    edges_removed = len(list(graph.edges(node)))  # Edges connected to the node
    if update_graph:
        graph.remove_node(node)
    if criterion == "shortcuts_added":
        rank = shortcuts_added
    elif criterion == "edges_removed":
        rank = edges_removed
    else:
        rank = shortcuts_added - edges_removed
    return rank, shortcuts_added


def create_contraction_hierarchy(
    graph: nx.Graph, online: bool = False, criterion: str = "edge_difference"
) -> Tuple[nx.Graph, List[str], int]:
    """Creates a contraction hierarchy using the criterion.

    Args:
        graph (nx.Graph): The input graph.
        online (bool): Whether to use online calculation.
        criterion (str): The criterion to order nodes by ("edge_difference", "shortcuts_added", or "edges_removed").

    Returns:
        Tuple[nx.Graph, List[str], int]: The contraction hierarchy graph, node order, and number of shortcuts added.
    """
    # Calculate offline ranks for all nodes
    rank: Dict[str, int] = {}
    nodes = list(
        graph.nodes()
    )  # Create a list of nodes to avoid modifying the graph during iteration
    for node in nodes:
        rank[node] = process_node(graph, node, criterion=criterion)[0]

    # Order nodes by the specified criterion (ascending)
    node_order = sorted(rank, key=rank.get)

    # Contract nodes in the calculated order
    temp_graph1 = graph.copy()
    shortcut_graph = graph.copy()
    shortcuts_added = 0
    final_node_order = []

    if online:
        remaining_node_order = node_order.copy()
        for _ in range(len(node_order) - 1):
            final_node_order.append(remaining_node_order[0])
            # Contract nodes in the calculated order
            shortcuts_added += process_node(
                temp_graph1,
                remaining_node_order[0],
                update_graph=True,
                shortcut_graph=shortcut_graph,
                criterion=criterion,
            )[1]
            # Recompute ranks for remaining nodes
            remaining_ranks = {}
            for remaining_node in temp_graph1.nodes():
                if remaining_node != remaining_node_order[0]:
                    temp_graph2 = temp_graph1.copy()
                    remaining_ranks[remaining_node] = process_node(
                        temp_graph2, remaining_node, criterion=criterion
                    )[0]
                    rank[remaining_node] = remaining_ranks[remaining_node]
            remaining_node_order = sorted(remaining_ranks, key=remaining_ranks.get)
            # print("Remaining Node Order:", remaining_node_order)

        # Reorder nodes by the specified criterion (ascending)
        final_node_order.append(remaining_node_order[0])
        node_order = final_node_order
    else:
        for node in node_order:
            shortcuts_added += process_node(
                temp_graph1,
                node,
                update_graph=True,
                shortcut_graph=shortcut_graph,
                criterion=criterion,
            )[1]

    return nx.compose(shortcut_graph, graph), node_order, shortcuts_added


def find_shortest_path_nx(
    graph: nx.Graph, source: str, target: str
) -> Tuple[List[str], int]:
    """Finds the shortest path and its length using the contraction hierarchy.

    Args:
        graph (nx.Graph): The contraction hierarchy graph.
        source (str): The source node.
        target (str): The target node.

    Returns:
        Tuple[List[str], int]: The shortest path and its length.
    """
    if source not in graph or target not in graph:
        raise ValueError("Source or target node not in graph")
    path = nx.shortest_path(graph, source, target, weight="weight")
    length = nx.shortest_path_length(graph, source, target, weight="weight")
    return path, length


def find_shortest_path_custom(
    graph: nx.Graph, source: str, target: str, node_order: List[str]
) -> Tuple[List[str], int]:
    """Finds the shortest path and its length using the contraction hierarchy.

    Args:
        graph (nx.Graph): The contraction hierarchy graph.
        source (str): The source node.
        target (str): The target node.
        node_order (List[str]): The order of nodes in the contraction hierarchy.

    Returns:
        Tuple[List[str], int]: The shortest path and its length.
    """
    if source not in graph or target not in graph:
        raise ValueError("Source or target node not in graph")
    # Create a mapping from node to its order
    node_order_map = {node: order for order, node in enumerate(node_order)}

    # Use custom bidirectional Dijkstra's algorithm
    path, length = bidirectional_dijkstra(graph, source, target, node_order_map)

    return path, length


# 1. Create the graph
graph = nx.Graph()
edges = [
    ("A", "B", 4),
    ("B", "C", 2),
    ("B", "G", 1),
    ("C", "D", 1),
    ("D", "E", 3),
    ("D", "I", 1),
    ("E", "J", 3),
    ("F", "G", 1),
    ("G", "H", 2),
    ("G", "L", 1),
    ("I", "J", 1),
    ("I", "N", 3),
    ("J", "O", 3),
    ("K", "L", 1),
    ("K", "P", 1),
    ("L", "M", 3),
    ("M", "N", 3),
    ("N", "O", 3),
    ("P", "Q", 1),
    ("Q", "R", 3),
    ("Q", "V", 1),
    ("R", "S", 3),
    ("S", "T", 3),
    ("T", "Y", 3),
    ("U", "V", 3),
    ("V", "W", 2),
    ("W", "X", 2),
    ("X", "Y", 2),
]
for u, v, weight in edges:
    graph.add_edge(u, v, weight=weight)

graph_copy = graph.copy()
criteria = ["edge_difference", "shortcuts_added", "edges_removed"]
online_options = [True, False]

for criterion in criteria:
    for online in online_options:
        # 2. Create the contraction hierarchy
        print(f"Criterion: {criterion}")
        print(f"Online calculation: {online}")
        graph_copy = graph.copy()
        ch_graph, node_order, shortcuts_added = create_contraction_hierarchy(
            graph_copy, online=online, criterion=criterion
        )

        # 3. Print the number of shortcuts added
        print(f"Shortcuts added: {shortcuts_added}")

        # 4. Print the node order
        # print("Node Order:", node_order)

        # 5. Find the shortest path

        # 6. Find all pairs shortest paths
        nodes_without_src = graph.nodes()
        for src in graph.nodes():
            nodes_without_src = nodes_without_src - {src}
            for tgt in nodes_without_src:
                    # Uncomment this to use the networkx shortest path function
                    # shortest_path, path_length = find_shortest_path_nx(ch_graph, source_node, target_node)
                    shortest_path, path_length = find_shortest_path_custom(
                        ch_graph, src, tgt, node_order
                    )
                    print(f"Shortest Path from {src} to {tgt}: {shortest_path} with length {path_length}")
