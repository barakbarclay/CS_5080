import networkx as nx
from typing import Tuple, List, Dict
from bidirectional_dijkstra import bidirectional_dijkstra

# This code was written with assistance from Gemini and GitHub Copilot


def contract_node(
    graph: nx.Graph,
    node: str,
    update_shortcut_graph: bool = False,
    shortcut_graph: nx.Graph = None,
) -> Tuple[int, int]:
    """Contracts a node, creates shortcuts, and optionally updates the shortcut graph.

    Args:
        graph (nx.Graph): The graph to contract the node in.
        node (str): The node to contract.
        update_shortcut_graph (bool): Whether to update the shortcut graph.
        shortcut_graph (nx.Graph): The shortcut graph to update if update_shortcut_graph is True.

    Returns:
        Tuple[int, int]: The edge difference and the number of shortcuts added.
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
                        if update_shortcut_graph:
                            print(f"Shortcut added: {u} --({weight})-- {v}")
                        shortcuts_added += 1
                    else:
                        if update_shortcut_graph and shortcut_graph is not None:
                            print(f"Shortcut updated: {u} --({weight})-- {v}")
                            shortcut_graph.remove_edge(u, v)
                        graph.remove_edge(u, v)
                    graph.add_edge(u, v, weight=weight)
                    if update_shortcut_graph and shortcut_graph is not None:
                        shortcut_graph.add_edge(u, v, weight=weight)

    edges_removed = len(list(graph.edges(node)))  # Edges connected to the node
    graph.remove_node(node)
    return shortcuts_added - edges_removed, shortcuts_added


def create_contraction_hierarchy(graph: nx.Graph) -> Tuple[nx.Graph, List[str], int]:
    """Creates a contraction hierarchy using edge difference ordering.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        Tuple[nx.Graph, List[str], int]: The contraction hierarchy graph, node order, and number of shortcuts added.
    """
    temp_graph1 = graph.copy()

    # Calculate offline edge differences for all nodes
    edge_differences: Dict[str, int] = {}
    nodes = list(
        temp_graph1.nodes()
    )  # Create a list of nodes to avoid modifying the graph during iteration
    for node in nodes:
        edge_differences[node] = contract_node(temp_graph1, node)[0]

    # Order nodes by edge difference (ascending)
    node_order = sorted(edge_differences, key=edge_differences.get)

    # Contract nodes in the calculated order
    temp_graph2 = graph.copy()
    shortcut_graph = graph.copy()
    shortcuts_added = 0
    for node in node_order:
        shortcuts_added += contract_node(
            temp_graph2, node, update_shortcut_graph=True, shortcut_graph=shortcut_graph
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

# 2. Create the contraction hierarchy (using edge difference ordering)
ch_graph, node_order, shortcuts_added = create_contraction_hierarchy(graph)

# 3. Print the number of shortcuts added
print(f"Shortcuts added: {shortcuts_added}")

# 4. Print the node order
print("Node Order:", node_order)

# 5. Find the shortest path
source_node = "A"
target_node = "Y"
shortest_path, path_length = find_shortest_path_nx(ch_graph, source_node, target_node)
print("Shortest Path:", shortest_path)
print("Shortest Path Length:", path_length)
