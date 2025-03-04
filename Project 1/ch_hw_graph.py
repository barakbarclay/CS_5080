import networkx as nx
from contraction_hierarchies import (
    create_contraction_hierarchy,
    find_shortest_path_custom,
)

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
                print(
                    f"Shortest Path from {src} to {tgt}: {shortest_path} with length {path_length}"
                )
