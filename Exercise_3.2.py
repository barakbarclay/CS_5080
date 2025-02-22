import networkx as nx
from bidirectional_dijkstra import bidirectional_dijkstra

def contract_node_node_order(graph, node):
    """Contracts a node, creates shortcuts, and prints them."""
    
    neighbors = list(graph.neighbors(node))
    edges_added = 0
    shortcut_graph = graph.copy()
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            v = neighbors[j]
            if graph.has_edge(u, node) and graph.has_edge(node, v):
                weight = graph[u][node]['weight'] + graph[node][v]['weight']
                if not graph.has_edge(u, v) or graph[u][v]['weight'] > weight:
                    graph.add_edge(u, v, weight=weight)
                    edges_added += 1  # Count potential new edges
    
    edges_removed = len(list(graph.edges(node)))  # Edges connected to the node
    graph.remove_node(node)
    
    return edges_added - edges_removed, edges_added, edges_removed  # Edge difference

def contract_node_shortcut_graph(graph, shortcut_graph, node):
    """Contracts a node, creates shortcuts, and prints them."""
    
    neighbors = list(graph.neighbors(node))
    shorcuts_added = 0
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            v = neighbors[j]
            if graph.has_edge(u, node) and graph.has_edge(node, v):
                weight = graph[u][node]['weight'] + graph[node][v]['weight']
                if not graph.has_edge(u, v) or graph[u][v]['weight'] > weight:
                    if not graph.has_edge(u, v):
                        print(f"Shortcut added: {u} --({weight})-- {v}")
                        shorcuts_added += 1
                    else:
                        print(f"Shortcut updated: {u} --({weight})-- {v}")
                        graph.remove_edge(u, v)
                        shortcut_graph.remove_edge(u, v)
                    graph.add_edge(u, v, weight=weight)
                    shortcut_graph.add_edge(u, v, weight=weight)
    
    graph.remove_node(node)
    
    return shorcuts_added

def create_contraction_hierarchy(graph):
    """Creates a contraction hierarchy using edge difference ordering."""
    
    temp_graph1 = graph.copy()
    
    # Calculate initial edge differences for all nodes
    edge_differences = {}
    nodes = list(temp_graph1.nodes())  # Create a list of nodes to avoid modifying the graph during iteration
    for node in nodes:
        edge_differences[node] = contract_node_node_order(temp_graph1, node)[0]
    
    # Order nodes by edge difference (ascending)
    node_order = sorted(edge_differences, key=edge_differences.get)

    temp_graph2 = graph.copy()
    shortcut_graph = graph.copy()
    shortcuts_added = 0
    
    print("Node Order:", node_order)
    remaining_node_order = node_order.copy()
    for _ in range(len(node_order) - 1):
        # Contract nodes in the calculated order
        shortcuts_added += contract_node_shortcut_graph(temp_graph2, shortcut_graph, remaining_node_order[0])  # Call contract_node again to modify the graph
        # Recompute edge differences for remaining nodes
        remaining_edge_differences = {}
        for remaining_node in temp_graph2.nodes():
            if remaining_node != node:
                temp_graph3 = temp_graph2.copy()
                remaining_edge_differences[remaining_node] = contract_node_node_order(temp_graph3, remaining_node)[0]
                edge_differences[remaining_node] = remaining_edge_differences[remaining_node]
        remaining_node_order = sorted(remaining_edge_differences, key=remaining_edge_differences.get)
        print("Node Order:", remaining_node_order)
    
    # Reorder nodes by edge difference (ascending)
    node_order = sorted(edge_differences, key=edge_differences.get)

    return nx.compose(shortcut_graph, graph), node_order, shortcuts_added

def find_shortest_path_ch(graph, source, target, node_order):
    """Finds the shortest path and its length using the contraction hierarchy with bidirectional Dijkstra's algorithm."""
    
    # Create a mapping from node to its order
    node_order_map = {node: order for order, node in enumerate(node_order)}
    
    # Use custom bidirectional Dijkstra's algorithm
    path, length = bidirectional_dijkstra(graph, source, target, node_order_map)
    
    return path, length

# 1. Create the graph
graph = nx.Graph()
edges = [
    ('A', 'B', 4),
    ('B', 'C', 2), ('B', 'G', 1),
    ('C', 'D', 1),
    ('D', 'E', 3), ('D', 'I', 1),
    ('E', 'J', 3),
    ('F', 'G', 1),
    ('G', 'H', 2), ('G', 'L', 1),
    ('I', 'J', 1), ('I', 'N', 3),
    ('J', 'O', 3), 
    ('K', 'L', 1), ('K', 'P', 1),
    ('L', 'M', 3),
    ('M', 'N', 3),
    ('N', 'O', 3),
    ('P', 'Q', 1),
    ('Q', 'R', 3), ('Q', 'V', 1),
    ('R', 'S', 3),
    ('S', 'T', 3),
    ('T', 'Y', 3),  
    ('U', 'V', 3),
    ('V', 'W', 2),
    ('W', 'X', 2),
    ('X', 'Y', 2),
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
source_node = 'A'
target_node = 'Y'
shortest_path, path_length = find_shortest_path_ch(ch_graph, source_node, target_node, node_order)
print("Shortest Path:", shortest_path)
print("Shortest Path Length:", path_length)
