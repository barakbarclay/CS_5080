import unittest
import networkx as nx
from contraction_hierarchies import contract_node, create_contraction_hierarchy, find_shortest_path_nx, find_shortest_path_custom

# This code was written with assistance from Gemini and GitHub Copilot

class TestContractNode(unittest.TestCase):

    def setUp(self):
        self.graph = nx.Graph()
        edges = [
            ("A", "B", 4),
            ("B", "A", 4),
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
            self.graph.add_edge(u, v, weight=weight)

    def test_contract_node_no_neighbors(self):
        graph = nx.Graph()
        graph.add_node("Z")
        edge_diff, shortcuts_added = contract_node(graph, "Z")
        self.assertEqual(shortcuts_added, 0)
        self.assertEqual(edge_diff, 0)  # 0 shortcuts added, 0 edges removed

    def test_contract_node_with_neighbors(self):
        edge_diff, shortcuts_added = contract_node(self.graph, "B")
        self.assertEqual(shortcuts_added, 3)
        self.assertEqual(edge_diff, 0)  # 3 shortcut added, 3 edges removed

    def test_contract_node_update_shortcut_graph(self):
        shortcut_graph = self.graph.copy()
        edge_diff, shortcuts_added = contract_node(self.graph, "B", update_graph=True, shortcut_graph=shortcut_graph)
        self.assertEqual(shortcuts_added, 3)
        self.assertEqual(edge_diff, 0)
        self.assertTrue(shortcut_graph.has_edge("A", "C"))
        self.assertEqual(shortcut_graph["A"]["C"]["weight"], 6)

    def test_contract_node_with_no_shortcuts(self):
        edge_diff, shortcuts_added = contract_node(self.graph, "F")
        self.assertEqual(shortcuts_added, 0)
        self.assertEqual(edge_diff, -1)  # 0 shortcuts added, 1 edge removed

    def test_create_contraction_hierarchy(self):
        ch_graph, node_order, shortcuts_added = create_contraction_hierarchy(self.graph)
        self.assertIsInstance(ch_graph, nx.Graph)
        self.assertIsInstance(node_order, list)
        self.assertIsInstance(shortcuts_added, int)

    def test_find_shortest_path_nx(self):
        ch_graph, node_order, shortcuts_added = create_contraction_hierarchy(self.graph)
        path, length = find_shortest_path_nx(ch_graph, "A", "Y")
        self.assertIsInstance(path, list)
        self.assertIsInstance(length, int)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "Y")

    def test_find_shortest_path_custom(self):
        ch_graph, node_order, shortcuts_added = create_contraction_hierarchy(self.graph)
        path, length = find_shortest_path_custom(ch_graph, "A", "Y", node_order)
        self.assertIsInstance(path, list)
        self.assertIsInstance(length, int)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "Y")

if __name__ == '__main__':
    unittest.main()