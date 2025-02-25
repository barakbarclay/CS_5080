import unittest
from bidirectional_dijkstra import bidirectional_dijkstra

# This code was written with assistance from Gemini and GitHub Copilot


class TestBidirectionalDijkstra(unittest.TestCase):

    def setUp(self):
        self.graph = {
            "A": {"B": {"weight": 1}, "C": {"weight": 4}},
            "B": {"A": {"weight": 1}, "C": {"weight": 2}, "D": {"weight": 5}},
            "C": {"A": {"weight": 4}, "B": {"weight": 2}, "D": {"weight": 1}},
            "D": {"B": {"weight": 5}, "C": {"weight": 1}},
        }
        self.node_order_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def test_shortest_path(self):
        path, length = bidirectional_dijkstra(self.graph, "A", "D", self.node_order_map)
        self.assertEqual(path, ["A", "B", "C", "D"])
        self.assertEqual(length, 4)

    def test_no_path(self):
        graph = {
            "A": {"B": {"weight": 1}},
            "B": {"A": {"weight": 1}},
            "C": {"D": {"weight": 1}},
            "D": {"C": {"weight": 1}},
        }
        node_order_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        path, length = bidirectional_dijkstra(graph, "A", "D", node_order_map)
        self.assertIsNone(path)
        self.assertEqual(length, float("inf"))

    def test_same_source_target(self):
        path, length = bidirectional_dijkstra(self.graph, "A", "A", self.node_order_map)
        self.assertEqual(path, ["A"])
        self.assertEqual(length, 0)


if __name__ == "__main__":
    unittest.main()
