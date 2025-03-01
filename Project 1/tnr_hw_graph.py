import osmnx as ox
import networkx as nx
import random
from tnr import CHNode, ContractionHierarchyTNR

# Original code from Kaylee

# Small sample plot: Falcon

def main():
    multiDiGraph, undirected_graph = create_graph()
    orig = "A"
    dest = "Y"

    # Create and preprocess CH-TNR
    ch_tnr = ContractionHierarchyTNR(multiDiGraph, undirected_graph)
    ch_tnr.preprocess(cell_size=0.01)  # Adjust cell size based on your graph scale

    # Query
    ch_tnr_distance = ch_tnr.query(orig, dest)
    print(f"CH-TNR distance from {orig} to {dest}: {ch_tnr_distance}")


def create_graph():
  undirected_graph = nx.Graph()
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
      undirected_graph.add_edge(u, v, weight=weight)

  multiDiGraph = nx.MultiDiGraph(undirected_graph)

  # Assign travel time and length using weight (handling missing values)
  for u, v, data in multiDiGraph.edges(data=True):
    weight = data["weight"]
    data["length"] = weight
    data["travel_time"] = weight * 50.0


  return multiDiGraph, undirected_graph


if __name__ == '__main__':
    main()