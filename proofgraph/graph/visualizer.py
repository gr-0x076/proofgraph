from pyvis.network import Network
import networkx as nx
import os

class ProofGraphVisualizer:
    """
    Renders an interactive HTML knowledge graph from a ProofGraph NetworkX object.
    Includes physics layout, search bar, zoom controls, and custom tooltips.
    """
    def __init__(self, height: str = "750px", width: str = "100%"):
        self.height = height
        self.width = width

    def export_html(self, graph: nx.DiGraph, output_filepath: str, title: str = "ProofGraph: Evidence Intelligence Explorer") -> str:
        """Exports interactive PyVis visualization to a standalone HTML file."""
        net = Network(height=self.height, width=self.width, directed=True, bgcolor="#0f172a", font_color="#f8fafc")
        
        # Load from NetworkX
        net.from_nx(graph)

        # Configure physics options for a clean, stable layout
        net.set_options("""
        {
          "nodes": {
            "font": {
              "face": "system-ui, sans-serif",
              "size": 13,
              "color": "#f8fafc"
            }
          },
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal",
              "roundness": 0.4
            },
            "font": {
              "size": 11,
              "align": "top"
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        net.save_graph(output_filepath)
        return output_filepath
