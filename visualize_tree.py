import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_tree(tree, max_nodes=50):
    """
    Visualize MCTS tree using pure NetworkX (no Graphviz dependency).
    """

    G = nx.DiGraph()

    children = np.array(tree["children"])
    N = np.array(tree["N"])

    num_nodes = min(max_nodes, children.shape[0])

    # -------------------------
    # Add nodes
    # -------------------------
    for node in range(num_nodes):
        visit_count = int(N[node].sum())
        G.add_node(node, label=f"{node}\nN={visit_count}")

    # -------------------------
    # Add edges
    # -------------------------
    for parent in range(num_nodes):
        for action in range(children.shape[1]):
            child = children[parent, action]
            if 0 <= child < num_nodes:
                G.add_edge(parent, child, label=f"a={action}")

    # -------------------------
    # Layout (pure Python)
    # -------------------------
    pos = nx.spring_layout(G, seed=42)

    # -------------------------
    # Draw
    # -------------------------
    labels = nx.get_node_attributes(G, "label")
    edge_labels = nx.get_edge_attributes(G, "label")

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=2000,
        node_color="lightblue",
        font_size=8,
        arrows=True,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("MCTS Tree Visualization")
    plt.axis("off")
    plt.show()
