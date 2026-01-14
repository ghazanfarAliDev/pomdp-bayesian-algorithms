import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def visualize_tree(tree, max_nodes=50):
    """
    Visualize MCTS tree and print hierarchy in the console,
    and also print total number of nodes actually created by MCTS.
    """
    G = nx.DiGraph()

    children = np.array(tree["children"])
    N = np.array(tree["N"])

    # -------------------------
    # Total nodes actually created
    # -------------------------
    total_nodes_created = int(tree["next_node_id"])
    print(f"\nTotal number of nodes actually created by MCTS: {total_nodes_created}")

    # Limit for visualization if needed
    num_nodes = min(max_nodes, total_nodes_created)

    # -------------------------
    # Add nodes
    # -------------------------
    for node in range(num_nodes):
        visit_count = int(N[node].sum())
        G.add_node(node, label=f"{node}\nN={visit_count}", visits=visit_count)

    # -------------------------
    # Add edges
    # -------------------------
    for parent in range(num_nodes):
        for action in range(children.shape[1]):
            child = children[parent, action]
            if 0 <= child < num_nodes:
                G.add_edge(parent, child, label=f"a={action}")

    # -------------------------
    # Print hierarchy in console
    # -------------------------
    print("\nMCTS Tree Hierarchy (Root -> Leaf):")
    visited = set()
    queue = deque([(0, 0)])  # (node, depth)
    while queue:
        node, depth = queue.popleft()
        if node in visited or node >= num_nodes or node < 0:
            continue
        visited.add(node)
        print("    " * depth + f"- Node {node} (N={int(N[node].sum())})")
        for action in range(children.shape[1]):
            child = int(children[node, action])
            if 0 <= child < num_nodes:
                print("    " * (depth + 1) + f"a={action} -> Node {child}")
                queue.append((child, depth + 2))

    # -------------------------
    # Layout (spring layout)
    # -------------------------
    pos = nx.spring_layout(G, seed=42)

    # -------------------------
    # Draw tree
    # -------------------------
    labels = nx.get_node_attributes(G, "label")
    edge_labels = nx.get_edge_attributes(G, "label")
    sizes = [G.nodes[n]["visits"] * 50 + 300 for n in G.nodes()]
    colors = [G.nodes[n]["visits"] for n in G.nodes()]

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=sizes,
        node_color=colors,
        cmap=plt.cm.Reds,
        font_size=8,
        arrows=True,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("MCTS Tree Visualization")
    plt.axis("off")
    plt.show()


