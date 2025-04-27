import matplotlib.pyplot as plt
import networkx as nx
import jax.numpy as jnp

UNVISITED = -1  # make sure this matches your Tree definition


def hierarchy_pos(
    G,
    root=None,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    xcenter=0.5,
    pos=None,
    parent=None,
):
    """
    If no positions are provided, assigns positions in a top-down hierarchy.
    Source: https://stackoverflow.com/a/29597209
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if not children:
        return pos
    dx = width / len(children)
    nextx = xcenter - width / 2 - dx / 2
    for child in children:
        nextx += dx
        pos = hierarchy_pos(
            G,
            child,
            width=dx,
            vert_gap=vert_gap,
            vert_loc=vert_loc - vert_gap,
            xcenter=nextx,
            pos=pos,
            parent=root,
        )
    return pos


import matplotlib.pyplot as plt
import networkx as nx

UNVISITED = -1


def visualize_tree(tree, batch: int = 0, root: int = 0, figsize=(12, 8)):
    """
    Draws only the reachable part of your MCTS for one batch index.
    """
    G = nx.DiGraph()
    num_nodes = tree.parents.shape[1]
    num_actions = tree.children.shape[2]

    for node in range(num_nodes):
        val = float(tree.values[batch, node])
        visits = int(tree.visits[batch, node])
        G.add_node(node, label=f"{node}\nval={val:.2f}\nvisits={visits}")

    # Add edges for visited children
    for node in range(num_nodes):
        for action in range(num_actions):
            child = int(tree.children[batch, node, action])
            if child != UNVISITED:
                G.add_edge(node, child, action=action)

    # Keep only nodes reachable from the root
    reachable = set(nx.descendants(G, root)) | {root}
    H = G.subgraph(reachable).copy()

    # Compute positions on the pruned graph
    pos = hierarchy_pos(H, root=root)

    plt.figure(figsize=figsize)
    nx.draw(
        H,
        pos,
        labels=nx.get_node_attributes(H, "label"),
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
    )
    edge_labels = {(u, v): d["action"] for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_color="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
