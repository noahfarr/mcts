from typing import NamedTuple, Callable
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import chex
import pgx

ROOT_INDEX: int = 0
UNVISITED: int = -1


def main():
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="tic_tac_toe")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=9)
    args = parser.parse_args()
    search(args.env_id, args.batch_size, args.num_simulations, args.max_depth)


@chex.dataclass(frozen=False)
class Tree:

    visits: chex.Array  # [batch_size, num_simulations]
    values: chex.Array  # [batch_size, num_simulations]
    parents: chex.Array  # [batch_size, num_simulations]
    parent_actions: chex.Array  # [batch_size, num_simulations]
    children: chex.Array  # [batch_size, num_simulations, num_actions]
    children_visits: chex.Array  # [batch_size, num_simulations, num_actions]
    children_rewards: chex.Array  # [batch_size, num_simulations, num_actions]
    children_discounts: chex.Array  # [batch_size, num_simulations, num_actions]
    children_values: chex.Array  # [batch_size, num_simulations, num_actions]
    states: pgx.State  # [batch_size, num_simulations, ...]


def get_state(tree: Tree, node: chex.Array):
    return jax.tree.map(lambda leaf: leaf[node], tree.states)


batch_get_state = jax.jit(jax.vmap(get_state, in_axes=(0, 0)))


def update(array, values, *indices):
    return array.at[indices].set(values)


batch_update = jax.vmap(update)


def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)


def _act_randomly(key, tree: Tree, node: chex.Array):
    """Ignore observation and choose randomly from legal actions"""
    state = get_state(tree, node)
    mask = (tree.children[node] == UNVISITED) & state.legal_action_mask  # type: ignore

    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)


def act_uct(key, tree: Tree, node: chex.Array, c: float = jnp.sqrt(2.0)):
    exploit = tree.children_values[node] / (tree.children_visits[node] + 1e-8)

    explore = jnp.sqrt(jnp.log(tree.visits[node] / (tree.children_visits[node] + 1e-8)))

    uct = exploit + c * explore

    legal_action_mask = tree.children[node] != UNVISITED
    uct = jnp.where(legal_action_mask, uct, -jnp.inf)
    return jnp.argmax(uct)


act_uct = jax.jit(act_uct, static_argnums=(3,))


def instantiate(
    env: pgx.Env,
    state: pgx.State,
    batch_size: int,
    num_simulations: int,
) -> Tree:
    states = jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], num_simulations, axis=1),
        state,
    )

    num_nodes = num_simulations + 1
    tree = Tree(
        visits=jnp.zeros((batch_size, num_nodes)),  # type: ignore
        values=jnp.zeros((batch_size, num_nodes)),  # type: ignore
        parents=jnp.full((batch_size, num_nodes), UNVISITED),  # type: ignore
        parent_actions=jnp.full((batch_size, num_nodes), UNVISITED),  # type: ignore
        children=jnp.full(  # type: ignore
            (batch_size, num_nodes, env.num_actions), UNVISITED
        ),
        children_visits=jnp.zeros((batch_size, num_nodes, env.num_actions)),  # type: ignore
        children_rewards=jnp.zeros((batch_size, num_nodes, env.num_actions)),  # type: ignore
        children_discounts=jnp.zeros((batch_size, num_nodes, env.num_actions)),  # type: ignore
        children_values=jnp.zeros((batch_size, num_nodes, env.num_actions)),  # type: ignore
        states=states,  # TODO: Currently put root state everywhere, reevaluate this
    )
    jax.debug.print("Initializing tree")

    root = jnp.full([batch_size], ROOT_INDEX)
    tree = update_node(
        tree,
        batch_size,
        root,
        jnp.ones(batch_size),
        state,
    )
    return tree


def update_node(
    tree: Tree,
    batch_size: int,
    node: chex.Array,
    value: chex.Array,
    state: pgx.State,
) -> Tree:
    batch_range = jnp.arange(batch_size)
    visits = tree.visits[batch_range, node] + 1  # type: ignore

    tree = tree.replace(  # type: ignore
        values=batch_update(tree.values, value, node),
        visits=batch_update(tree.visits, visits, node),
        states=jax.tree.map(
            lambda state, next_state: batch_update(state, next_state, node),
            tree.states,
            state,
        ),
    )
    return tree


class SelectionState(NamedTuple):
    key: chex.PRNGKey
    node: chex.Array
    action: chex.Array
    next_node: chex.Array
    depth: chex.Array
    terminal: chex.Array


def selection(key: chex.PRNGKey, tree: Tree, max_depth: int) -> chex.Array:
    jax.debug.print("Selecting node")

    def select(state: SelectionState) -> SelectionState:
        key, subkey = jax.random.split(state.key)
        node = state.next_node
        action = jax.lax.cond(
            jnp.any(tree.children[node] == UNVISITED),
            _act_randomly,
            act_uct,
            key,
            tree,
            node,
        )
        next_node = tree.children[node, action]  # type: ignore
        terminal = jnp.logical_or(next_node == UNVISITED, state.depth >= max_depth)
        return SelectionState(
            key=key,
            node=node,
            action=action,
            next_node=next_node,
            depth=state.depth + 1,
            terminal=terminal,
        )

    state = SelectionState(
        key=key,
        node=UNVISITED,  # type: ignore
        action=UNVISITED,  # type: ignore
        next_node=jnp.array(ROOT_INDEX, dtype=jnp.int32),
        depth=jnp.zeros((), jnp.int32),
        terminal=jnp.array(False),
    )
    state = jax.lax.while_loop(
        lambda state: jnp.logical_not(state.terminal), select, state
    )
    return state.node, state.action


def expansion(
    key: chex.PRNGKey,
    tree: Tree,
    step: Callable,
    node: chex.Array,
    action: chex.Array,
    next_node: chex.Array,
) -> chex.Array:
    jax.debug.print("Expanding node")
    batch_size = tree.parents.shape[0]
    key, _ = jax.random.split(key)
    state = batch_get_state(tree, node)
    next_state = step(state, action)
    tree = update_node(
        tree,
        batch_size,
        next_node,
        jnp.zeros((batch_size)),
        next_state,
    )
    return tree.replace(
        parents=batch_update(tree.parents, node, next_node),
        parent_actions=batch_update(tree.parent_actions, action, next_node),
        children=batch_update(tree.children, next_node, node, action),
        children_rewards=batch_update(
            tree.children_rewards, state.rewards[:, 0], node, action
        ),
        children_discounts=batch_update(
            tree.children_discounts, jnp.ones(batch_size), node, action
        ),
    )


class SimulationState(NamedTuple):
    key: chex.PRNGKey
    state: pgx.State


def simulation(key: chex.PRNGKey, init_state: pgx.State, step: Callable):
    jax.debug.print("Simulating")

    def simulate(state: SimulationState) -> SimulationState:
        key, _ = jax.random.split(state.key)
        action = act_randomly(
            key, state.state.observation, state.state.legal_action_mask
        )
        state: pgx.State = step(state.state, action)
        return SimulationState(
            key=key,
            state=state,
        )

    state = SimulationState(
        key=key,
        state=init_state,
    )
    state = jax.lax.while_loop(
        lambda state: jnp.logical_not(state.state.terminated),
        simulate,
        state,
    )
    return state.state


class BackpropagationState(NamedTuple):
    tree: Tree
    node: chex.Array
    value: chex.Array


def backpropagation(tree: Tree, node: chex.Array, rewards: chex.Array):
    jax.debug.print("Backpropagating")

    def backpropagate(state: BackpropagationState) -> BackpropagationState:
        parent = state.tree.parents[state.node]
        visits = state.tree.visits[parent]
        action = state.tree.parent_actions[state.node]
        reward = state.tree.children_rewards[state.node, action]
        value = reward + state.tree.children_discounts[parent, action] * state.value
        parent_value = (state.tree.values[parent] * visits + value) / (visits + 1)

        tree = state.tree.replace(
            values=update(state.tree.values, parent_value, parent),
            visits=update(state.tree.visits, visits + 1, parent),
            children_values=update(
                state.tree.children_values,
                value,
                parent,
                action,
            ),
            children_visits=update(
                state.tree.children_visits,
                state.tree.children_visits[parent, action] + 1,
                parent,
                action,
            ),
        )
        return BackpropagationState(
            tree=tree,
            node=parent,
            value=value,
        )

    state = BackpropagationState(tree, node, rewards)
    tree, _, _ = jax.lax.while_loop(
        lambda state: state.node != ROOT_INDEX, backpropagate, state
    )
    return tree


class SearchState(NamedTuple):
    key: chex.PRNGKey
    tree: Tree


def search(
    env_id: str,
    batch_size: int,
    num_simulations: int,
    max_depth: int,
):
    env = pgx.make(env_id)
    batch_init = jax.jit(jax.vmap(env.init))
    batch_step = jax.jit(jax.vmap(env.step))
    batch_selection = jax.jit(jax.vmap(selection, in_axes=(0, 0, None), out_axes=0))
    batch_simulation = jax.jit(
        jax.vmap(simulation, in_axes=(0, 0, None)),
        static_argnums=(2,),
    )
    batch_backpropagation = jax.jit(
        jax.vmap(backpropagation, in_axes=(0, 0, 0)),
    )
    batch_range = jnp.arange(batch_size)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)

    state = batch_init(keys)
    tree = instantiate(env, state, batch_size, num_simulations)

    def step(i: int, state: SearchState) -> SearchState:
        jax.debug.print("⟳ iteration = {}", i)
        key, subkey = jax.random.split(state.key)
        keys = jax.random.split(subkey, batch_size)

        nodes, actions = batch_selection(keys, state.tree, max_depth)

        next_nodes = state.tree.children[batch_range, nodes, actions]
        next_nodes = jnp.where(next_nodes == UNVISITED, i + 1, next_nodes)
        key, subkey = jax.random.split(key)
        tree: Tree = expansion(
            subkey, state.tree, batch_step, nodes, actions, next_nodes
        )
        states = batch_get_state(tree, next_nodes)
        state = batch_simulation(keys, states, env.step)
        tree = batch_backpropagation(tree, next_nodes, state.rewards[:, 0])
        return SearchState(key, tree)

    state = SearchState(key, tree)
    _, tree = jax.lax.fori_loop(0, num_simulations, step, state)
    visualize_tree(tree)


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

    # Add all nodes (we’ll prune unreachable later)
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


if __name__ == "__main__":
    main()
