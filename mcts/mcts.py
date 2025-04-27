from typing import Callable
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import chex
import pgx

from tree import (
    Tree,
    update,
    instantiate,
    get_state,
)
from action_selection import (
    act_randomly,
    act_uct,
    act_greedy,
    batch_act_randomly,
)
from visualization import visualize_tree


def selection(tree: Tree, action_selection_fun: Callable):

    def cond_fun(carry: tuple) -> jax.Array:
        _, _, next_node = carry
        return next_node != Tree.UNVISITED

    def body_fun(carry: tuple) -> tuple:
        node, _, next_node = carry
        node: jax.Array = next_node
        action: jax.Array = action_selection_fun(tree, node)
        next_node: jax.Array = tree.children[node, action]
        return (node, action, next_node)

    node = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
    action = action_selection_fun(tree, node)
    next_node = tree.children[node, action]
    node, action, next_node = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (node, action, next_node),
    )
    return node, action, next_node


def expansion(
    tree: Tree,
    step: Callable,
    node: jax.Array,
    action: jax.Array,
    next_node: jax.Array,
) -> Tree:
    state: pgx.State = get_state(tree, node)
    next_state: pgx.State = step(state, action)
    return tree.replace(  # type: ignore
        parents=update(tree.parents, node, next_node),
        actions=update(tree.actions, action, next_node),
        children=update(tree.children, next_node, node, action),
        states=jax.tree.map(
            lambda state, next_state: update(state, next_state, next_node),
            tree.states,
            next_state,
        ),
    )


def simulation(
    key: chex.PRNGKey,
    step: Callable,
    tree: Tree,
    node: jax.Array,
    # action_selection_fun: Callable,
):

    state = get_state(tree, node)
    player = 1 - state.current_player

    def cond_fun(carry: tuple):
        _, state = carry
        return jnp.logical_not(state.terminated)

    def body_fun(carry: tuple):
        key, state = carry
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state.observation, state.legal_action_mask)
        state: pgx.State = step(state, action)
        return (key, state)

    key, state = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (key, state),
    )

    visits = tree.visits[node] + 1
    return tree.replace(  # type: ignore
        rewards=update(tree.rewards, state.rewards, node),
        values=update(tree.values, state.rewards[player], node),
        visits=update(tree.visits, visits, node),
    )


def backpropagation(tree: Tree, node: jax.Array):

    def cond_fun(carry: tuple) -> bool:
        _, node, _ = carry
        return node != Tree.ROOT_INDEX

    def body_fun(carry: tuple) -> tuple:
        tree, node, reward = carry

        reward *= -1

        parent = tree.parents[node]
        visit = tree.visits[parent]

        value = tree.values[parent] + (reward - tree.values[parent]) / (visit + 1)
        tree = tree.replace(
            values=update(tree.values, value, parent),
            visits=update(tree.visits, visit + 1, parent),
        )
        return (tree, parent, reward)

    tree, *_ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (
            tree,
            node,
            tree.values[node],
        ),
    )
    return tree


def search(
    key: chex.PRNGKey,
    tree: Tree,
    step: Callable,
    num_simulations: int,
):

    def body_fun(i: int, carry: tuple) -> tuple:
        key, tree = carry
        key, subkey = jax.random.split(key)

        node, action, next_node = selection(tree, act_uct)

        next_node = jax.lax.select(next_node == Tree.UNVISITED, i + 1, next_node)

        tree = jax.lax.cond(
            tree.states.terminated[node],
            lambda tree: tree,
            lambda tree: expansion(tree, step, node, action, next_node),
            tree,
        )
        node = jax.lax.select(
            tree.states.terminated[node],
            node,
            next_node,
        )
        tree = simulation(subkey, step, tree, node)
        tree = backpropagation(tree, node)
        return (key, tree)

    _, tree = jax.lax.fori_loop(0, num_simulations, body_fun, (key, tree))
    action = act_greedy(tree, jnp.array(Tree.ROOT_INDEX))
    return tree, action


batch_search = jax.jit(
    jax.vmap(search, in_axes=(0, 0, None, None), out_axes=(0, 0)),
    static_argnums=(2, 3),
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="tic_tac_toe")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_simulations", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    i = 0
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, args.batch_size)

    env = pgx.make(args.env_id)
    batch_init = jax.jit(jax.vmap(env.init))
    batch_step = jax.jit(jax.vmap(env.step))
    state = batch_init(keys)

    print("Current player:", state.current_player)

    rewards = state.rewards
    while not (state.terminated | state.truncated).all():
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, args.batch_size)
        tree = instantiate(env, state, args.batch_size, args.num_simulations)
        tree, action_A = batch_search(
            keys,
            tree,
            env.step,
            args.num_simulations,
        )
        if args.render:
            visualize_tree(tree, 0, Tree.ROOT_INDEX)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, args.batch_size)
        # action_B = batch_act_randomly(keys, state.observation, state.legal_action_mask)
        # action_B = batch_search(keys, tree, env.step, args.num_simulations)
        # action = jnp.where(state.current_player == 0, action_A, action_B)
        action = action_A

        state = batch_step(state, action)
        state.save_svg(f"iteration_{i}.svg")
        rewards += state.rewards
        i += 1

    print(f"Return of agent A = {rewards[:, 0]}")
    print(f"Mean return of agent A = {rewards[:, 0].mean()}")


if __name__ == "__main__":
    main()
