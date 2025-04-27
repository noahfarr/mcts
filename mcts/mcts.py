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
from action_selection import act_randomly, act_uct, act_greedy, batch_act_randomly
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

    return tree.replace(  # type: ignore
        rewards=update(tree.rewards, state.rewards, node),
        values=update(tree.values, state.rewards[player], node),
        visits=update(tree.visits, tree.visits[node] + 1, node),
    )


def backpropagation(tree: Tree, node: jax.Array):

    def cond_fun(carry: tuple) -> bool:
        _, node, _ = carry
        return node != Tree.ROOT_INDEX

    def body_fun(carry: tuple) -> tuple:
        tree, node, value = carry

        value *= -1

        parent = tree.parents[node]
        visit = tree.visits[parent]

        parent_value = tree.values[parent] + (value - tree.values[parent]) / (visit + 1)
        tree = tree.replace(
            values=update(tree.values, parent_value, parent),
            visits=update(tree.visits, visit + 1, parent),
        )
        return (tree, parent, value)

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

        tree, node = jax.lax.cond(
            tree.states.terminated[node],
            lambda _: (tree, node),
            lambda _: (expansion(tree, step, node, action, next_node), next_node),
            operand=None,
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
