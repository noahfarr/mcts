from typing import ClassVar
import jax
import jax.numpy as jnp
import chex
import pgx


@chex.dataclass(frozen=True)
class Tree:

    visits: jax.Array  # [batch_size, num_simulations]
    rewards: jax.Array  # [batch_size, num_simulations, num_players]
    values: jax.Array  # [batch_size, num_simulations]
    parents: jax.Array  # [batch_size, num_simulations]
    actions: jax.Array  # [batch_size, num_simulations]
    children: jax.Array  # [batch_size, num_simulations, num_actions]
    states: pgx.State  # [batch_size, num_simulations, ...]

    ROOT_INDEX: ClassVar[int] = 0
    UNVISITED: ClassVar[int] = -1


def get_state(tree: Tree, node: jax.Array):
    return jax.tree.map(lambda leaf: leaf[node], tree.states)


batch_get_state = jax.jit(jax.vmap(get_state, in_axes=(0, 0)))


def update(array, values, *indices):
    return array.at[indices].set(values)


batch_update = jax.vmap(update)


def instantiate(
    env: pgx.Env,
    state: pgx.State,
    batch_size: int,
    num_simulations: int,
) -> Tree:
    num_nodes = num_simulations + 1
    states = jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], num_nodes, axis=1),
        state,
    )

    tree: Tree = Tree(
        visits=jnp.zeros((batch_size, num_nodes)),  # type: ignore
        rewards=jnp.zeros((batch_size, num_nodes, env.num_players)),  # type: ignore
        values=jnp.zeros((batch_size, num_nodes)),  # type: ignore
        parents=jnp.full((batch_size, num_nodes), Tree.UNVISITED),  # type: ignore
        actions=jnp.full((batch_size, num_nodes), Tree.UNVISITED),  # type: ignore
        children=jnp.full((batch_size, num_nodes, env.num_actions), Tree.UNVISITED),  # type: ignore
        states=states,  # type: ignore
    )

    root = jnp.full([batch_size], Tree.ROOT_INDEX)
    values = jnp.zeros([batch_size])
    rewards = jnp.zeros([batch_size, env.num_players])
    visits = jnp.zeros([batch_size])
    return tree.replace(  # type: ignore
        visits=batch_update(tree.visits, visits, root),
        rewards=batch_update(tree.rewards, rewards, root),
        values=batch_update(tree.values, values, root),
        states=jax.tree.map(
            lambda state, next_state: batch_update(state, next_state, root),
            tree.states,
            state,
        ),
    )
