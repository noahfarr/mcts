from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import pgx
import tyro
from tqdm import tqdm

from tree import instantiate
from mcts import batch_search
from action_selection import batch_act_randomly
from visualization import visualize_tree


# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Runtime configuration parsed from CLI."""

    env_id: str = "tic_tac_toe"
    batch_size: int = 64
    num_simulations: int = 4096
    seed: int = 0
    render: bool = False

    agent0: str = "mcts"  # Player 0 behaviour: mcts | random | human
    agent1: str = "random"  # Player 1 behaviour: mcts | random | human

    save_states: bool = False
    save_dir: pathlib.Path = pathlib.Path("./states")


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def select_action_mcts(
    keys: jax.Array,
    env: pgx.Env,
    state: pgx.State,
    num_simulations: int,
    batch_size: int,
):
    """Run batch MCTS and return the action recommendation for the *current* player."""
    tree = instantiate(env, state, batch_size, num_simulations)
    tree, action = batch_search(keys, tree, env.step, num_simulations)
    return tree, action


def select_action_random(
    keys: jax.Array, env: pgx.Env, state: pgx.State, num_simulations, batch_size
):
    return None, batch_act_randomly(keys, state.observation, state.legal_action_mask)


def select_action_human(keys, env, state, num_simulations, batch_size):
    print("Legal actions:", jnp.where(state.legal_action_mask[0])[0])
    try:
        idx = int(input("Enter action index: "))
    except ValueError:
        idx = -1
    if idx < 0 or not bool(state.legal_action_mask[0, idx]):
        raise ValueError("Invalid/illegal move.")
    action = jnp.full((state.observation.shape[0],), idx, dtype=jnp.int32)
    return None, action


AGENT_LOOKUP: Dict[str, Callable] = {
    "mcts": select_action_mcts,
    "random": select_action_random,
    "human": select_action_human,
}


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------


def play_episode(args: Config):
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env = pgx.make(args.env_id)
    batch_init = jax.jit(jax.vmap(env.init))
    batch_step = jax.jit(jax.vmap(env.step))

    key = jax.random.PRNGKey(args.seed)
    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, args.batch_size)

    state = batch_init(keys)

    rewards = state.rewards
    turn = 0

    with tqdm(desc="Running") as pbar:
        while not (state.terminated | state.truncated).all():
            key, sub = jax.random.split(key)
            keys = jax.random.split(sub, args.batch_size)

            tree_A, action_A = AGENT_LOOKUP[args.agent0](
                keys, env, state, args.num_simulations, args.batch_size
            )
            tree_B, action_B = AGENT_LOOKUP[args.agent1](
                keys, env, state, args.num_simulations, args.batch_size
            )

            action = jnp.where(state.current_player == 0, action_A, action_B)

            if args.render:
                if tree_A is not None:
                    visualize_tree(tree_A)
                if tree_B is not None:
                    visualize_tree(tree_B)

            state = batch_step(state, action)
            rewards += state.rewards

            if args.save_states:
                state.save_svg(args.save_dir / f"state_{turn}.svg")

            turn += 1
            pbar.update(1)

    return rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Config)
    rewards = play_episode(args)
    print("Mean return of agent 0:", float(rewards[:, 0].mean()))
    print("Mean return of agent 1:", float(rewards[:, 1].mean()))
