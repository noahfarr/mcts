# Monte Carlo Tree Search (MCTS) ― JAX implementation

A **fully‑vectorised, differentiable‑friendly** implementation of Monte Carlo Tree Search written with [JAX](https://github.com/google/jax) and [pgx](https://github.com/kurowasan/pgx).

---

## ✨ Key features

| Feature | Description |
|---------|-------------|
| **Pure JAX** | No Python loops inside the algorithm; selection, expansion, simulation and back‑propagation are compiled with `jax.jit` and naturally support CPU, GPU & TPU back‑ends. |
| **Batch search** | `batch_search` wraps `search` with `jax.vmap`, enabling thousands of simultaneous tree searches for large‑scale self‑play or policy evaluation. |
| **Plug‑and‑play policies** | `action_selection.py` provides `act_randomly`, `act_uct`, `act_greedy`, but you can inject any custom policy function. |
| **Environment‑agnostic** | Built on **pgx**, so any OpenAI‑Gym‑compatible environment that pgx supports will work out‑of‑the‑box. |
| **Visualization** | `visualization.py` renders the search tree at any point for debugging or demos. |

---

## 🗂 Repository layout

```
.
├── action_selection.py    # Policies: random, UCT, greedy, batch helpers
├── example.py             # Minimal runnable example
├── mcts.py                # Core MCTS implementation (selection/expansion/…)
├── tree.py                # Immutable tree datastructure utilities
├── visualization.py       # ASCII / graphviz tree printer
├── states/                # (Optional) environment‑specific helper modules
└── __init__.py            # Library entry‑point
```

---

## 🚀 Quick start

### 1. Install
Ensure you have **Python ≥ 3.9**. Then run:

```bash
pip install -r requirements.txt
```

### 2. Run the example

```bash
python example.py --env tic_tac_toe --num_sim 1028 --render
```
This will open a simple Tic‑Tac‑Toe match where the agent selects moves using MCTS and plays vs. a random opponent.

---

## 🧩 API sketch

```python
from mcts import search, batch_search
import jax, pgx

# Initialise environment & tree
state = pgx.make("tic_tac_toe")
key   = jax.random.PRNGKey(0)

# Instantiate an empty tree with one root node
tree  = instantiate(state)  # from tree.py

# Single‑search (e.g. per time‑step)
num_sim = 200  # simulations per move
key, subkey = jax.random.split(key)
tree, action = search(subkey, tree, state.step, num_sim)

# Vectorised search (e.g. for self‑play)
keys   = jax.random.split(key, batch_size)
trees  = jax.vmap(instantiate)(states)
new_trees, actions = batch_search(keys, trees, state.step, num_sim)
```

Every helper is **function‑pure**; the only mutable state is returned explicitly, making it trivial to checkpoint or feed into JAX transformations (`grad`, `pmap`, etc.).

---

## ⚙️ Configuration flags

| Flag | File | Purpose |
|------|------|---------|
| `--env` | *example.py* | Environment string passed to `pgx.make`. |
| `--num_sim` | *example.py* | Number of simulations per search call. |
| `--render` | *example.py* | Visualise the tree and board after each move. |

---

## 🏗️ Design overview

The algorithm follows the canonical four phases:

1. **Selection** – `selection()` walks the current tree using a supplied `action_selection_fun` (default: UCT).
2. **Expansion** – `expansion()` lazily adds one new leaf for the first unexplored action.
3. **Simulation** – `simulation()` rolls out with `act_randomly` until a terminal state.
4. **Back‑propagation** – `backpropagation()` updates in‑place‑immutable arrays (`jax.numpy`) for value & visit‑counts.

Everything is kept static‑shape so XLA can compile a single fused kernel.

---

## 📈 Performance tips

* Prefer **GPU/TPU** for large trees or batched searches.
* Try alternative playout policies (e.g. a learned network) by changing `simulation()`.
* Tune exploration constant in `act_uct` to balance exploitation vs exploration.

---

## 🤝 Contributing

Bug reports, feature requests and pull‑requests are welcome! Please open an issue first so we can discuss your proposal.

### Dev setup
```bash
pip install -e .[dev]       # installs pytest, black, isort, etc.
pytest -q                   # run unit tests
```

---

## 📝 License

This project is licensed under the **MIT** License – see the [LICENSE](LICENSE) file for details.

