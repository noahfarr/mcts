from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import jax
import numpy as np
from graphviz import Digraph

from tree import Tree, get_state  # noqa: F401 – imported for possible extensions

__all__ = ["visualize_tree"]


def visualize_tree(
    tree: Tree,
    batch: int = 0,
    node: int = Tree.ROOT_INDEX,
    *,
    max_depth: Optional[int] = None,
    show_unvisited: bool = False,
    filename: Optional[Union[str, Path]] = None,
    fmt: str = "svg",
    inline: bool = False,
    view: bool = True,
):
    """Render an MCTS *Tree* to SVG/PNG/PDF using **Graphviz**.

    Parameters
    ----------
    tree : Tree
        The batched MCTS tree.
    batch : int, default 0
        Which batch element to visualise.
    node : int, default ``Tree.ROOT_INDEX``
        Index of the root node for the rendered subtree.
    max_depth : int | None, optional
        If given, stop the recursion after *max_depth* layers.
    show_unvisited : bool, default False
        Visualise *UNVISITED* children as grey diamonds.
    filename : str | Path | None, optional
        Where to save the figure.  The suffix will be replaced by *fmt*.
        Ignored if *inline* is ``True``.
    fmt : str, default "svg"
        Any output format supported by Graphviz (``"png"``, ``"pdf"``…).
    inline : bool, default False
        Return SVG/PNG bytes and, if running inside IPython, display them
        immediately.  Disables writing to disk.
    view : bool, default False
        Ask Graphviz to open the rendered file with the system viewer.
        Ignored when *inline* is ``True``.

    Returns
    -------
    pathlib.Path | bytes
        *Path* to the generated file (disk output) **or** SVG/PNG *bytes*
        (inline mode).
    """

    # ─────────────────── strip the batch dimension ────────────────────────
    t = jax.tree.map(lambda x: np.asarray(x[batch]), tree)  # type: ignore[arg-type]

    dot = Digraph(
        "MCTS",
        node_attr={
            "shape": "circle",
            "fontname": "Helvetica",
            "fontsize": "10",
            "style": "filled",
            "fillcolor": "white",
        },
    )
    dot.attr(rankdir="TB")  # top → bottom

    n_children = t.children.shape[1]

    # ────────────────────────── colour helpers ────────────────────────────
    _VISIT_CMAP = np.array(
        [
            "#ffffff",  # 0
            "#e6f2ff",  # 1–1
            "#cce5ff",  # 2–3
            "#b3d8ff",  # 4–7
            "#99ccff",  # 8–15
            "#80bfff",  # 16+
        ]
    )

    def _visit_color(n_visits: int) -> str:
        """Map visit count → colour bucket (log2)."""
        bucket = min(len(_VISIT_CMAP) - 1, int(np.log2(n_visits + 1)))
        return _VISIT_CMAP[bucket]

    _TERMINAL_COLOR = "#c8e6c9"  # light‑green fill for terminal states
    _PLAYER_BORDER = {0: "#e74c3c", 1: "#2471a3"}  # red / blue borders

    # ─────────────────────── depth‑first recursion ────────────────────────
    def _add(cur: int, depth: int) -> None:
        visits = int(t.visits[cur])
        value = float(t.values[cur])
        terminated = bool(t.states.terminated[cur])
        player = int(t.states.current_player[cur])

        dot.node(
            str(cur),
            label=f"{cur}\\nN={visits}\\nV={value:+.3f}\\nP={player}",
            fillcolor=_TERMINAL_COLOR if terminated else _visit_color(visits),
            color=_PLAYER_BORDER[player],
            penwidth="1.6",
        )

        if max_depth is not None and depth >= max_depth:
            return

        for action in range(n_children):
            child = int(t.children[cur, action])
            edge_label = f"a={action}"

            if child == Tree.UNVISITED and not show_unvisited:
                continue

            if child == Tree.UNVISITED:
                # grey diamond placeholder
                ghost = f"u{cur}_{action}"
                dot.node(
                    ghost,
                    shape="diamond",
                    label="?",
                    style="dotted",
                    fillcolor="#eeeeee",
                )
                dot.edge(str(cur), ghost, label=edge_label, style="dotted")
            else:
                dot.edge(str(cur), str(child), label=edge_label)
                _add(child, depth + 1)

    _add(int(node), 0)

    # ────────────────────────── output helpers ────────────────────────────
    def _emit_inline() -> bytes:
        data = dot.pipe(format=fmt)
        try:
            # if in IPython, display inline
            from IPython.display import display, SVG  # type: ignore

            if fmt == "svg":
                display(SVG(data))
        except Exception:
            pass  # silent fallback for plain Python REPL
        return data

    def _emit_file() -> Path:
        out_path = (
            Path(f"tree_batch{batch}_node{node}.{fmt}")
            if filename is None
            else Path(filename).with_suffix(f".{fmt}")
        )
        # Use stem to let Graphviz append extension
        dot.render(out_path.with_suffix(""), format=fmt, cleanup=True, view=view)
        return out_path

    # ─────────────────────────── choose mode ──────────────────────────────
    if inline:
        return _emit_inline()

    return _emit_file()
