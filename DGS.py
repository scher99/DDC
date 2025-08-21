# duo_gimbal_telemetry.py — compact, CLI-only loader & selector
from __future__ import annotations
import os, pickle, math
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "zero_input",
    "load_telemetry_dataset",
    "duo_gimbal_system",
    "doGimbalSystem",     # CLI selector
    "snippet_plot_all",
]

# ------------------------------ basics
zero_input = lambda: (lambda t: np.zeros_like(t))

def _list_pickles(d: Path, n: int = 5) -> List[Path]:
    ped = sorted(d.glob("PEDTLEM*.pkl"))
    cands = ped or sorted(d.glob("*.pkl"))
    return cands[:n]

def _resolve_file(pth: Union[str, os.PathLike], filename: str | None = None, *, verbose: bool = True) -> Path:
    """
    Resolve a telemetry file path.

    Priority:
    1) If `filename` is an existing file (absolute/relative), use it.
    2) If `pth` is a directory and `filename` exists inside it, use it.
    3) If `pth` is an existing file, use it.
    4) Otherwise, list pickles in `pth` (if dir) and fallback to first.
    """
    # 1) If filename is a direct valid file path, use it
    if filename:
        direct = Path(filename)
        if direct.is_file():
            return direct

    p = Path(pth)

    # 2) If p is a directory and filename exists inside it
    if p.is_dir() and filename:
        cand = p / filename
        if cand.is_file():
            return cand
        elif verbose:
            print(f"Requested file '{filename}' not found under {p}. Falling back to auto-detect.\n")

    # 3) If p is a file, use it
    if p.is_file():
        return p

    # 4) Otherwise, p must be a directory we can scan
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is neither a file nor a directory")

    opts = _list_pickles(p)
    if not opts:
        raise FileNotFoundError(f"No *.pkl files found in {p}")

    if verbose:
        print("\nDetected telemetry logs — first five (sorted):")
        for i, fp in enumerate(opts, 1):
            print(f"  [{i}] {fp.name}")
        print(f"\n→ Defaulting to: {opts[0].name}\n")

    return opts[0]

# ------------------------------ data loader
def load_telemetry_dataset(
    telemetry_path: Union[str, os.PathLike] = "PKL",
    *,
    filename: str | None = None,
    keys: Sequence[str] | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    stride: int = 1,
    add_noise: float | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Callable[[Sequence[float]], np.ndarray]]:
    """Return (t, y, u_fun). If a directory is given, picks the first pickle deterministically,
    unless `filename` points to a valid file."""
    f = _resolve_file(telemetry_path, filename=filename, verbose=verbose)
    if verbose:
        print(f"Opening telemetry file: {f}\n")
    with open(f, "rb") as fh:
        rec: dict = pickle.load(fh)
    if "time" not in rec:
        raise KeyError("'time' key not found in telemetry file")

    if keys is None:
        keys = [k for k in rec if k != "time"]
    else:
        miss = [k for k in keys if k not in rec]
        if miss:
            raise KeyError(f"Requested keys not present: {miss}")

    lens = [len(rec["time"])] + [len(rec[k]) for k in keys]
    m = min(lens)
    if len(set(lens)) > 1 and verbose:
        print("Length mismatch detected:")
        for k, L in zip(["time"] + list(keys), lens):
            print(f"   {k:>20s}: {L}")
        print(f"→ Trimming to {m} samples.\n")

    t_full = np.asarray(rec["time"])[:m]
    data = {k: np.asarray(rec[k])[:m] for k in keys}

    if t_full.min() < 0:
        sh = -t_full.min(); t_full = t_full + sh
        if verbose:
            print(f"Negative times — shifted by +{sh:.6g} s.\n")

    mask = np.ones(m, bool)
    if t_start is not None: mask &= t_full >= t_start
    if t_end   is not None: mask &= t_full <= t_end

    t = t_full[mask][::stride]
    rows = []
    for k in keys:
        sig = data[k][mask][::stride]
        if add_noise is not None:
            sig = sig * (1 + np.random.normal(scale=add_noise, size=sig.size))
        rows.append(sig)
    y = np.vstack(rows)
    return t, y, zero_input()

# ------------------------------ sim-compatible wrapper
def duo_gimbal_system(**kwargs):
    return load_telemetry_dataset(**kwargs)

# ------------------------------ quick plot (supports index labels)
def snippet_plot_all(clean: dict, *, keys_sample: Sequence[str] | None = None,
                     snippet_len: int = 2_000, n_cols: int = 1, sharex: bool = True,
                     file_name: str | os.PathLike | None = None, block: bool = True,
                     index_map: dict[str, int] | None = None):
    """
    Plot first snippet_len samples. If index_map is provided, the y-labels include
    the adjusted selection index like: "[03] LoadAngle".
    """
    if "time" not in clean: raise KeyError("'time' key missing")
    keys = list(clean) if keys_sample is None else [k for k in keys_sample if k in clean]
    n = len(keys); n_cols = max(1, int(n_cols)); n_rows = math.ceil(n / n_cols)
    t = np.asarray(clean["time"]); sn = slice(0, min(snippet_len, t.size))
    fig, axes = plt.subplots(n_rows, n_cols, sharex=sharex, figsize=(12, 2.5*n_rows), squeeze=False)
    flat = axes.flatten()

    def _label_for(k: str) -> str:
        if index_map is not None and k in index_map:
            return f"[{index_map[k]:02d}] {k}"
        return k if k != "time" else "time [s]"

    for i, k in enumerate(keys):
        ax = flat[i]
        if k == "time":
            ax.plot(t[sn], t[sn]); ax.set_ylabel(_label_for(k))
        else:
            ax.plot(t[sn], np.asarray(clean[k])[sn]); ax.set_ylabel(_label_for(k))
        if i // n_cols == n_rows - 1 or not sharex: ax.set_xlabel("time [s]")
    for ax in flat[n:]: ax.set_visible(False)
    pre = f"{Path(file_name).name} – " if file_name else ""
    fig.suptitle(f"{pre}First {sn.stop} samples – {n} channels (incl. time)", y=0.995)
    fig.tight_layout(); plt.show(block=block)
    return fig

# ------------------------------ helpers for CLI selection
def _order_key(name: str):
    nl = name.lower()
    if "el" in nl and "current" in nl: return (0, name)
    if "az" in nl and "current" in nl: return (1, name)
    return (2, name)

def _u_from_signals(t: np.ndarray, a: np.ndarray, b: np.ndarray, n1: str, n2: str):
    t = np.asarray(t, float); a = np.asarray(a, float); b = np.asarray(b, float)
    idx = np.unique(t, return_index=True)[1]; order = np.sort(idx)
    tu, au, bu = t[order], a[order], b[order]
    def u_fun(tq):
        tq = np.asarray(tq)
        v1 = np.interp(tq, tu, au); v2 = np.interp(tq, tu, bu)
        return np.array([float(v1), float(v2)]) if tq.ndim==0 else np.vstack([v1, v2])
    u_fun.__name__ = f"u_from_{n1}_{n2}"; return u_fun

# ------------------------------ CLI selector
def doGimbalSystem(*, telemetry_path: Union[str, os.PathLike] = "data/PKL",
                   filename: str | None = None,
                   snippet_len: int = 25_000, n_cols: int = 2,
                   verbose: bool = True):
    """
    CLI-only interactive selector.
    Returns: t, X, u_fun
    """
    t, y, _ = load_telemetry_dataset(telemetry_path, filename=filename, stride=1, verbose=verbose)
    f = _resolve_file(telemetry_path, filename=filename, verbose=verbose)
    with open(f, "rb") as fh: rec = pickle.load(fh)
    keys_all = [k for k in rec if k != "time"]

    # Build clean dict once
    clean = {"time": t, **{k: row for k, row in zip(keys_all, y)}}

    # Index map so plot labels show adjusted indices matching the printed list
    # [00] time, then [01] keys_all[0], [02] keys_all[1], ...
    index_map = {"time": 0}
    index_map.update({k: i+1 for i, k in enumerate(keys_all)})

    # Show a preview with indexed labels before prompting for indices
    fig_prev = snippet_plot_all(clean, snippet_len=snippet_len, n_cols=n_cols, sharex=True,
                                file_name=Path(f).name, block=False, index_map=index_map)
    try:
        out_path = Path.cwd() / f"{Path(f).stem}_preview.png"
        fig_prev.savefig(out_path, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"Saved preview image to: {out_path}")
    except Exception:
        pass

    # Now list channels and prompt
    for i, k in enumerate(["time"]+keys_all):
        print(f"[{i:02d}] {k}")
    sel_feat = input("Indices for X (comma-separated, exclude 'time'): ").strip()
    idx_feat = [int(s) for s in sel_feat.split(',') if s.strip().isdigit()] if sel_feat else []
    chosen_x = [keys_all[i-1] for i in idx_feat if i>0 and "current" not in keys_all[i-1].lower()]
    sel_curr = input("Two indices for CURRENT channels (comma-separated): ").strip()
    idx_curr = [int(s) for s in sel_curr.split(',') if s.strip().isdigit()][:2] if sel_curr else []
    chosen_u = [keys_all[i-1] for i in idx_curr if i>0]
    if len(chosen_u)<2:
        for k in [k for k in keys_all if "current" in k.lower()]:
            if k not in chosen_u: chosen_u.append(k)
            if len(chosen_u)==2: break

    # Assemble outputs
    X = np.vstack([clean[k] for k in chosen_x]) if chosen_x else np.zeros((0, t.size))
    if len(chosen_u)>=2:
        chosen_u = sorted(chosen_u, key=_order_key)[:2]
        u_fun = _u_from_signals(t, clean[chosen_u[0]], clean[chosen_u[1]], chosen_u[0], chosen_u[1])
    elif len(chosen_u)==1:
        u_fun = _u_from_signals(t, clean[chosen_u[0]], np.zeros_like(t), chosen_u[0], "zeros")
    else:
        u_fun = _u_from_signals(t, np.zeros_like(t), np.zeros_like(t), "zeros", "zeros")

    # Keep the preview window around until user closes it (optional)
    try:
        plt.show(block=False)
    except Exception:
        pass

    return t, X, u_fun

# ------------------------------ run directly
if __name__ == "__main__":
    root = "data/PKL"
    # Optionally pass filename="PedTelem_20230629011211.pkl"
    t, X, u_fun = doGimbalSystem(telemetry_path=root,
                                 filename=None,
                                 snippet_len=25_000, n_cols=2, verbose=True)
    print("\n—— Duo-Gimbal telemetry selection complete ——")
    print(f"Samples: {len(t)}  |  X shape (rows x N): {X.shape}")
    try:
        print("U(t[:5]) sample (2 x 5):\n", u_fun(t[:5]))
    except Exception as e:
        print("Could not evaluate u_fun on sample times:", e)
