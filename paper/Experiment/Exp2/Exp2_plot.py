import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List

def plot_metrics_one_row_with_box_auc(
    methods_cfg: Dict[str, Dict],
    figsize: Tuple[int, int]=(9, 2.2), 
    dpi: int=120,
    band: str="se",              
    ci_level: float=0.95,
    jitter: float=0.08,
    point_size: int=22,
    showfliers: bool=False,
    box_alpha: float=0.35,
    colors: Optional[List[str]]=None,
    save_path: Optional[str]=None,
    title: str=None,
    n_boot: int = 1000,
    random_state: Optional[int] = None,
):

    def _bootstrap_ci_mean(vals: np.ndarray, level: float, n_boot: int, rng: np.random.Generator):
        x = np.asarray(vals, dtype=float)
        x = x[~np.isnan(x)]
        m = len(x)
        if m == 0:
            return np.nan, np.nan, np.nan
        mu = float(np.mean(x))
        if m == 1 or n_boot <= 0:
            return mu, mu, mu
        boots = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, m, size=m)  
            boots[b] = float(np.mean(x[idx]))
        alpha = (1.0 - level) / 2.0
        lo, up = np.percentile(boots, [100*alpha, 100*(1.0 - alpha)], method="linear")
        return mu, float(lo), float(up)

    method_names = list(methods_cfg.keys())
    if colors is None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#E45756",
                  "#17becf", "#bcbd22", "#8c564b", "#e377c2"][:len(method_names)]
    color_map = {m: c for m, c in zip(method_names, colors)}

    frames: Dict[str, pd.DataFrame] = {m: pd.read_csv(cfg["path"]) for m, cfg in methods_cfg.items()}

    def extract_vals(df: pd.DataFrame, base: str, suffix: str) -> np.ndarray:
        col = f"{base}{suffix}" if f"{base}{suffix}" in df.columns else base
        return df[col].dropna().to_numpy() if col in df.columns else np.array([])

    stats = {m: {} for m in method_names}
    for m in method_names:
        df = frames[m]
        suffix = methods_cfg[m].get("suffix", "")
        auc   = extract_vals(df, "auc_score", suffix)
        type1 = extract_vals(df, "type1",     suffix)
        power = extract_vals(df, "power",     suffix)

        count_col = f"count{suffix}" if f"count{suffix}" in df.columns else ("count" if "count" in df.columns else None)
        if count_col is not None and count_col in df.columns and (f"power{suffix}" in df.columns or "power" in df.columns):
            sub = df[[ (f"power{suffix}" if f"power{suffix}" in df.columns else "power"), count_col ]].dropna()
            union = (5.0 * sub.iloc[:, 0] + sub.iloc[:, 1]).to_numpy() / 10.0 if not sub.empty else np.array([])
        else:
            union = power.copy()

        stats[m]["AUC"]   = auc
        stats[m]["Power"] = power
        stats[m]["TypeI"] = type1
        stats[m]["Union"] = union

    rng = np.random.default_rng(random_state)

    def mean_err(x: np.ndarray):
        if len(x) == 0:
            return np.nan, 0.0
        x = np.asarray(x, dtype=float)
        mu = float(np.mean(x))
        n = len(x)
        if n == 1:
            return mu, 0.0

        if band == "sd":
            sd = float(np.std(x, ddof=1))
            return mu, sd
        elif band == "se":
            sd = float(np.std(x, ddof=1))
            se = sd / np.sqrt(n)
            return mu, z_val * se
        else:  
            mu_b, lo, up = _bootstrap_ci_mean(x, ci_level, n_boot, rng)
            if not np.isfinite(mu_b):
                return mu, 0.0
            low_err = max(0.0, mu_b - lo)
            up_err  = max(0.0, up - mu_b)
            return mu_b, (low_err, up_err)

    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)

    ax_auc = axes[0]
    x_positions = np.arange(len(method_names))
    y_min, y_max = +np.inf, -np.inf
    for x0, method in zip(x_positions, method_names):
        vals = stats[method]["AUC"]
        if vals.size == 0:
            continue
        x = np.random.normal(loc=x0, scale=jitter, size=len(vals))
        ax_auc.scatter(x, vals, color=color_map[method], alpha=0.75, s=point_size, edgecolor="none")
        bp = ax_auc.boxplot(
            [vals], vert=True, positions=[x0], widths=0.55, patch_artist=True,
            showfliers=showfliers,
            medianprops=dict(lw=2.0, color="black", linestyle="--"),
            whiskerprops=dict(lw=1.4, color="black"),
            capprops=dict(lw=1.4, color="black"),
            boxprops=dict(lw=1.2, edgecolor="black"),
            showmeans=False, meanline=False
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color_map[method]); patch.set_alpha(box_alpha)
        y_min = min(y_min, np.min(vals)); y_max = max(y_max, np.max(vals))

    ax_auc.set_xticks(x_positions)
    ax_auc.set_xticklabels(method_names, rotation=18, ha='right')
    ax_auc.set_title("AUC Score")
    ax_auc.grid(True, axis="y", alpha=0.3)
    ax_auc.set_ylim(
        max(0.45, y_min - 0.02 if np.isfinite(y_min) else 0.45),
        min(1.005, y_max + 0.02 if np.isfinite(y_max) else 1.005),
    )

    panels = [("Power", "Power(C1)"), ("TypeI", "Type I Error"), ("Union", "Power(C1∪C2)")]
    for ax, (key, ttl) in zip(axes[1:], panels):
        xs = np.arange(len(method_names))
        for i, m in enumerate(method_names):
            mu, err = mean_err(stats[m][key])
            c = color_map[m]
            if isinstance(err, tuple):  
                low, up = err
                yerr = np.array([[low], [up]])
            else: 
                yerr = err
            ax.errorbar(i, mu, yerr=yerr,
                        fmt='o', markersize=7.8, markerfacecolor=c, markeredgecolor='black', markeredgewidth=0.7,
                        ecolor=c, elinewidth=2.2, capsize=5, capthick=1.8)
        ax.set_xticks(xs); ax.set_xticklabels(method_names, rotation=18, ha='right')
        ax.set_title(ttl)
        ax.grid(True, alpha=0.25)

        if key == "TypeI":
            ax.axhline(0.05, ls="--", lw=2.2, color="magenta", alpha=0.9) 
            ax.set_ylim(-0.01, 0.102)
        else:
            ax.set_ylim(-0.02, 1.05)

    handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                          markerfacecolor=color_map[m], markeredgecolor='black', markeredgewidth=0.7,
                          color=color_map[m], label=m, markersize=8) for m in method_names]
    ref_line = plt.Line2D([0],[0], color="magenta", lw=2.2, ls="--", label=r"$\alpha=0.05$")
    handles.append(ref_line)
    labels = method_names + [r"$\alpha=0.05$"]

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 8),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=12,        
        markerscale=1.1,   
        handlelength=2.7,  
        handletextpad=0.5, 
        labelspacing=0.3,  
        borderpad=0.2,     
        columnspacing=0.8  
    )

    if title is not None:
        fig.suptitle(title, y=1.03, fontsize=18)

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0., dpi=300)

    return fig, axes


if __name__ == "__main__":
    methods_cfg = {
        "LOCO":  {"path": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp2\1000\summary\loco_summary.csv",  "suffix": "_0_loco"},
        "nLOCO": {"path": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp2\1000\summary\nloco_summary.csv", "suffix": "_0_nloco"},
        "dLOCO": {"path": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp2\1000\summary\dloco_summary.csv", "suffix": "_0_dloco"},
        "DFI":   {"path": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp2\1000\summary\dfi_summary.csv",   "suffix": "_x_dfi"},
        "FDFI":  {"path": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp2\1000\summary\cpi_summary.csv",  "suffix": "_x_cpi"},
    }

    fig, axes = plot_metrics_one_row_with_box_auc(
        methods_cfg,
        band="ci",
        ci_level=0.95,
        n_boot=1000,
        random_state=42,
        save_path=r"Exp2\Mixture.pdf",
        title=None 
    )
    plt.show()
