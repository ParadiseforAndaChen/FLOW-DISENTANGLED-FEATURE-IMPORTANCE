import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List



def extract_vals(df: pd.DataFrame, base: str, suffix: str) -> np.ndarray:
    col = f"{base}{suffix}" if f"{base}{suffix}" in df.columns else base
    return df[col].dropna().to_numpy() if col in df.columns else np.array([])

def mean_err_symmetric(x: np.ndarray, band: str, z_val: float):
    if len(x) == 0:
        return np.nan, 0.0
    mu = float(np.mean(x))
    if len(x) == 1:
        return mu, 0.0
    sd = float(np.std(x, ddof=1))
    if band == "sd":
        return mu, sd
    se = sd / np.sqrt(len(x))
    return (mu, z_val * se) if band == "ci" else (mu, se)

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


def plot_metrics_row_on_axes(
    methods_cfg: Dict[str, Dict],   
    axes_row,                      
    *,
    band: str="se",
    ci_level: float=0.95,
    jitter: float=0.08,
    point_size: int=22,
    showfliers: bool=False,
    box_alpha: float=0.35,
    colors: Optional[List[str]]=None,
    row_title: Optional[str]=None,    
    panel_title_fs: int=13,            
    axis_label_fs: int=12,            
    tick_fs: int=11,                   
    n_boot: int = 1000,
    random_state: Optional[int] = None,
):
    z_val = _z_from_ci(ci_level)
    rng = np.random.default_rng(random_state)

    method_names = list(methods_cfg.keys())
    if colors is None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#E45756",
                  "#17becf", "#bcbd22", "#8c564b", "#e377c2"][:len(method_names)]
    color_map = {m: c for m, c in zip(method_names, colors)}

    stats = {m: {} for m in method_names}
    for m in method_names:
        df = methods_cfg[m]["df"]
        suffix = methods_cfg[m]["suffix"]

        auc   = extract_vals(df, "auc_score", suffix)
        type1 = extract_vals(df, "type1",     suffix)
        power = extract_vals(df, "power",     suffix)

        count_col = f"count{suffix}" if f"count{suffix}" in df.columns else ("count" if "count" in df.columns else None)
        if count_col is not None and count_col in df.columns and (f"power{suffix}" in df.columns or "power" in df.columns):
            sub = df[[ (f"power{suffix}" if f"power{suffix}" in df.columns else "power"), count_col ]].dropna()
            union = (5.0 * sub.iloc[:, 0] + sub.iloc[:, 1]).to_numpy() / 10.0 if not sub.empty else power.copy()
        else:
            union = power.copy()

        stats[m]["AUC"]   = auc
        stats[m]["Power"] = power
        stats[m]["TypeI"] = type1
        stats[m]["Union"] = union

    ax_auc = axes_row[0]
    x_positions = np.arange(len(method_names))
    y_min, y_max = +np.inf, -np.inf
    for x0, method in zip(x_positions, method_names):
        vals = stats[method]["AUC"]
        if vals.size == 0:
            continue
        xj = np.random.normal(loc=x0, scale=jitter, size=len(vals))
        ax_auc.scatter(xj, vals, color=color_map[method], alpha=0.75, s=point_size, edgecolor="none")
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
    ax_auc.set_xticklabels(method_names, rotation=18, ha='right', fontsize=tick_fs)
    ax_auc.set_title("AUC Score", fontsize=panel_title_fs)
    ax_auc.grid(True, axis="y", alpha=0.3)
    ax_auc.set_ylim(
        max(0.45, y_min - 0.02 if np.isfinite(y_min) else 0.45),
        min(1.005, y_max + 0.02 if np.isfinite(y_max) else 1.005),
    )
    ax_auc.tick_params(axis='y', labelsize=tick_fs)

    panels = [("Power", "Power(C1)"), ("TypeI", "Type I Error"), ("Union", "Power(C1∪C2)")]
    for ax, (key, ttl) in zip(axes_row[1:], panels):
        xs = np.arange(len(method_names))
        for i, m in enumerate(method_names):
            vals = stats[m][key]
            c = color_map[m]
            if band == "ci":
                mu_b, lo, up = _bootstrap_ci_mean(vals, ci_level, n_boot, rng)
                if not np.isfinite(mu_b):
                    continue
                low_err = max(0.0, mu_b - lo)
                up_err  = max(0.0, up - mu_b)
                yerr = np.array([[low_err], [up_err]])  
                ax.errorbar(i, mu_b, yerr=yerr,
                            fmt='o', markersize=7.8, markerfacecolor=c, markeredgecolor='black', markeredgewidth=0.7,
                            ecolor=c, elinewidth=2.2, capsize=5, capthick=1.8)
            else:
                mu, err = mean_err_symmetric(vals, band, z_val)
                ax.errorbar(i, mu, yerr=err,
                            fmt='o', markersize=7.8, markerfacecolor=c, markeredgecolor='black', markeredgewidth=0.7,
                            ecolor=c, elinewidth=2.2, capsize=5, capthick=1.8)

        ax.set_xticks(xs); ax.set_xticklabels(method_names, rotation=18, ha='right', fontsize=tick_fs)
        ax.set_title(ttl, fontsize=panel_title_fs)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='y', labelsize=tick_fs)

        if key == "TypeI":
            ax.axhline(0.05, ls="--", lw=2.2, color="magenta", alpha=0.9)
            ax.set_ylim(-0.01, 0.102)
        else:
            ax.set_ylim(-0.02, 1.05)

    if row_title is not None:
        axes_row[0].set_ylabel(row_title, fontsize=axis_label_fs)

    for ax in axes_row:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)


if __name__ == "__main__":

    group_defs = [
        ("LOCO",       "loco", "_0_loco"),
        ("CPI",        "cpi",  "_0_cpi"),
        ("FDFI(SCPI)", "loco", "_x_loco"),
        ("FDFI(CPI)",  "cpi",  "_x_cpi"),
        ("DFI",        "dfi",  "_x_dfi"),
    ]

    file_map_RF = {
        "cpi":  r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\RF\summary\cpi_summary.csv",
        "loco": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\RF\summary\loco_summary.csv",
        "dfi":  r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\RF\summary\dfi_summary.csv",
    }
    file_map_NN = {
        "cpi":  r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\NN\summary\cpi_summary.csv",
        "loco": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\NN\summary\loco_summary.csv",
        "dfi":  r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\NN\summary\dfi_summary.csv",
    }
    file_map_Lasso = {
        "cpi":  r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\Lasso\summary\cpi_summary.csv",
        "loco": r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\Lasso\summary\loco_summary.csv",
        "dfi":  r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\Lasso\summary\dfi_summary.csv",
    }

    cached_frames: Dict[str, pd.DataFrame] = {}
    def read_once(path: str) -> pd.DataFrame:
        if path not in cached_frames:
            cached_frames[path] = pd.read_csv(path)
        return cached_frames[path]

    def build_methods_cfg(file_map: Dict[str, str]) -> Dict[str, Dict]:
        cfg: Dict[str, Dict] = {}
        for disp_name, file_key, suffix in group_defs:
            df = read_once(file_map[file_key])
            cfg[disp_name] = {"df": df, "suffix": suffix}
        return cfg

    cfg_rf    = build_methods_cfg(file_map_RF)
    cfg_nn    = build_methods_cfg(file_map_NN)
    cfg_lasso = build_methods_cfg(file_map_Lasso)

    method_names = list(cfg_rf.keys())
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#E45756",
                   "#17becf", "#bcbd22", "#8c564b", "#e377c2"][:len(method_names)]

    FIGSIZE        = (14.5, 9.2) 
    DPI            = 140
    PANEL_TITLE_FS = 14           
    AXIS_LABEL_FS  = 18           
    TICK_FS        = 12          
    LEGEND_FS      = 18           
    SUPTITLE       = None         
    SUPTITLE_FS    = 16

    fig, axes = plt.subplots(3, 4, figsize=FIGSIZE, dpi=DPI, squeeze=False)
    fig.subplots_adjust(hspace=0.6, wspace=0.35, bottom=0.14)  

    plot_metrics_row_on_axes(cfg_rf,    axes[0, :], band="ci", ci_level=0.95,
                             row_title="random forest",
                             colors=base_colors,
                             panel_title_fs=PANEL_TITLE_FS,
                             axis_label_fs=AXIS_LABEL_FS,
                             tick_fs=TICK_FS,
                             n_boot=1000, random_state=42)

    plot_metrics_row_on_axes(cfg_nn,    axes[1, :], band="ci", ci_level=0.95,
                             row_title="neural network",
                             colors=base_colors,
                             panel_title_fs=PANEL_TITLE_FS,
                             axis_label_fs=AXIS_LABEL_FS,
                             tick_fs=TICK_FS,
                             n_boot=1000, random_state=42)

    plot_metrics_row_on_axes(cfg_lasso, axes[2, :], band="ci", ci_level=0.95,
                             row_title="Lasso",
                             colors=base_colors,
                             panel_title_fs=PANEL_TITLE_FS,
                             axis_label_fs=AXIS_LABEL_FS,
                             tick_fs=TICK_FS,
                             n_boot=1000, random_state=42)

    handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                          markerfacecolor=base_colors[i], markeredgecolor='black', markeredgewidth=0.7,
                          color=base_colors[i], label=method_names[i], markersize=8)
               for i in range(len(method_names))]
    ref_line = plt.Line2D([0],[0], color="magenta", lw=2.2, ls="--", label=r"$\alpha=0.05$")
    handles.append(ref_line)
    labels = method_names + [r"$\alpha=0.05$"]

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 8),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),   
        fontsize=LEGEND_FS,
        markerscale=1.1,
        handlelength=2.52,
        handletextpad=0.6,
        labelspacing=0.35,
        borderpad=0.2,
        columnspacing=0.9
    )

    if SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=SUPTITLE_FS, y=0.995)

    out_path = r"F:\Code\2025\ICLR_Flow_Disentangle\Exp3\summary\Exp3_ThreeRows_AllMetrics_withBottomLegend.pdf"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.show()
    print(f"Saved to: {out_path}")
