
import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SAMPLE_FILES = {
    200: r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\200\summary\runtime_summary.csv",
    400: r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\400\summary\runtime_summary.csv",
    600: r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\600\summary\runtime_summary.csv",
    800: r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\800\summary\runtime_summary.csv",
    1000: r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\1000\summary\runtime_summary.csv",
}

OUT_DIR = r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2"
os.makedirs(OUT_DIR, exist_ok=True)


USE_LOG_Y = True
PREFERRED_ORDER = ["CPI", "LOCO", "nLOCO", "dLOCO", "DFI", "FDFI-Z", "FDFI", "SHAP"]



def tidy_long_dataframe(sample_files: dict) -> pd.DataFrame:
    frames = []
    for ss, path in sample_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found for sample size {ss}: {path}")
        df = pd.read_csv(path)
        time_cols = [c for c in df.columns if c.startswith("time_")]
        if not time_cols:
            raise ValueError(f"No time_* columns found in {path}")

        sub = df[time_cols].copy()
        sub["SampleSize"] = ss
        long = sub.melt(id_vars="SampleSize", var_name="EstimatorRaw", value_name="Time")

        def clean_name(s):
            s = re.sub(r"^time_", "", s)
            s = re.sub(r"_sec$", "", s)
            s = s.replace("_", "-")
            s_up = s.upper()
            mapping = {
                "LOCO":"LOCO", "NLOCO":"nLOCO", "DLOCO":"dLOCO",
                "CPI":"CPI", "DFI":"DFI", "DFI-Z":"DFI-Z",
                "FDFI":"FDFI", "FDFI-Z":"FDFI-Z", "SHAP": "Shapley value",
            }
            return mapping.get(s_up, s)
        long["Estimator"] = long["EstimatorRaw"].apply(clean_name)
        long = long.drop(columns=["EstimatorRaw"])
        long = long[np.isfinite(long["Time"]) & (long["Time"] > 0)]
        frames.append(long)
    return pd.concat(frames, ignore_index=True)

def compute_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    stats = df_long.groupby(["Estimator", "SampleSize"])["Time"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    return stats

def order_estimators(est_list, preferred=PREFERRED_ORDER):
    present = set(est_list)
    ordered = [m for m in preferred if m in present]
    others = sorted(list(present - set(ordered)))
    return ordered + others


df_long = tidy_long_dataframe(SAMPLE_FILES)
stats_long = compute_stats(df_long)

estimators = order_estimators(stats_long["Estimator"].unique().tolist())
sample_sizes = sorted(SAMPLE_FILES.keys())


wide_mean = stats_long.pivot(index="Estimator", columns="SampleSize", values="mean").reindex(estimators)
wide_std  = stats_long.pivot(index="Estimator", columns="SampleSize", values="std").reindex(estimators)
pd.concat({"mean": wide_mean, "std": wide_std}, axis=1).to_csv(os.path.join(OUT_DIR, "time_stats_wide.csv"))


fig, ax = plt.subplots(figsize=(10.5, 3.6), dpi=220)

x = np.arange(len(estimators), dtype=float)


k = len(sample_sizes)
if k == 1:
    offsets = [0.0]
else:
    span = 0.6  
    offsets = np.linspace(-span/2, span/2, k)


handles = []
labels = []
for off, ss in zip(offsets, sample_sizes):
    means = []
    stds  = []
    for est in estimators:
        row = stats_long[(stats_long["Estimator"] == est) & (stats_long["SampleSize"] == ss)]
        if len(row) == 1:
            means.append(float(row["mean"].values[0]))
            std_val = float(row["std"].values[0])
            stds.append(0.0 if math.isnan(std_val) else std_val)
        else:
            means.append(np.nan)
            stds.append(np.nan)
    means = np.array(means, dtype=float)
    stds  = np.array(stds, dtype=float)
    mask = np.isfinite(means)

    h = ax.errorbar(
        x[mask] + off, means[mask],
        yerr=stds[mask],
        fmt='o',          
        capsize=3,
        elinewidth=0.9,
        linewidth=0.9,
        markersize=4.0,
        label=str(ss)
    )
    handles.append(h)
    labels.append(str(ss))


ax.set_xticks(x)
ax.set_xticklabels(estimators, rotation=0)
ax.set_xlabel("Methods")
ax.set_ylabel("Time (seconds)")


if USE_LOG_Y:
    ax.set_yscale('log')

ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(handles, labels, title="Sample Size", loc="upper left", ncol=2, fontsize=8, title_fontsize=9, frameon=False)

fig.tight_layout()
png_path = os.path.join(OUT_DIR, "runtime_by_estimator_points_errbar.png")
pdf_path = os.path.join(OUT_DIR, "runtime_by_estimator_points_errbar.pdf")
fig.savefig(png_path, bbox_inches='tight', pad_inches=0., dpi=300)
fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0., dpi=300)
plt.close(fig)

print(f"[OK] Saved:\n- {png_path}\n- {pdf_path}\n[OK] Stats saved:\n- {os.path.join(OUT_DIR, 'time_stats_wide.csv')}")
