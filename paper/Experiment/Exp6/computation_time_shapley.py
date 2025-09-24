import csv
import os
import time

num_runs = 10

summary_dir = r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\1000\summary"
os.makedirs(summary_dir, exist_ok=True)
csv_path_runtime = os.path.join(summary_dir, "shap_runtime_summary.csv")

if not os.path.exists(csv_path_runtime):
    with open(csv_path_runtime, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run',
            'time_shap_sec',
        ])

for run in range(1, num_runs + 1):
    print(f"\n================ {run} ================\n")

    import sys
    sys.path.append('ICLR_Flow_Disentangle')

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from Inference.data import Exp2
    from sklearn.ensemble import RandomForestRegressor


    seed = np.random.randint(0, 1000)
    rho1 = 0.8
    rho2 = 0.2
    mix_weight = 0.2

 


  
    X_full, y = Exp2().generate(n=1000, rho1=rho1, rho2=rho2, seed=seed, mix_weight=mix_weight)
    print(X_full.shape)

    from Inference.estimators import ShapleyEstimator

  
    estimator1 = ShapleyEstimator(
        n_mc=100,         
        exact=False,       
        regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
                )
    )
    t0 = time.perf_counter()
    phi_0, std_0 = estimator1.importance(X_full, y)
    t1 = time.perf_counter()
    time_shap = t1 - t0
    print(f"[SHAP] time: {time_shap:.3f} s")


    with open(csv_path_runtime, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(time_shap, 6),
        ])
