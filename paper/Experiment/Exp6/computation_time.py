import csv
import os
import time

num_runs = 10

summary_dir = r"F:\Code\2025\ICLR_Flow_Disentangle\Computation_time\Exp2\1000\summary"
os.makedirs(summary_dir, exist_ok=True)
csv_path_runtime = os.path.join(summary_dir, "runtime_summary.csv")

if not os.path.exists(csv_path_runtime):
    with open(csv_path_runtime, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run',
            'time_loco_sec',
            'time_nloco_sec',
            'time_dloco_sec',
            'time_DFI_sec',
            'time_CPI_sec',
            'time_FDFI_Z_sec',
            'time_FDFI_sec',
        ])

for run in range(1, num_runs + 1):
    print(f"\n================ 第 {run} 次运行 ================\n")

    import sys
    sys.path.append('ICLR_Flow_Disentangle')

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from Inference.data import Exp2
    from sklearn.ensemble import RandomForestRegressor
    from Flow_Matching.flow_matching import FlowMatchingModel

    seed = np.random.randint(0, 1000)
    rho1 = 0.8
    rho2 = 0.2
    mix_weight = 0.2

 
    X_full_train, y_train = Exp2().generate(n=3000, rho1=rho1, rho2=rho2, seed=seed, mix_weight=mix_weight)

    D = X_full_train.shape[1]
    n_jobs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowMatchingModel(
        X=X_full_train,
        dim=D,
        device=device,
        hidden_dim=96,
        time_embed_dim=64,
        num_blocks=1,
        use_bn=False
    )

    model.fit(num_steps=15000, batch_size=256, lr=1e-3, show_plot=False)

  
    X_full, y = Exp2().generate(n=1000, rho1=rho1, rho2=rho2, seed=seed, mix_weight=mix_weight)
    print(X_full.shape)

    from Inference.estimators import LOCOEstimator, nLOCOEstimator, dLOCOEstimator, DFIEstimator, CPIEstimator, CPIZ_Flow_Model_Estimator, CPI_Flow_Model_Estimator

    # --- LOCO ---
    estimator1 = LOCOEstimator(
        regressor=RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=n_jobs
        )
    )
    t0 = time.perf_counter()
    phi_0_loco, se_0_loco = estimator1.importance(X_full, y)
    t1 = time.perf_counter()
    time_loco = t1 - t0
    print(f"[LOCO] time: {time_loco:.3f} s")

    # --- nLOCO ---
    estimator2 = nLOCOEstimator(
        regressor=RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=n_jobs
        )
    )
    t0 = time.perf_counter()
    phi_0_nloco, se_0_nloco = estimator2.importance(X_full, y)
    t1 = time.perf_counter()
    time_nloco = t1 - t0
    print(f"[nLOCO] time: {time_nloco:.3f} s")

    # --- dLOCO ---
    estimator3 = dLOCOEstimator(
        regressor=RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=n_jobs
        )
    )
    t0 = time.perf_counter()
    phi_0_dloco, se_0_dloco = estimator3.importance(X_full, y)
    t1 = time.perf_counter()
    time_dloco = t1 - t0
    print(f"[dLOCO] time: {time_dloco:.3f} s")

    # --- DFI ---
    estimator4 = DFIEstimator(
        regressor = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )
    )
    t0 = time.perf_counter()
    phi_dfi, se_dfi = estimator4.importance(X_full, y)
    t1 = time.perf_counter()
    time_DFI = t1 - t0
    print(f"[DFI] time: {time_DFI:.3f} s")

    # --- CPI ---
    estimator5 = CPIEstimator(
            regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )
    )

    t0 = time.perf_counter()
    phi_0_cpi, se_0_cpi = estimator5.importance(X_full, y)
    t1 = time.perf_counter()
    time_CPI = t1 - t0
    print(f"[CPI] time: {time_CPI:.3f} s")

    # --- FDFI-Z ---
    estimator6 = CPIZ_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 ),
        flow_model=model
    
    )
    
    t0 = time.perf_counter()
    phi_z_cpi, se_z_cpi = estimator6.importance(X_full, y)
    t1 = time.perf_counter()
    time_FDFI_Z = t1 - t0
    print(f"[FDFI_Z] time: {time_FDFI_Z:.3f} s")

    # --- FDFI ---
    estimator7 = CPI_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 ),
        flow_model=model
    )
    

    t0 = time.perf_counter()
    phi_x_cpi, se_x_cpi = estimator7.importance(X_full, y)
    t1 = time.perf_counter()
    time_FDFI = t1 - t0
    print(f"[FDFI] time: {time_FDFI:.3f} s")



    with open(csv_path_runtime, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(time_loco, 6),
            round(time_nloco, 6),
            round(time_dloco, 6),
            round(time_DFI, 6),
            round(time_CPI, 6),
            round(time_FDFI_Z, 6),
            round(time_FDFI, 6),

        ])

