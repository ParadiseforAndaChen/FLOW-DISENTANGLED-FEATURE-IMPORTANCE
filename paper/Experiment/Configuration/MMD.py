import sys
sys.path.append('ICLR_Flow_Disentangle')

import numpy as np
import torch
from itertools import product
from Inference.data import Exp1
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist

from Flow_Matching.flow_matching import FlowMatchingModel


def mmd2_unbiased_rbf(X, Y, gamma):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return term_xx + term_yy - 2.0 * term_xy


n_total = 2000       
steps   = 5000       
batch_sizes  = [32, 64, 128, 256, 384]  
hidden_dims  = [64, 128, 256, 384]               
test_size = 0.5      
rho = 0.8
repeat_runs = 50     

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


results_per_combo = {
    (hd, bs): {"train": [], "test": []}
    for hd, bs in product(hidden_dims, batch_sizes)
}

print(f"n_total={n_total}, steps={steps}, repeats={repeat_runs}, "
      f"hidden_dims={hidden_dims}, batches={batch_sizes}")
print(f"device={device}")

for run in range(1, repeat_runs + 1):
    
    seed = np.random.randint(0, 100000)
    X_full, y = Exp1().generate(n_total, rho, seed)


    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=0, shuffle=True
    )
    n_train, n_test = len(X_train), len(X_test)
    D = X_train.shape[1]
 

    print(f"\n===== RUN {run}/{repeat_runs} | train={n_train}, test={n_test}, D={D} =====")

   
    for hidden_dim, batch_size in product(hidden_dims, batch_sizes):
        print(f"[run {run}] training with hidden_dim={hidden_dim}, batch={batch_size} ...")

      
        model = FlowMatchingModel(
            X=X_train,
            dim=D,
            device=device,
            hidden_dim=hidden_dim,   
            time_embed_dim=64,
            num_blocks=1,
            use_bn=False
        )
        model.fit(num_steps=steps, batch_size=batch_size, lr=1e-3, show_plot=False)

     
        with torch.no_grad():
            Z_train = np.random.randn(n_train, D).astype(np.float32)
            X_new_train = model.sample_batch(
                torch.from_numpy(Z_train).to(device), t_span=(0, 1)
            ).cpu().numpy()

            Z_test = np.random.randn(n_test, D).astype(np.float32)
            X_new_test = model.sample_batch(
                torch.from_numpy(Z_test).to(device), t_span=(0, 1)
            ).cpu().numpy()

      
        Z_all_train = np.vstack([X_train, X_new_train])
        med_train = np.median(pdist(Z_all_train))
        gamma_train = 1.0 / (med_train**2 + 1e-12)

        Z_all_test = np.vstack([X_test, X_new_test])
        med_test = np.median(pdist(Z_all_test))
        gamma_test = 1.0 / (med_test**2 + 1e-12)

        
        mmd_train = mmd2_unbiased_rbf(X_train, X_new_train, gamma_train)
        mmd_test  = mmd2_unbiased_rbf(X_test,  X_new_test,  gamma_test)

        print(f"[run {run}] (hd={hidden_dim}, batch={batch_size}) "
              f"-> MMD^2_train={mmd_train:.6f} | MMD^2_test={mmd_test:.6f}")

        
        results_per_combo[(hidden_dim, batch_size)]["train"].append(float(mmd_train))
        results_per_combo[(hidden_dim, batch_size)]["test"].append(float(mmd_test))


out_path = "mmd_results_by_hidden_batch_train_test_layer_1.txt"
with open(out_path, "w") as f:
    f.write("hidden_dim\tbatch\tn_train\tn_test\tsteps\tsteps_per_epoch_train\t"
            "reuse_per_sample_train\tmmd_train_mean\tmmd_train_std\t"
            "mmd_test_mean\tmmd_test_std\n")

    
    for (hidden_dim, batch_size), vals in results_per_combo.items():
        steps_per_epoch_train = int(np.ceil(n_train / batch_size))
        reuse_per_sample_train = steps * batch_size / n_train

        arr_tr = np.array(vals["train"], dtype=float)
        arr_te = np.array(vals["test"], dtype=float)

        mmd_train_mean = float(arr_tr.mean()) if len(arr_tr) > 0 else float('nan')
        mmd_train_std  = float(arr_tr.std(ddof=1)) if len(arr_tr) > 1 else 0.0
        mmd_test_mean  = float(arr_te.mean()) if len(arr_te) > 0 else float('nan')
        mmd_test_std   = float(arr_te.std(ddof=1)) if len(arr_te) > 1 else 0.0

        print(f"\n>> (hd={hidden_dim}, batch={batch_size}): "
              f"train_mean={mmd_train_mean:.6f}, train_std={mmd_train_std:.6f} | "
              f"test_mean={mmd_test_mean:.6f}, test_std={mmd_test_std:.6f}")

        f.write(f"{hidden_dim}\t{batch_size}\t{n_train}\t{n_test}\t{steps}\t"
                f"{steps_per_epoch_train}\t{reuse_per_sample_train:.2f}\t"
                f"{mmd_train_mean:.6f}\t{mmd_train_std:.6f}\t"
                f"{mmd_test_mean:.6f}\t{mmd_test_std:.6f}\n")

print(f"\nResults have been saved at {out_path}")
