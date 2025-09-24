import sys
sys.path.append('ICLR_Flow_Disentangle')

import numpy as np
import torch
from Inference.data import Complex_example_4
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
    term_xx = Kxx.sum() / (n*(n-1))
    term_yy = Kyy.sum() / (m*(m-1))
    term_xy = Kxy.mean()
    return term_xx + term_yy - 2.0*term_xy

n_list     = [2000, 4000, 6000, 8000, 10000]    
steps_list = [5000, 10000, 15000, 20000, 25000] 
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
rho = 0.8
batch_size = 256
repeat_runs = 50          
test_size = 0.5           
split_random_state = 0    

results = []

for n_val, steps in zip(n_list, steps_list):
    mmd_train_values = []
    mmd_test_values  = []

    for run in range(1, repeat_runs + 1):
        print(f"\n=== n={n_val}, steps={steps}, run={run} ===")

        seed = np.random.randint(0, 10_000)
        X_full, y = Complex_example_4().generate(n_val, rho, seed)
        D = X_full.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y, test_size=test_size, random_state=split_random_state, shuffle=True
        )
        n_train, n_test = len(X_train), len(X_test)
        assert n_train + n_test == n_val

        model = FlowMatchingModel(
            X=X_train,
            dim=D,
            device=device,
            hidden_dim=128,
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

        print(f"MMD^2_train: {mmd_train:.6f} | MMD^2_test: {mmd_test:.6f}")

        mmd_train_values.append(float(mmd_train))
        mmd_test_values.append(float(mmd_test))
      
    mmd_train_mean = float(np.mean(mmd_train_values))
    mmd_train_std  = float(np.std(mmd_train_values, ddof=1))
    mmd_test_mean  = float(np.mean(mmd_test_values))
    mmd_test_std   = float(np.std(mmd_test_values, ddof=1))

    print(f"\n>> n={n_val}, steps={steps} → "
          f"train_mean={mmd_train_mean:.6f}, train_std={mmd_train_std:.6f} | "
          f"test_mean={mmd_test_mean:.6f}, test_std={mmd_test_std:.6f}")

    results.append((n_val, steps, n_train, n_test,
                    mmd_train_mean, mmd_train_std, mmd_test_mean, mmd_test_std))

out_path = "mmd_results_train_test.txt"
with open(out_path, "w") as f:
    f.write("n\tsteps\tn_train\tn_test\tmmd_train_mean\tmmd_train_std\tmmd_test_mean\tmmd_test_std\n")
    for (n_val, steps, ntr, nte, trm, trs, tem, tes) in results:
        f.write(f"{n_val}\t{steps}\t{ntr}\t{nte}\t{trm:.6f}\t{trs:.6f}\t{tem:.6f}\t{tes:.6f}\n")

print(f"\n results have been saved at {out_path}")
