import sys
sys.path.append('ICLR_Flow_Disentangle')

import numpy as np
import torch
import matplotlib.pyplot as plt

from Inference.data import Exp_structure
from sklearn.ensemble import RandomForestRegressor


seed = np.random.randint(0, 100000)   
rho = 0.8
X_full, y, T = Exp_structure().generate(3000, rho, seed)
print(seed)


D=X_full.shape[1]

n_jobs=42


print(X_full.shape)




device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


from Flow_Matching.flow_matching import FlowMatchingModel
import torch

model = FlowMatchingModel(
    X=X_full,
    dim=D,
    device=device,
    hidden_dim=64,        
    time_embed_dim=64,     
    num_blocks=1,
    use_bn=False
)
model.fit(num_steps=15000, batch_size=256, lr=1e-3, show_plot=True)


X_full, y, T = Exp_structure().generate(1000, rho, seed) 

from Inference.estimators import  CPI_Flow_Model_Estimator
from scipy.stats import norm


estimator8 = CPI_Flow_Model_Estimator(
    regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                ),
    flow_model=model

)


phi_x_cpi, se_x_cpi = estimator8.importance(X_full, y)


phi_x_cpi_test = phi_x_cpi 

se_x_cpi_test = se_x_cpi 

z_score_x_cpi = phi_x_cpi_test / se_x_cpi_test


p_value_x_cpi = 1 - norm.cdf(z_score_x_cpi)
rounded_p_value_x_cpi = np.round(p_value_x_cpi, 3)

print(rounded_p_value_x_cpi)
Z = estimator8._encode_to_Z(X_full)
estimator8.plot_H(Z=Z, export_txt_path="map.tsv",
            savepath="map.pdf")

