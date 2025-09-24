import csv
import os

num_runs = 100 


txt_dir_loco = "Exp2/1000_new/phi_logs/loco_phi_logs"
csv_path_loco = "Exp2/1000_new/summary/loco_summary.csv"


txt_dir_nloco = "Exp2/1000_new/phi_logs/nloco_phi_logs"
csv_path_nloco = "Exp2/1000_new/summary/nloco_summary.csv"

txt_dir_dloco = "Exp2/1000_new/phi_logs/dloco_phi_logs"
csv_path_dloco = "Exp2/1000_new/summary/dloco_summary.csv"

txt_dir_dfi = "Exp2/1000_new/phi_logs/dfi_phi_logs"
csv_path_dfi = "Exp2/1000_new/summary/dfi_summary.csv"


txt_dir_cpi = "Exp2/1000_new/phi_logs/cpi_phi_logs"
csv_path_cpi = "Exp2/1000_new/summary/cpi_summary.csv"


os.makedirs(txt_dir_loco, exist_ok=True)
os.makedirs(txt_dir_nloco, exist_ok=True)
os.makedirs(txt_dir_dloco, exist_ok=True)
os.makedirs(txt_dir_dfi, exist_ok=True)
os.makedirs(txt_dir_cpi, exist_ok=True)


with open(csv_path_loco, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_loco', 
        'power_0_loco', 'type1_0_loco', 'count_0_loco',
    ])

with open(csv_path_nloco, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_nloco', 
        'power_0_nloco', 'type1_0_nloco', 'count_0_nloco',
    ])

with open(csv_path_dloco, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_dloco', 
        'power_0_dloco', 'type1_0_dloco', 'count_0_dloco',
    ])

with open(csv_path_dfi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_dfi',
        'power_x_dfi', 'type1_x_dfi', 'count_x_dfi',
    ])



with open(csv_path_cpi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_cpi',
        'power_x_cpi', 'type1_x_cpi', 'count_x_cpi',
         
    ])   





for run in range(1, num_runs + 1):
    print(f"\n================  {run} ================\n")
    
    #########################################################################################################################################################################
    #                                                                           Flow Matching                                                                               #
    #########################################################################################################################################################################
    
    import sys
    sys.path.append('ICLR_Flow_Disentangle')
    
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    

    from Inference.data import Exp2
    from sklearn.ensemble import RandomForestRegressor
    
    from sklearn.model_selection import train_test_split
    
    
    seed = np.random.randint(0, 1000)  
    rho1 = 0.8
    rho2 = 0.2
    mix_weight = 0.2
    X_full, y = Exp2().generate(n=3000, rho1=rho1, rho2=rho2, seed=seed, mix_weight=mix_weight) 

    
    D=X_full.shape[1]
    
    n_jobs=42


    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    from Flow_Matching.flow_matching import FlowMatchingModel
    import torch
    
    model = FlowMatchingModel(
        X=X_full,
        dim=D,
        device=device,
        hidden_dim=96,        
        time_embed_dim=64,     
        num_blocks=1,
        use_bn=False
    )
    model.fit(num_steps=15000, batch_size=256, lr=1e-3, show_plot=True)

    from scipy.stats import norm
    from Inference.utils import evaluate_importance



    X_full, y = Exp2().generate(n=1000, rho1=rho1, rho2=rho2, seed=seed, mix_weight=mix_weight) 

    
    #########################################################################################################################################################################
    #                                                                           LOCO                                                                                        #
    #########################################################################################################################################################################
    
    from Inference.estimators import LOCOEstimator, nLOCOEstimator, dLOCOEstimator
    from sklearn.ensemble import RandomForestRegressor

    estimator1 = LOCOEstimator(
            regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 )
    )
    
    phi_0_loco, se_0_loco = estimator1.importance(X_full, y)


    phi_0_loco_test = phi_0_loco 

    se_0_loco_test = se_0_loco 
    
    z_score_0_loco = phi_0_loco_test / se_0_loco_test
    
    p_value_0_loco = 1 - norm.cdf(z_score_0_loco)
    rounded_p_value_0_loco = np.round(p_value_0_loco, 3)
    
    print(rounded_p_value_0_loco)



    estimator2 = nLOCOEstimator(
            regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 )
    )
    
    phi_0_nloco, se_0_nloco = estimator2.importance(X_full, y)
    

    phi_0_nloco_test = phi_0_nloco 

    se_0_nloco_test = se_0_nloco 
    
    z_score_0_nloco = phi_0_nloco_test / se_0_nloco_test
    
    p_value_0_nloco = 1 - norm.cdf(z_score_0_nloco)
    rounded_p_value_0_nloco = np.round(p_value_0_nloco, 3)
    
    print(rounded_p_value_0_nloco)



    estimator3 = dLOCOEstimator(
            regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 )
    )
    
    phi_0_dloco, se_0_dloco = estimator3.importance(X_full, y)
    

    phi_0_dloco_test = phi_0_dloco 

    se_0_dloco_test = se_0_dloco 
    
    z_score_0_dloco = phi_0_dloco_test / se_0_dloco_test
    
    p_value_0_dloco = 1 - norm.cdf(z_score_0_dloco)
    rounded_p_value_0_dloco = np.round(p_value_0_dloco, 3)
    
    print(rounded_p_value_0_dloco)


    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_loco = roc_auc_score(y_true[mask], np.array(phi_0_loco)[mask])

    auc_score_0_nloco = roc_auc_score(y_true[mask], np.array(phi_0_nloco)[mask])

    auc_score_0_dloco = roc_auc_score(y_true[mask], np.array(phi_0_dloco)[mask])

    
    print(f"AUC for phi_0_loco: {auc_score_0_loco:.4f}")
    print(f"AUC for phi_0_nloco: {auc_score_0_nloco:.4f}")
    print(f"AUC for phi_0_dloco: {auc_score_0_dloco:.4f}")


    alpha = 0.05
    
    power_0_loco, type1_0_loco, count_0_loco = evaluate_importance(p_value_0_loco, y_true, alpha)
    power_0_nloco, type1_0_nloco, count_0_nloco = evaluate_importance(p_value_0_nloco, y_true, alpha)
    power_0_dloco, type1_0_dloco, count_0_dloco = evaluate_importance(p_value_0_dloco, y_true, alpha)

        
    print(f"[loco]    Power: {power_0_loco:.3f}   Type I Error: {type1_0_loco:.3f}   Count: {count_0_loco}")
    print(f"[nloco]    Power: {power_0_nloco:.3f}   Type I Error: {type1_0_nloco:.3f}   Count: {count_0_nloco}")
    print(f"[dloco]    Power: {power_0_dloco:.3f}   Type I Error: {type1_0_dloco:.3f}   Count: {count_0_dloco}")


    with open(f"{txt_dir_loco}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\n")
        for i in range(len(phi_0_loco)):
            txtfile.write(
                          f"{phi_0_loco[i]:.6f}\t{se_0_loco[i]:.6f}\t")
            
    with open(f"{txt_dir_nloco}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\n")
        for i in range(len(phi_0_nloco)):
            txtfile.write(
                          f"{phi_0_nloco[i]:.6f}\t{se_0_nloco[i]:.6f}\t")
            
    with open(f"{txt_dir_dloco}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\n")
        for i in range(len(phi_0_dloco)):
            txtfile.write(
                          f"{phi_0_dloco[i]:.6f}\t{se_0_dloco[i]:.6f}\t")



    with open(csv_path_loco, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_loco, 6), 
            round(power_0_loco, 6), round(type1_0_loco, 6), int(count_0_loco),
        ])

    with open(csv_path_nloco, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_nloco, 6), 
            round(power_0_nloco, 6), round(type1_0_nloco, 6), int(count_0_nloco),
        ])

    with open(csv_path_dloco, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_dloco, 6), 
            round(power_0_dloco, 6), round(type1_0_dloco, 6), int(count_0_dloco),
        ])



    #########################################################################################################################################################################
    #                                                                           DFI                                                                                         #
    #########################################################################################################################################################################

    from Inference.estimators import  DFIEstimator
    from scipy.stats import norm
    from sklearn.ensemble import RandomForestRegressor
    from Inference.utils import evaluate_importance
    
    

    
    estimator4 = DFIEstimator(
        regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )

    )

    
    phi_x_dfi, se_x_dfi = estimator4.importance(X_full, y)
    
    phi_x_dfi_test = phi_x_dfi 

    se_x_dfi_test = se_x_dfi 
    
    z_score_x_dfi = phi_x_dfi_test / se_x_dfi_test
    
    p_value_x_dfi = 1 - norm.cdf(z_score_x_dfi)
    rounded_p_value_x_dfi = np.round(p_value_x_dfi, 3)
    
    print(rounded_p_value_x_dfi)

    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#



    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_x_dfi = roc_auc_score(y_true[mask], np.array(phi_x_dfi)[mask])

    

    print(f"AUC for phi_x_dfi: {auc_score_x_dfi:.4f}")

    alpha = 0.05
    
    power_x_dfi, type1_x_dfi, count_x_dfi = evaluate_importance(p_value_x_dfi, y_true, alpha)


        
    print(f"[X]    Power: {power_x_dfi:.3f}   Type I Error: {type1_x_dfi:.3f}   Count: {count_x_dfi}")



    with open(f"{txt_dir_dfi}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_x\tstd_x\n")
        for i in range(len(phi_x_dfi)):
            txtfile.write(
                          f"{phi_x_dfi[i]:.6f}\t{se_x_dfi[i]:.6f}\n")



    with open(csv_path_dfi, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_x_dfi, 6),
            round(power_x_dfi, 6), round(type1_x_dfi, 6), int(count_x_dfi),
        ])
    


    #########################################################################################################################################################################
    #                                                                           FDFI(CPI)                                                                                        #
    #########################################################################################################################################################################


    from Inference.estimators import CPI_Flow_Model_Estimator



    estimator5 = CPI_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 ),
        flow_model=model
    
    )
    
    
    phi_x_cpi, se_x_cpi = estimator5.importance(X_full, y)
    
    phi_x_cpi_test = phi_x_cpi 

    se_x_cpi_test = se_x_cpi 
    
    z_score_x_cpi = phi_x_cpi_test / se_x_cpi_test
    
    
    p_value_x_cpi = 1 - norm.cdf(z_score_x_cpi)
    rounded_p_value_x_cpi = np.round(p_value_x_cpi, 3)
    
    print(rounded_p_value_x_cpi)

    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_x_cpi = roc_auc_score(y_true[mask], np.array(phi_x_cpi)[mask])

        
    print(f"AUC for phi_x_cpi: {auc_score_x_cpi:.4f}")


    alpha = 0.05
    
    power_x_cpi, type1_x_cpi, count_x_cpi = evaluate_importance(p_value_x_cpi, y_true, alpha)


        
    print(f"[X]    Power: {power_x_cpi:.3f}   Type I Error: {type1_x_cpi:.3f}   Count: {count_x_cpi}")

    
    with open(f"{txt_dir_cpi}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_x\tstd_x\n")
        for i in range(len(phi_x_cpi)):
            txtfile.write(
                          f"{phi_x_cpi[i]:.6f}\t{se_x_cpi[i]:.6f}\n")
            
    with open(csv_path_cpi, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_x_cpi, 6),
            round(power_x_cpi, 6), round(type1_x_cpi, 6), int(count_x_cpi),
        ])
