import csv
import os

num_runs = 100 

txt_dir_loco = "Exp1/0.8/1000/phi_logs/loco_phi_logs"
csv_path_loco = "Exp1/0.8/1000/summary/loco_summary.csv"


txt_dir_cpi = "Exp1/0.8/1000/phi_logs/cpi_phi_logs"
csv_path_cpi = "Exp1/0.8/1000/summary/cpi_summary.csv"


txt_dir_dfi = "Exp1/0.8/1000/phi_logs/dfi_phi_logs"
csv_path_dfi = "Exp1/0.8/1000/summary/dfi_summary.csv"


os.makedirs(txt_dir_loco, exist_ok=True)

os.makedirs(txt_dir_cpi, exist_ok=True)

os.makedirs(txt_dir_dfi, exist_ok=True)


with open(csv_path_loco, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_loco',  'auc_score_x_loco',
        'power_0_loco', 'type1_0_loco', 'count_0_loco',
        'power_x_loco', 'type1_x_loco', 'count_x_loco',
    ])


with open(csv_path_cpi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_cpi',  'auc_score_x_cpi',
        'power_0_cpi', 'type1_0_cpi', 'count_0_cpi', 
        'power_x_cpi', 'type1_x_cpi', 'count_x_cpi',
         
    ])   

with open(csv_path_dfi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_dfi',
        'power_x_dfi', 'type1_x_dfi', 'count_x_dfi',
         
    ])   




for run in range(1, num_runs + 1):
    print(f"\n================ {run}  ================\n")
    
    #########################################################################################################################################################################
    #                                                                           Flow Matching                                                                               #
    #########################################################################################################################################################################
    
    import sys
    sys.path.append('ICLR_Flow_Disentangle')
    
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    from Inference.data import Exp1
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    from sklearn.model_selection import train_test_split
    
    
    seed = np.random.randint(0, 1000)  
    rho = 0.8
    X_full, y = Exp1().generate(3000, rho, seed)
    
    D=X_full.shape[1]
    
    n_jobs=42


    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    from Flow_Matching.flow_matching import FlowMatchingModel
    import torch
    
    model = FlowMatchingModel(
        X=X_full,
        dim=D,
        device=device,
        hidden_dim=128,        
        time_embed_dim=64,     
        num_blocks=1,
        use_bn=False
    )
    model.fit(num_steps=15000, batch_size=256, lr=1e-3, show_plot=False)

    from scipy.stats import norm
    from Inference.utils import evaluate_importance



    X_full, y = Exp1().generate(1000, rho, seed) 
    


    
    #########################################################################################################################################################################
    #                                                                           LOCO/FDFI(SCPI)                                                                                        #
    #########################################################################################################################################################################
    
    from Inference.estimators import LOCOEstimator, SCPI_Flow_Model_Estimator
    
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
    
    

    
    estimator3 = SCPI_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                ),
        flow_model=model
    )

    
    phi_x_loco, se_x_loco = estimator3.importance(X_full, y)
    

    

    phi_x_loco_test = phi_x_loco 

    se_x_loco_test = se_x_loco 
    
    z_score_x_loco = phi_x_loco_test / se_x_loco_test
    
    p_value_x_loco = 1 - norm.cdf(z_score_x_loco)
    rounded_p_value_x_loco = np.round(p_value_x_loco, 3)
    
    print(rounded_p_value_x_loco)
    
    
    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_loco = roc_auc_score(y_true[mask], np.array(phi_0_loco)[mask])
    #auc_score_z_loco = roc_auc_score(y_true[mask], np.array(phi_z_loco)[mask])
    auc_score_x_loco = roc_auc_score(y_true[mask], np.array(phi_x_loco)[mask])

    
    print(f"AUC for phi_0_loco: {auc_score_0_loco:.4f}")
    #print(f"AUC for phi_z_loco: {auc_score_z_loco:.4f}")
    print(f"AUC for phi_x_loco: {auc_score_x_loco:.4f}")

    alpha = 0.05
    
    power_x_loco, type1_x_loco, count_x_loco = evaluate_importance(p_value_x_loco, y_true, alpha)
    #power_z_loco, type1_z_loco, count_z_loco = evaluate_importance(p_value_z_loco, y_true, alpha)
    power_0_loco, type1_0_loco, count_0_loco = evaluate_importance(p_value_0_loco, y_true, alpha)

        
    print(f"[X]    Power: {power_x_loco:.3f}   Type I Error: {type1_x_loco:.3f}   Count: {count_x_loco}")
    #print(f"[Z]    Power: {power_z_loco:.3f}   Type I Error: {type1_z_loco:.3f}   Count: {count_z_loco}")
    print(f"[0]    Power: {power_0_loco:.3f}   Type I Error: {type1_0_loco:.3f}   Count: {count_0_loco}")


    with open(f"{txt_dir_loco}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_loco)):
            txtfile.write(
                          f"{phi_0_loco[i]:.6f}\t{se_0_loco[i]:.6f}\t"
                          f"{phi_x_loco[i]:.6f}\t{se_x_loco[i]:.6f}\n")



    with open(csv_path_loco, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_loco, 6),  round(auc_score_x_loco, 6),
            round(power_0_loco, 6), round(type1_0_loco, 6), int(count_0_loco),
            round(power_x_loco, 6), round(type1_x_loco, 6), int(count_x_loco),
        ])

    


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
#                                                                           CPI/FDFI(CPI)                                                                                        #
#########################################################################################################################################################################

    from Inference.estimators import  CPIEstimator, CPIZ_Flow_Model_Estimator, CPI_Flow_Model_Estimator


    estimator6 = CPIEstimator(
            regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )
    )


    phi_0_cpi, se_0_cpi = estimator6.importance(X_full, y)

    
    phi_0_cpi_test = phi_0_cpi 

    se_0_cpi_test = se_0_cpi 

    z_score_0_cpi = phi_0_cpi_test / se_0_cpi_test
    
    p_value_0_cpi = 1 - norm.cdf(z_score_0_cpi)
    rounded_p_value_0_cpi = np.round(p_value_0_cpi, 3)

    print(rounded_p_value_0_cpi)




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

#---------------------------------------------------------------------------------------#
#                               Power&Type 1 Error&Auc                                  #
#---------------------------------------------------------------------------------------#
    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_cpi = roc_auc_score(y_true[mask], np.array(phi_0_cpi)[mask])
    #auc_score_z_cpi = roc_auc_score(y_true[mask], np.array(phi_z_cpi)[mask])
    auc_score_x_cpi = roc_auc_score(y_true[mask], np.array(phi_x_cpi)[mask])

        
    print(f"AUC for phi_0_cpi: {auc_score_0_cpi:.4f}")
    #print(f"AUC for phi_z_cpi: {auc_score_z_cpi:.4f}")
    print(f"AUC for phi_x_cpi: {auc_score_x_cpi:.4f}")


    alpha = 0.05
    
    power_x_cpi, type1_x_cpi, count_x_cpi = evaluate_importance(p_value_x_cpi, y_true, alpha)
    #power_z_cpi, type1_z_cpi, count_z_cpi = evaluate_importance(p_value_z_cpi, y_true, alpha)
    power_0_cpi, type1_0_cpi, count_0_cpi = evaluate_importance(p_value_0_cpi, y_true, alpha)

        
    print(f"[X]    Power: {power_x_cpi:.3f}   Type I Error: {type1_x_cpi:.3f}   Count: {count_x_cpi}")
    #print(f"[Z]    Power: {power_z_cpi:.3f}   Type I Error: {type1_z_cpi:.3f}   Count: {count_z_cpi}")
    print(f"[0]    Power: {power_0_cpi:.3f}   Type I Error: {type1_0_cpi:.3f}   Count: {count_0_cpi}")
    
    with open(f"{txt_dir_cpi}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_cpi)):
            txtfile.write(
                          f"{phi_0_cpi[i]:.6f}\t{se_0_cpi[i]:.6f}\t"
                          f"{phi_x_cpi[i]:.6f}\t{se_x_cpi[i]:.6f}\n")
            
    with open(csv_path_cpi, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_cpi, 6),  round(auc_score_x_cpi, 6),
            round(power_0_cpi, 6), round(type1_0_cpi, 6), int(count_0_cpi),
            round(power_x_cpi, 6), round(type1_x_cpi, 6), int(count_x_cpi),
        ])




    
    
