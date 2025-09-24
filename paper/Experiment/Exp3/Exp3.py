import csv
import os

num_runs = 100 

txt_dir_loco_rf = "Exp3/RF/phi_logs/loco_phi_logs"
csv_path_loco_rf = "Exp3/RF/summary/loco_summary.csv"


txt_dir_cpi_rf = "Exp3/RF/phi_logs/cpi_phi_logs"
csv_path_cpi_rf = "Exp3/RF/summary/cpi_summary.csv"


txt_dir_dfi_rf = "Exp3/RF/phi_logs/dfi_phi_logs"
csv_path_dfi_rf = "Exp3/RF/summary/dfi_summary.csv"

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

txt_dir_loco_ls = "Exp3/Lasso/phi_logs/loco_phi_logs"
csv_path_loco_ls = "Exp3/Lasso/summary/loco_summary.csv"


txt_dir_cpi_ls = "Exp3/Lasso/phi_logs/cpi_phi_logs"
csv_path_cpi_ls = "Exp3/Lasso/summary/cpi_summary.csv"


txt_dir_dfi_ls = "Exp3/Lasso/phi_logs/dfi_phi_logs"
csv_path_dfi_ls = "Exp3/Lasso/summary/dfi_summary.csv"

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

txt_dir_loco_nn = "Exp3/NN/phi_logs/loco_phi_logs"
csv_path_loco_nn = "Exp3/NN/summary/loco_summary.csv"


txt_dir_cpi_nn = "Exp3/NN/phi_logs/cpi_phi_logs"
csv_path_cpi_nn = "Exp3/NN/summary/cpi_summary.csv"


txt_dir_dfi_nn = "Exp3/NN/phi_logs/dfi_phi_logs"
csv_path_dfi_nn = "Exp3/NN/summary/dfi_summary.csv"


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
os.makedirs(txt_dir_loco_rf, exist_ok=True)

os.makedirs(txt_dir_cpi_rf, exist_ok=True)

os.makedirs(txt_dir_dfi_rf, exist_ok=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
os.makedirs(txt_dir_loco_ls, exist_ok=True)

os.makedirs(txt_dir_cpi_ls, exist_ok=True)

os.makedirs(txt_dir_dfi_ls, exist_ok=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
os.makedirs(txt_dir_loco_nn, exist_ok=True)

os.makedirs(txt_dir_cpi_nn, exist_ok=True)

os.makedirs(txt_dir_dfi_nn, exist_ok=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

with open(csv_path_loco_rf, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_loco',  'auc_score_x_loco',
        'power_0_loco', 'type1_0_loco', 'count_0_loco',
        'power_x_loco', 'type1_x_loco', 'count_x_loco',
    ])


with open(csv_path_cpi_rf, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_cpi',  'auc_score_x_cpi',
        'power_0_cpi', 'type1_0_cpi', 'count_0_cpi', 
        'power_x_cpi', 'type1_x_cpi', 'count_x_cpi',
         
    ])   

with open(csv_path_dfi_rf, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_dfi',
        'power_x_dfi', 'type1_x_dfi', 'count_x_dfi',
         
    ])   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
with open(csv_path_loco_ls, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_loco',  'auc_score_x_loco',
        'power_0_loco', 'type1_0_loco', 'count_0_loco',
        'power_x_loco', 'type1_x_loco', 'count_x_loco',
    ])


with open(csv_path_cpi_ls, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_cpi',  'auc_score_x_cpi',
        'power_0_cpi', 'type1_0_cpi', 'count_0_cpi', 
        'power_x_cpi', 'type1_x_cpi', 'count_x_cpi',
         
    ])   

with open(csv_path_dfi_ls, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_dfi',
        'power_x_dfi', 'type1_x_dfi', 'count_x_dfi',
         
    ])   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
with open(csv_path_loco_nn, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_loco',  'auc_score_x_loco',
        'power_0_loco', 'type1_0_loco', 'count_0_loco',
        'power_x_loco', 'type1_x_loco', 'count_x_loco',
    ])


with open(csv_path_cpi_nn, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_cpi',  'auc_score_x_cpi',
        'power_0_cpi', 'type1_0_cpi', 'count_0_cpi', 
        'power_x_cpi', 'type1_x_cpi', 'count_x_cpi',
         
    ])   

with open(csv_path_dfi_nn, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_dfi',
        'power_x_dfi', 'type1_x_dfi', 'count_x_dfi',
         
    ])   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#




for run in range(1, num_runs + 1):
    print(f"\n================ {run} ================\n")
    
    #########################################################################################################################################################################
    #                                                                           Flow Matching                                                                               #
    #########################################################################################################################################################################
    
    import sys
    sys.path.append('ICLR_Flow_Disentangle')
    
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    from Inference.data import Exp1
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    from sklearn.neural_network import MLPRegressor

    
    
    seed = np.random.randint(0, 1000)  
    rho = 0.4
    X_full, y = Exp1().generate(3000, rho, seed)
    
    D=X_full.shape[1]
    
    n_jobs=42

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
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
    
    
    from Inference.estimators import LOCOEstimator, LOCO_Flow_Model_Estimator
    
    from sklearn.ensemble import RandomForestRegressor

    
    #---------------------------------------------------------------------------------------#
    #                                       LOCO_0                                          #
    #---------------------------------------------------------------------------------------#
    
    estimator1 = LOCOEstimator(
    
            regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 )
    
    )
    
    
    phi_0_loco_rf, std_0_loco_rf = estimator1.importance(X_full, y)
    
    
    print("Feature\tLOCO_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_loco_rf, std_0_loco_rf)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"LOCO_0总和: {D* np.mean(phi_0_loco_rf)}")
    
    

    phi_0_loco_test_rf = phi_0_loco_rf 

    std_0_loco_test_rf = std_0_loco_rf 
    
    z_score_0_loco_rf = phi_0_loco_test_rf / std_0_loco_test_rf
    
    p_value_0_loco_rf = 1 - norm.cdf(z_score_0_loco_rf)
    rounded_p_value_0_loco_rf = np.round(p_value_0_loco_rf, 3)
    
    print(rounded_p_value_0_loco_rf)
    
    #---------------------------------------------------------------------------------------#
    #                                       LOCO_X                                          #
    #---------------------------------------------------------------------------------------#
    
    estimator3 = LOCO_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                #random_state=42,
                n_jobs=n_jobs
                ),
        flow_model=model
    )

    
    phi_x_loco_rf, std_x_loco_rf = estimator3.importance(X_full, y)
    
    
    print("Feature\tLOCO_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_loco_rf, std_x_loco_rf)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"LOCO_X总和: {D* np.mean(phi_x_loco_rf)}")
    
    
    

    phi_x_loco_test_rf = phi_x_loco_rf 

    std_x_loco_test_rf = std_x_loco_rf 
    
    z_score_x_loco_rf = phi_x_loco_test_rf / std_x_loco_test_rf
    
    p_value_x_loco_rf = 1 - norm.cdf(z_score_x_loco_rf)
    rounded_p_value_x_loco_rf = np.round(p_value_x_loco_rf, 3)
    
    print(rounded_p_value_x_loco_rf)
    
    
    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_loco = roc_auc_score(y_true[mask], np.array(phi_0_loco_rf)[mask])
    #auc_score_z_loco = roc_auc_score(y_true[mask], np.array(phi_z_loco)[mask])
    auc_score_x_loco = roc_auc_score(y_true[mask], np.array(phi_x_loco_rf)[mask])

    
    print(f"AUC for phi_0_loco_rf: {auc_score_0_loco:.4f}")
    #print(f"AUC for phi_z_loco: {auc_score_z_loco:.4f}")
    print(f"AUC for phi_x_loco_rf: {auc_score_x_loco:.4f}")

    alpha = 0.05
    
    power_x_loco, type1_x_loco, count_x_loco = evaluate_importance(p_value_x_loco_rf, y_true, alpha)
    #power_z_loco, type1_z_loco, count_z_loco = evaluate_importance(p_value_z_loco, y_true, alpha)
    power_0_loco, type1_0_loco, count_0_loco = evaluate_importance(p_value_0_loco_rf, y_true, alpha)

        
    print(f"[X]    Power: {power_x_loco:.3f}   Type I Error: {type1_x_loco:.3f}   Count: {count_x_loco}")
    #print(f"[Z]    Power: {power_z_loco:.3f}   Type I Error: {type1_z_loco:.3f}   Count: {count_z_loco}")
    print(f"[0]    Power: {power_0_loco:.3f}   Type I Error: {type1_0_loco:.3f}   Count: {count_0_loco}")


    with open(f"{txt_dir_loco_rf}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_loco_rf)):
            txtfile.write(
                          f"{phi_0_loco_rf[i]:.6f}\t{std_0_loco_rf[i]:.6f}\t"
                          f"{phi_x_loco_rf[i]:.6f}\t{std_x_loco_rf[i]:.6f}\n")



    with open(csv_path_loco_rf, mode='a', newline='') as f:
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
                #random_state=42,
                n_jobs=n_jobs
                )

    )

    
    phi_x_dfi_rf, std_x_dfi_rf = estimator4.importance(X_full, y)
    
    
    print("Feature\tDFI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_dfi_rf, std_x_dfi_rf)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"DFI_X总和: {D* np.mean(phi_x_dfi_rf)}")
    
    
    
   
    phi_x_dfi_test_rf = phi_x_dfi_rf 

    std_x_dfi_test_rf = std_x_dfi_rf 
    
    z_score_x_dfi_rf = phi_x_dfi_test_rf / std_x_dfi_test_rf
    
    p_value_x_dfi_rf = 1 - norm.cdf(z_score_x_dfi_rf)
    rounded_p_value_x_dfi_rf = np.round(p_value_x_dfi_rf, 3)
    
    print(rounded_p_value_x_dfi_rf)

    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#



    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_x_dfi = roc_auc_score(y_true[mask], np.array(phi_x_dfi_rf)[mask])

    

    print(f"AUC for phi_x_dfi_rf: {auc_score_x_dfi:.4f}")

    alpha = 0.05
    
    power_x_dfi, type1_x_dfi, count_x_dfi = evaluate_importance(p_value_x_dfi_rf, y_true, alpha)


        
    print(f"[X]    Power: {power_x_dfi:.3f}   Type I Error: {type1_x_dfi:.3f}   Count: {count_x_dfi}")



    with open(f"{txt_dir_dfi_rf}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_x\tstd_x\n")
        for i in range(len(phi_x_dfi_rf)):
            txtfile.write(
                          f"{phi_x_dfi_rf[i]:.6f}\t{std_x_dfi_rf[i]:.6f}\n")



    with open(csv_path_dfi_rf, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_x_dfi, 6),
            round(power_x_dfi, 6), round(type1_x_dfi, 6), int(count_x_dfi),
        ])


#########################################################################################################################################################################
#                                                                           CPI                                                                                        #
#########################################################################################################################################################################

    from Inference.estimators import  CPIEstimator, CPI_Flow_Model_Estimator

    #---------------------------------------------------------------------------------------#
    #                                       CPI_0                                           #
    #---------------------------------------------------------------------------------------#

    estimator6 = CPIEstimator(
            regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )
    )

    phi_0_cpi_rf, std_0_cpi_rf = estimator6.importance(X_full, y)

    
    print("Feature\tCPI_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_cpi_rf, std_0_cpi_rf)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"CPI_0总和: {D* np.mean(phi_0_cpi_rf)}")
    
    


    phi_0_cpi_test_rf = phi_0_cpi_rf 

    std_0_cpi_test_rf = std_0_cpi_rf 

    z_score_0_cpi_rf = phi_0_cpi_test_rf / std_0_cpi_test_rf
    
    p_value_0_cpi_rf = 1 - norm.cdf(z_score_0_cpi_rf)
    rounded_p_value_0_cpi_rf = np.round(p_value_0_cpi_rf, 3)

    print(rounded_p_value_0_cpi_rf)



    #---------------------------------------------------------------------------------------#
    #                                       CPI_X                                           #
    #---------------------------------------------------------------------------------------#

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
    
    
    phi_x_cpi_rf, std_x_cpi_rf = estimator8.importance(X_full, y)
    
    
    print("Feature\tCPI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_cpi_rf, std_x_cpi_rf)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"CPI_X总和: {D* np.mean(phi_x_cpi_rf)}")
    
    

    phi_x_cpi_test_rf = phi_x_cpi_rf 

    std_x_cpi_test_rf = std_x_cpi_rf 
    
    z_score_x_cpi_rf = phi_x_cpi_test_rf / std_x_cpi_test_rf
    
    
    p_value_x_cpi_rf = 1 - norm.cdf(z_score_x_cpi_rf)
    rounded_p_value_x_cpi_rf = np.round(p_value_x_cpi_rf, 3)
    
    print(rounded_p_value_x_cpi_rf)

#---------------------------------------------------------------------------------------#
#                               Power&Type 1 Error&Auc                                  #
#---------------------------------------------------------------------------------------#
    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_cpi = roc_auc_score(y_true[mask], np.array(phi_0_cpi_rf)[mask])
    #auc_score_z_cpi = roc_auc_score(y_true[mask], np.array(phi_z_cpi)[mask])
    auc_score_x_cpi = roc_auc_score(y_true[mask], np.array(phi_x_cpi_rf)[mask])

        
    print(f"AUC for phi_0_cpi: {auc_score_0_cpi:.4f}")
    #print(f"AUC for phi_z_cpi: {auc_score_z_cpi:.4f}")
    print(f"AUC for phi_x_cpi: {auc_score_x_cpi:.4f}")


    alpha = 0.05
    
    power_x_cpi, type1_x_cpi, count_x_cpi = evaluate_importance(p_value_x_cpi_rf, y_true, alpha)
    #power_z_cpi, type1_z_cpi, count_z_cpi = evaluate_importance(p_value_z_cpi, y_true, alpha)
    power_0_cpi, type1_0_cpi, count_0_cpi = evaluate_importance(p_value_0_cpi_rf, y_true, alpha)

        
    print(f"[X]    Power: {power_x_cpi:.3f}   Type I Error: {type1_x_cpi:.3f}   Count: {count_x_cpi}")
    #print(f"[Z]    Power: {power_z_cpi:.3f}   Type I Error: {type1_z_cpi:.3f}   Count: {count_z_cpi}")
    print(f"[0]    Power: {power_0_cpi:.3f}   Type I Error: {type1_0_cpi:.3f}   Count: {count_0_cpi}")
    
    with open(f"{txt_dir_cpi_rf}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_cpi_rf)):
            txtfile.write(
                          f"{phi_0_cpi_rf[i]:.6f}\t{std_0_cpi_rf[i]:.6f}\t"
                          f"{phi_x_cpi_rf[i]:.6f}\t{std_x_cpi_rf[i]:.6f}\n")
            
    with open(csv_path_cpi_rf, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_cpi, 6),  round(auc_score_x_cpi, 6),
            round(power_0_cpi, 6), round(type1_0_cpi, 6), int(count_0_cpi),
            round(power_x_cpi, 6), round(type1_x_cpi, 6), int(count_x_cpi),
        ])

    


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#




    
    #---------------------------------------------------------------------------------------#
    #                                       LOCO_0                                          #
    #---------------------------------------------------------------------------------------#
    
    estimator1 = LOCOEstimator(
    
    
    
    regressor = make_pipeline(
        StandardScaler(),                  # Lasso 对特征缩放较敏感
        Lasso(alpha=1e-3, max_iter=10000, tol=1e-4)
    )
    
    )
    
    
    phi_0_loco_ls, std_0_loco_ls = estimator1.importance(X_full, y)
    
    
    print("Feature\tLOCO_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_loco_ls, std_0_loco_ls)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"LOCO_0总和: {D* np.mean(phi_0_loco_ls)}")
    
    

    phi_0_loco_test_ls = phi_0_loco_ls 

    std_0_loco_test_ls = std_0_loco_ls 
    
    z_score_0_loco_ls = phi_0_loco_test_ls / std_0_loco_test_ls
    
    p_value_0_loco_ls = 1 - norm.cdf(z_score_0_loco_ls)
    rounded_p_value_0_loco_ls = np.round(p_value_0_loco_ls, 3)
    
    print(rounded_p_value_0_loco_ls)
    
    

    
    
    #---------------------------------------------------------------------------------------#
    #                                       LOCO_X                                          #
    #---------------------------------------------------------------------------------------#
    
    estimator3 = LOCO_Flow_Model_Estimator(
        regressor = make_pipeline(
                    StandardScaler(),                  # Lasso 对特征缩放较敏感
                    Lasso(alpha=1e-3, max_iter=10000, tol=1e-4)
                ),
        flow_model=model
    )

    
    phi_x_loco_ls, std_x_loco_ls = estimator3.importance(X_full, y)
    
    
    print("Feature\tLOCO_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_loco_ls, std_x_loco_ls)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"LOCO_X总和: {D* np.mean(phi_x_loco_ls)}")
    
    
    

    phi_x_loco_test_ls = phi_x_loco_ls 

    std_x_loco_test_ls = std_x_loco_ls 
    
    z_score_x_loco_ls = phi_x_loco_test_ls / std_x_loco_test_ls
    
    p_value_x_loco_ls = 1 - norm.cdf(z_score_x_loco_ls)
    rounded_p_value_x_loco_ls = np.round(p_value_x_loco_ls, 3)
    
    print(rounded_p_value_x_loco_ls)
    
    
    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_loco = roc_auc_score(y_true[mask], np.array(phi_0_loco_ls)[mask])
    #auc_score_z_loco = roc_auc_score(y_true[mask], np.array(phi_z_loco)[mask])
    auc_score_x_loco = roc_auc_score(y_true[mask], np.array(phi_x_loco_ls)[mask])

    
    print(f"AUC for phi_0_loco: {auc_score_0_loco:.4f}")
    #print(f"AUC for phi_z_loco: {auc_score_z_loco:.4f}")
    print(f"AUC for phi_x_loco: {auc_score_x_loco:.4f}")

    alpha = 0.05
    
    power_x_loco, type1_x_loco, count_x_loco = evaluate_importance(p_value_x_loco_ls, y_true, alpha)
    #power_z_loco, type1_z_loco, count_z_loco = evaluate_importance(p_value_z_loco, y_true, alpha)
    power_0_loco, type1_0_loco, count_0_loco = evaluate_importance(p_value_0_loco_ls, y_true, alpha)

        
    print(f"[X]    Power: {power_x_loco:.3f}   Type I Error: {type1_x_loco:.3f}   Count: {count_x_loco}")
    #print(f"[Z]    Power: {power_z_loco:.3f}   Type I Error: {type1_z_loco:.3f}   Count: {count_z_loco}")
    print(f"[0]    Power: {power_0_loco:.3f}   Type I Error: {type1_0_loco:.3f}   Count: {count_0_loco}")



    with open(f"{txt_dir_loco_ls}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_loco_ls)):
            txtfile.write(
                          f"{phi_0_loco_ls[i]:.6f}\t{std_0_loco_ls[i]:.6f}\t"
                          f"{phi_x_loco_ls[i]:.6f}\t{std_x_loco_ls[i]:.6f}\n")



    with open(csv_path_loco_ls, mode='a', newline='') as f:
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
            regressor = make_pipeline(
            StandardScaler(),                  # Lasso 对特征缩放较敏感
            Lasso(alpha=1e-3, max_iter=10000, tol=1e-4)
        )

    )

    
    phi_x_dfi_ls, std_x_dfi_ls = estimator4.importance(X_full, y)
    
    
    print("Feature\tDFI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_dfi_ls, std_x_dfi_ls)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"DFI_X总和: {D* np.mean(phi_x_dfi_ls)}")
    
    
    
   
    phi_x_dfi_test_ls = phi_x_dfi_ls 

    std_x_dfi_test_ls = std_x_dfi_ls 
    
    z_score_x_dfi_ls = phi_x_dfi_test_ls / std_x_dfi_test_ls
    
    p_value_x_dfi_ls = 1 - norm.cdf(z_score_x_dfi_ls)
    rounded_p_value_x_dfi_ls = np.round(p_value_x_dfi_ls, 3)
    
    print(rounded_p_value_x_dfi_ls)

    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#



    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_x_dfi = roc_auc_score(y_true[mask], np.array(phi_x_dfi_ls)[mask])

    

    print(f"AUC for phi_x_dfi: {auc_score_x_dfi:.4f}")

    alpha = 0.05
    
    power_x_dfi, type1_x_dfi, count_x_dfi = evaluate_importance(p_value_x_dfi_ls, y_true, alpha)

    print(f"[X]    Power: {power_x_dfi:.3f}   Type I Error: {type1_x_dfi:.3f}   Count: {count_x_dfi}")


    with open(f"{txt_dir_dfi_ls}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_x\tstd_x\n")
        for i in range(len(phi_x_dfi_ls)):
            txtfile.write(
                          f"{phi_x_dfi_ls[i]:.6f}\t{std_x_dfi_ls[i]:.6f}\n")



    with open(csv_path_dfi_ls, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_x_dfi, 6),
            round(power_x_dfi, 6), round(type1_x_dfi, 6), int(count_x_dfi),
        ])




#########################################################################################################################################################################
#                                                                           CPI                                                                                        #
#########################################################################################################################################################################

    from Inference.estimators import  CPIEstimator,  CPI_Flow_Model_Estimator

    #---------------------------------------------------------------------------------------#
    #                                       CPI_0                                           #
    #---------------------------------------------------------------------------------------#

    estimator6 = CPIEstimator(
        regressor = make_pipeline(
                StandardScaler(),                  # Lasso 对特征缩放较敏感
                Lasso(alpha=1e-3, max_iter=10000, tol=1e-4)
            )
    )


    phi_0_cpi_ls, std_0_cpi_ls = estimator6.importance(X_full, y)

    
    print("Feature\tCPI_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_cpi_ls, std_0_cpi_ls)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"CPI_0总和: {D* np.mean(phi_0_cpi_ls)}")
    
    


    phi_0_cpi_test_ls = phi_0_cpi_ls 

    std_0_cpi_test_ls = std_0_cpi_ls 

    z_score_0_cpi_ls = phi_0_cpi_test_ls / std_0_cpi_test_ls
    
    p_value_0_cpi_ls = 1 - norm.cdf(z_score_0_cpi_ls)
    rounded_p_value_0_cpi_ls = np.round(p_value_0_cpi_ls, 3)

    print(rounded_p_value_0_cpi_ls)





    #---------------------------------------------------------------------------------------#
    #                                       CPI_X                                           #
    #---------------------------------------------------------------------------------------#

    estimator8 = CPI_Flow_Model_Estimator(
        regressor = make_pipeline(
                StandardScaler(),                  # Lasso 对特征缩放较敏感
                Lasso(alpha=1e-3, max_iter=10000, tol=1e-4)
            ),
        flow_model=model
    
    )
    
    
    phi_x_cpi_ls, std_x_cpi_ls = estimator8.importance(X_full, y)
    
    
    print("Feature\tCPI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_cpi_ls, std_x_cpi_ls)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"CPI_X总和: {D* np.mean(phi_x_cpi_ls)}")
    
    

    phi_x_cpi_test_ls = phi_x_cpi_ls 

    std_x_cpi_test_ls = std_x_cpi_ls 
    
    z_score_x_cpi_ls = phi_x_cpi_test_ls / std_x_cpi_test_ls
    
    
    p_value_x_cpi_ls = 1 - norm.cdf(z_score_x_cpi_ls)
    rounded_p_value_x_cpi_ls = np.round(p_value_x_cpi_ls, 3)
    
    print(rounded_p_value_x_cpi_ls)

#---------------------------------------------------------------------------------------#
#                               Power&Type 1 Error&Auc                                  #
#---------------------------------------------------------------------------------------#
    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_cpi = roc_auc_score(y_true[mask], np.array(phi_0_cpi_ls)[mask])
    #auc_score_z_cpi = roc_auc_score(y_true[mask], np.array(phi_z_cpi)[mask])
    auc_score_x_cpi = roc_auc_score(y_true[mask], np.array(phi_x_cpi_ls)[mask])

        
    print(f"AUC for phi_0_cpi: {auc_score_0_cpi:.4f}")
    #print(f"AUC for phi_z_cpi: {auc_score_z_cpi:.4f}")
    print(f"AUC for phi_x_cpi: {auc_score_x_cpi:.4f}")


    alpha = 0.05
    
    power_x_cpi, type1_x_cpi, count_x_cpi = evaluate_importance(p_value_x_cpi_ls, y_true, alpha)
    #power_z_cpi, type1_z_cpi, count_z_cpi = evaluate_importance(p_value_z_cpi, y_true, alpha)
    power_0_cpi, type1_0_cpi, count_0_cpi = evaluate_importance(p_value_0_cpi_ls, y_true, alpha)

        
    print(f"[X]    Power: {power_x_cpi:.3f}   Type I Error: {type1_x_cpi:.3f}   Count: {count_x_cpi}")
    #print(f"[Z]    Power: {power_z_cpi:.3f}   Type I Error: {type1_z_cpi:.3f}   Count: {count_z_cpi}")
    print(f"[0]    Power: {power_0_cpi:.3f}   Type I Error: {type1_0_cpi:.3f}   Count: {count_0_cpi}")


    with open(f"{txt_dir_cpi_ls}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_cpi_ls)):
            txtfile.write(
                          f"{phi_0_cpi_ls[i]:.6f}\t{std_0_cpi_ls[i]:.6f}\t"
                          f"{phi_x_cpi_ls[i]:.6f}\t{std_x_cpi_ls[i]:.6f}\n")
            
    with open(csv_path_cpi_ls, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_cpi, 6),  round(auc_score_x_cpi, 6),
            round(power_0_cpi, 6), round(type1_0_cpi, 6), int(count_0_cpi),
            round(power_x_cpi, 6), round(type1_x_cpi, 6), int(count_x_cpi),
        ])


    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    

    #########################################################################################################################################################################
    #                                                                           LOCO                                                                                        #
    #########################################################################################################################################################################
    
    from Inference.estimators import LOCOEstimator,  LOCO_Flow_Model_Estimator
    

    
    
    #---------------------------------------------------------------------------------------#
    #                                       LOCO_0                                          #
    #---------------------------------------------------------------------------------------#
    
    estimator1 = LOCOEstimator(
    
    
    
    regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128,64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed
            )
        )
    
    )
    
    
    phi_0_loco_nn, std_0_loco_nn = estimator1.importance(X_full, y)
    
    
    print("Feature\tLOCO_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_loco_nn, std_0_loco_nn)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"LOCO_0总和: {D* np.mean(phi_0_loco_nn)}")
    
    

    phi_0_loco_test_nn = phi_0_loco_nn 

    std_0_loco_test_nn = std_0_loco_nn 
    
    z_score_0_loco_nn = phi_0_loco_test_nn / std_0_loco_test_nn
    
    p_value_0_loco_nn = 1 - norm.cdf(z_score_0_loco_nn)
    rounded_p_value_0_loco_nn = np.round(p_value_0_loco_nn, 3)
    
    print(rounded_p_value_0_loco_nn)
    
    

    
    
    #---------------------------------------------------------------------------------------#
    #                                       LOCO_X                                          #
    #---------------------------------------------------------------------------------------#
    
    estimator3 = LOCO_Flow_Model_Estimator(
     regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128,64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed
            )
        ),
        flow_model=model
    )

    
    phi_x_loco_nn, std_x_loco_nn = estimator3.importance(X_full, y)
    
    
    print("Feature\tLOCO_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_loco_nn, std_x_loco_nn)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"LOCO_X总和: {D* np.mean(phi_x_loco_nn)}")
    
    
    

    phi_x_loco_test_nn = phi_x_loco_nn 

    std_x_loco_test_nn = std_x_loco_nn 
    
    z_score_x_loco_nn = phi_x_loco_test_nn / std_x_loco_test_nn
    
    p_value_x_loco_nn = 1 - norm.cdf(z_score_x_loco_nn)
    rounded_p_value_x_loco_nn = np.round(p_value_x_loco_nn, 3)
    
    print(rounded_p_value_x_loco_nn)
    
    
    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_loco = roc_auc_score(y_true[mask], np.array(phi_0_loco_nn)[mask])
    #auc_score_z_loco = roc_auc_score(y_true[mask], np.array(phi_z_loco)[mask])
    auc_score_x_loco = roc_auc_score(y_true[mask], np.array(phi_x_loco_nn)[mask])

    
    print(f"AUC for phi_0_loco: {auc_score_0_loco:.4f}")
    #print(f"AUC for phi_z_loco: {auc_score_z_loco:.4f}")
    print(f"AUC for phi_x_loco: {auc_score_x_loco:.4f}")

    alpha = 0.05
    
    power_x_loco, type1_x_loco, count_x_loco = evaluate_importance(p_value_x_loco_nn, y_true, alpha)
    #power_z_loco, type1_z_loco, count_z_loco = evaluate_importance(p_value_z_loco, y_true, alpha)
    power_0_loco, type1_0_loco, count_0_loco = evaluate_importance(p_value_0_loco_nn, y_true, alpha)

        
    print(f"[X]    Power: {power_x_loco:.3f}   Type I Error: {type1_x_loco:.3f}   Count: {count_x_loco}")
    #print(f"[Z]    Power: {power_z_loco:.3f}   Type I Error: {type1_z_loco:.3f}   Count: {count_z_loco}")
    print(f"[0]    Power: {power_0_loco:.3f}   Type I Error: {type1_0_loco:.3f}   Count: {count_0_loco}")



    with open(f"{txt_dir_loco_nn}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_loco_nn)):
            txtfile.write(
                          f"{phi_0_loco_nn[i]:.6f}\t{std_0_loco_nn[i]:.6f}\t"
                          f"{phi_x_loco_nn[i]:.6f}\t{std_x_loco_nn[i]:.6f}\n")



    with open(csv_path_loco_nn, mode='a', newline='') as f:
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
        regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128,64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed
            )
        )

    )

    
    phi_x_dfi_nn, std_x_dfi_nn = estimator4.importance(X_full, y)
    
    
    print("Feature\tDFI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_dfi_nn, std_x_dfi_nn)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"DFI_X总和: {D* np.mean(phi_x_dfi_nn)}")
    
    
    
   
    phi_x_dfi_test_nn = phi_x_dfi_nn 

    std_x_dfi_test_nn = std_x_dfi_nn 
    
    z_score_x_dfi_nn = phi_x_dfi_test_nn / std_x_dfi_test_nn
    
    p_value_x_dfi_nn = 1 - norm.cdf(z_score_x_dfi_nn)
    rounded_p_value_x_dfi_nn = np.round(p_value_x_dfi_nn, 3)
    
    print(rounded_p_value_x_dfi_nn)

    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#



    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_x_dfi = roc_auc_score(y_true[mask], np.array(phi_x_dfi_nn)[mask])

    

    print(f"AUC for phi_x_dfi: {auc_score_x_dfi:.4f}")

    alpha = 0.05
    
    power_x_dfi, type1_x_dfi, count_x_dfi = evaluate_importance(p_value_x_dfi_nn, y_true, alpha)


        
    print(f"[X]    Power: {power_x_dfi:.3f}   Type I Error: {type1_x_dfi:.3f}   Count: {count_x_dfi}")


    with open(f"{txt_dir_dfi_nn}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_x\tstd_x\n")
        for i in range(len(phi_x_dfi_nn)):
            txtfile.write(
                          f"{phi_x_dfi_nn[i]:.6f}\t{std_x_dfi_nn[i]:.6f}\n")



    with open(csv_path_dfi_nn, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_x_dfi, 6),
            round(power_x_dfi, 6), round(type1_x_dfi, 6), int(count_x_dfi),
        ])




#########################################################################################################################################################################
#                                                                           CPI                                                                                        #
#########################################################################################################################################################################

    from Inference.estimators import  CPIEstimator,  CPI_Flow_Model_Estimator

    #---------------------------------------------------------------------------------------#
    #                                       CPI_0                                           #
    #---------------------------------------------------------------------------------------#

    estimator6 = CPIEstimator(
    regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128,64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed
            )
        )
    )


    phi_0_cpi_nn, std_0_cpi_nn = estimator6.importance(X_full, y)

    
    print("Feature\tCPI_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_cpi_nn, std_0_cpi_nn)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"CPI_0总和: {D* np.mean(phi_0_cpi_nn)}")
    
    


    phi_0_cpi_test_nn = phi_0_cpi_nn 

    std_0_cpi_test_nn = std_0_cpi_nn 

    z_score_0_cpi_nn = phi_0_cpi_test_nn / std_0_cpi_test_nn
    
    p_value_0_cpi_nn = 1 - norm.cdf(z_score_0_cpi_nn)
    rounded_p_value_0_cpi_nn = np.round(p_value_0_cpi_nn, 3)

    print(rounded_p_value_0_cpi_nn)





    #---------------------------------------------------------------------------------------#
    #                                       CPI_X                                           #
    #---------------------------------------------------------------------------------------#

    estimator8 = CPI_Flow_Model_Estimator(
        regressor = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(128,64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=1000,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=seed
                )
            ),
        flow_model=model
    
    )
    
    
    phi_x_cpi_nn, std_x_cpi_nn = estimator8.importance(X_full, y)
    
    
    print("Feature\tCPI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_cpi_nn, std_x_cpi_nn)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"CPI_X总和: {D* np.mean(phi_x_cpi_nn)}")
    
    

    phi_x_cpi_test_nn = phi_x_cpi_nn 

    std_x_cpi_test_nn = std_x_cpi_nn 
    
    z_score_x_cpi_nn = phi_x_cpi_test_nn / std_x_cpi_test_nn
    
    
    p_value_x_cpi_nn = 1 - norm.cdf(z_score_x_cpi_nn)
    rounded_p_value_x_cpi_nn = np.round(p_value_x_cpi_nn, 3)
    
    print(rounded_p_value_x_cpi_nn)

#---------------------------------------------------------------------------------------#
#                               Power&Type 1 Error&Auc                                  #
#---------------------------------------------------------------------------------------#
    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)


    ignore_idx = list(range(5, 10))


    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False


    auc_score_0_cpi = roc_auc_score(y_true[mask], np.array(phi_0_cpi_nn)[mask])
    #auc_score_z_cpi = roc_auc_score(y_true[mask], np.array(phi_z_cpi)[mask])
    auc_score_x_cpi = roc_auc_score(y_true[mask], np.array(phi_x_cpi_nn)[mask])

        
    print(f"AUC for phi_0_cpi: {auc_score_0_cpi:.4f}")
    #print(f"AUC for phi_z_cpi: {auc_score_z_cpi:.4f}")
    print(f"AUC for phi_x_cpi: {auc_score_x_cpi:.4f}")


    alpha = 0.05
    
    power_x_cpi, type1_x_cpi, count_x_cpi = evaluate_importance(p_value_x_cpi_nn, y_true, alpha)
    #power_z_cpi, type1_z_cpi, count_z_cpi = evaluate_importance(p_value_z_cpi, y_true, alpha)
    power_0_cpi, type1_0_cpi, count_0_cpi = evaluate_importance(p_value_0_cpi_nn, y_true, alpha)

        
    print(f"[X]    Power: {power_x_cpi:.3f}   Type I Error: {type1_x_cpi:.3f}   Count: {count_x_cpi}")
    #print(f"[Z]    Power: {power_z_cpi:.3f}   Type I Error: {type1_z_cpi:.3f}   Count: {count_z_cpi}")
    print(f"[0]    Power: {power_0_cpi:.3f}   Type I Error: {type1_0_cpi:.3f}   Count: {count_0_cpi}")


    with open(f"{txt_dir_cpi_nn}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tstd_0\tphi_x\tstd_x\n")
        for i in range(len(phi_x_cpi_nn)):
            txtfile.write(
                          f"{phi_0_cpi_nn[i]:.6f}\t{std_0_cpi_nn[i]:.6f}\t"
                          f"{phi_x_cpi_nn[i]:.6f}\t{std_x_cpi_nn[i]:.6f}\n")
            
    with open(csv_path_cpi_nn, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_cpi, 6),  round(auc_score_x_cpi, 6),
            round(power_0_cpi, 6), round(type1_0_cpi, 6), int(count_0_cpi),
            round(power_x_cpi, 6), round(type1_x_cpi, 6), int(count_x_cpi),
        ])
    



