# FLOW-DISENTANGLED-FEATURE-IMPORTANCE
We use these real-world open-source datasets to illustrate how our method can be applied in practice and to demonstrate the complete workflow of our implementation: **Pima Indians Diabetes(PID)**, **Cardiotocography (CTG)**, **MicroMass**, **TCGA-PANCAN-HiSeq bulk RNA-seq** , **Human single-cell RNA-seq**, **Default of Credit Card Clients (DCCC)** and **Superconductivity**   . All of them are publicly available and can be downloaded from the following links:  

- [Pima Indians Diabetes(PID)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- [Cardiotocography (CTG)](https://archive.ics.uci.edu/dataset/193/cardiotocography)  
- [MicroMass](https://archive.ics.uci.edu/dataset/253/micromass)
- [TCGA-PANCAN-HiSeq bulk RNA-seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) 
- [Human single-cell RNA-seq](https://www.pnas.org/doi/10.1073/pnas.2104683118)
- [Default of Credit Card Clients (DCCC)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Superconductivity](https://openml.org/search?type=data&status=active&id=44148)
## Description
### Description of folders in **paper**
1. **Exp1**: Implementation of the experiments in Section 4.1.  
   Run `Exp1.py` to evaluate different combinations of sample size and $\rho$.  
   Use `Exp1_plot.py` to generate the corresponding figures.

2. **Exp2**: Implementation of the experiments in Appendix E.1.3.  
   Run `Exp2.py` with the provided settings to reproduce the benchmark comparisons on complex feature distributions.  
   Use `Exp2_plot.py` to generate the corresponding figures.

3. **Exp3**: Implementation of the experiments in Appendix E.1.1.  
   Run `Exp3.py` with the provided settings to reproduce the benchmark comparisons with different predictors.  
   Use `Exp3_plot.py` to generate the corresponding figures.

4. **Real_data/CTG**: Implementation of the experiments in Section 4.3 and Appendix E.2.1.  
   Run `CTG.ipynb` to execute the experiments and obtain the corresponding results and figures.

5. **Real_data/PID**: Implementation of the experiments in Appendix E.2.2.  
   Run `PID.ipynb` to execute the experiments and obtain the corresponding results and figures.

6. **Real_data/MicroMass**: Implementation of the experiments in Appendix E.2.3.  
   Run `MicroMass.ipynb` to execute the experiments and obtain the corresponding results and figures.

7. **Exp4**: Implementation of the experiments in Appendix D.3.1.  
   Run `Correction.ipynb` to generate the results.

8. **Exp5**: Implementation of the experiments in Appendix D.2.  
   Run `MMD_combination.py` and `MMD_sample_size.py` to generate the results.

9. **Exp6**: Implementation of the experiments in Appendix D.4.  
   Run `computation_time.py` and `computation_time_shapley.py` to generate the results for different methods.  
   Use `Exp6_plot.py` to generate the corresponding figures.

10. **Exp7**: Implementation of the experiments in Appendix E.1.2.  
   Run `map.py` to generate the result and the figure.

11. **Real_data/TCGA**: Implementation of the experiments in Section 4.2 and Appendix E.2.5.  
   Run `TCGA.ipynb` to execute the experiments and obtain the corresponding results.

12. **Real_data/Human**: Implementation of the experiments in Section 4.2 and Appendix E.2.5.  
   Run `Human.ipynb` to execute the experiments and obtain the corresponding results.

13. **Real_data/DCCC**: Implementation of the experiments in Appendix E.2.4.  
   Run `DCCC.ipynb` to execute the experiments and obtain the corresponding results.

14. **Real_data/Superconduct**: Implementation of the experiments in Appendix E.2.4.  
   Run `Superconduct.ipynb` to execute the experiments and obtain the corresponding results.

