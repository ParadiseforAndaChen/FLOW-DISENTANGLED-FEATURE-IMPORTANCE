# FLOW-DISENTANGLED-FEATURE-IMPORTANCE
We use three real-world open-source datasets in our experiments: **Pima Indians Diabetes**, **Cardiotocography (CTG)**, and **MicroMass**. All of them are publicly available and can be downloaded from the following links:  

- [Pima Indians Diabetes(PID)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- [Cardiotocography (CTG)](https://archive.ics.uci.edu/dataset/193/cardiotocography)  
- [MicroMass](https://archive.ics.uci.edu/dataset/253/micromass)  

## Description
### Description of folders in **paper**
1. **Exp1**: Implementation of the experiments in Section 4.1.  
   Run `Exp1.py` to evaluate different combinations of sample size and $\rho$.  
   Use `Exp1_plot.py` to generate the corresponding figures.

2. **Exp2**: Implementation of the experiments in Section 4.2.  
   Run `Exp2.py` with the provided settings to reproduce the benchmark comparisons on complex feature distributions.  
   Use `Exp2_plot.py` to generate the corresponding figures.

3. **Exp3**: Implementation of the experiments in Appendix E.1.  
   Run `Exp3.py` with the provided settings to reproduce the benchmark comparisons with different predictors.  
   Use `Exp3_plot.py` to generate the corresponding figures.

4. **Real_data/CTG**: Implementation of the experiments in Section 4.3 and Appendix E.3.1.  
   Run `CTG.ipynb` to execute the experiments and obtain the corresponding results and figures.

5. **Exp4**: Implementation of the experiments in Appendix D.3.1.  
   Run `Correction.ipynb` to generate the results.

6. **Exp5**: Implementation of the experiments in Appendix D.2.  
   Run `MMD_combination.py` and `MMD_sample_size.py` to generate the results.

7. **Exp6**: Implementation of the experiments in Appendix D.4.  
   Run `computation_time.py` and `computation_time_shapley.py` to generate the results for different methods.  
   Use `Exp6_plot.py` to generate the corresponding figures.

9. **Exp7**: Implementation of the experiments in Appendix E.2.  
   Run `map.py` to generate the result and the figure.



