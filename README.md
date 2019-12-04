# ML_Project
Enhancing Toxicological Testing through Machine Learning

# Repo structure
- data_analysis.ipynb: all the preprocessing (find feature, extract them, first possible prepreocesses)
- baseline_model.ipynbs: simple classification and regression without smiles
- classification_with_smiles2.ipynb: preprocessing on smiles and classification on data with them
- ridge_polynomial_regression.ipynb: ridge (with polynomial expanded features) using data with smiles
- find_smiles.ipynb: show the R code to extract smiles from data
- matrix_fact (folder):
  - Factorization_preparation.ipynb: preprocess data (find score, build a sparse matrix) to be used for recommender system
  - Factorization_surprise.ipynb: factorization using surprise library
  - Factorization_Lab.ipynb: factorization using raw functions done in the homeworks (not good results, better consider surprise case).
- smiles_proc.py: functions to extrapolate data from smiles

# Authors
M. Leone, G. Macchi, M. Vicentini, M. Baity-Jesi, K. Schirmer
