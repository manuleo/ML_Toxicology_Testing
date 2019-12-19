# Toxicological testing through Machine Learning

## Introduction
In our project we use a subset of the ECOTOX database and we use **k-Nearest Neighbors (k-NN)** and **Matrix Factorization** algorithms to predict the effect of untested chemicals on tested species and vice versa. 
Aim is to predict the **sensibility** of a particular organism with respect to a given chemical. Specifically, by setting thresholds on the concentration needed to kill an half of the population we give a *score* to each pair species-chemical (under certain testing condition) and predict this score.

## Prerequites
### Basic prerequisites
The minimum requirements for the `run.py` are:
- `Python` (tested on version **_3.6_**)
- [pip](https://pip.pypa.io/en/stable/) (tested on version *19.3.1*) (For package installation, if needed)
- `Anaconda` (Test on version *3.18.8*) (More information about Anaconda installation on your OS [here](https://docs.anaconda.com/anaconda/install/)) (For package installation)
- `numpy` (tested on version *1.13.3*)
- `scikit-learn` (tested on version *0.22.0*)
- `pandas` (tested on version *0.25.3*)

**NOTE**: with this configuration, the `run.py` will run **ONLY** with **KNN algorithm and no preprocessing**

### Preprocessing prerequisites 
To run the preprocessing phase the : `rdkit` (Tested on version *2017.09.1*) package is needed.  

To install `rdkit` in your environment use the command
```bash/CMD
conda install -c rdkit rdkit
```
*Note*: the `run.py` will work also without `rdkit` if no preprocessing is used. An already preprocessed dataset will be used.

### Matrix Factorization prerequisites (full installation)
To run the Matrix Factorization algorithm, the `turicreate` library is mandatory. We tested on version *6.0*. Documentation in the [official website](https://github.com/apple/turicreate).  
**Important: `turicreate` works only with Ubuntu or MacOS and Python up to 3.6** (tested on *3.6*). To use it with Windows, a Windows Subsystem Linux (WSL) with conda installed inside is needed. For more information on its installation, check [this tutorial](https://github.com/kapsakcj/win10-linux-conda-how-to).  
To install `turicreate`, use pip. The library requires `mxnet` to be used (our tests confirm that using the version *1.1.0* is the most stable one, so we recommend it). We recommend to create a new environment with Python 3.6 to avoid any conflict, following this steps:

```bash
conda create -n mltox python=3.6 pip numpy scikit-learn pandas
conda activate mltox
pip install --upgrade pip
pip install -U turicreate
pip install -U mxnet==1.1.0
conda install -c rdkit rdkit
```
(The last step is needed if you want a full installation, able to run also the preprocessing)

## Usage instruction
1. Open CMD/Bash
2. Activate the environment with needed packages installed
3. Move to the root folder of the unzipped zip, where the `run.py` is located
4. Execute the command ```python run.py``` with one or more of the following arguments:
```
Mandatory arguments:
  - Possible algorithms:
      Select either one of the two possible algorithms:
       -knn: Run KNN algorithm
       -mf: Run Matrix Factorization algorithm
  - Classification type:
      Select either one of the two possible classification:
        -singleclass: Execute single class classification
        -multiclass: Execute multiclass classification
Optional arguments:    
  -h, --help: show arguments help message and exit
  -preproc:  Execute all the preprocessing steps from the raw dataset. If not set, an already preprocessed dataset will be loaded.
  -feature_sel: Run feature selection on the chosen algorithm. If not set, already selected features will be used.   
  -cross: Run Cross Validation on the chosen algorithm. If not set, already selected best parameters will be used.  
```
Notes about durations:
- The CV algorithm requires about 9 hours for the KNN and about 16 hours for the MF
- The feature_selection algorithm requires 2 hour for the KNN and 1 hour for the MF

(Considering an *Intel Core i7, 3.2 Ghz, 16 GB or RAM*)

Note about requirements: The KNN algorithm requires **from 4 to 6 GB** of free RAM

## Folder structure
```
    .
    ├── data 
    |   ├── processed                          # Already preprocessed data directly usable
    |   |     ├── final_db_processed.csv          
    │   |     └── cas_to_smiles.csv  
    │   └── raw                                # Raw data (need preprocessing)
    |        ├── results.txt 
    |        ├── species.txt 
    │        └── tests.txt 
    ├── output 
    |    ├── knn                               # Folder to store KNN results
    |    └── mf                                # Folder to store MF results
    |    
    ├── src                                    # Source files
    |    ├── knn                               # KNN algorithm helpers and main
    |    |    ├── helper_knn.py          
    │    |    └── main_knn.py
    |    ├── mf                                # MF algorithm helpers and main
    |    |    ├── helper_mf.py          
    │    |    └── main_mf.py
    |    ├── preprocessing                     # Preprocessing algorithm helpers and main
    |    |    ├── helper_preprocessing.py          
    │    |    └── main_preprocessing.py
    |    └── general_helper.py                 # General helper for both algorithms
    ├── run.py                                 # Main entry point for the algorithms
    └── README.md

```

## Code reproducibility
Even if all the random seeds are set in the `run.py`, `turicreate` algorithm to run Matrix Factorization always has a bit of randomness inside the algorithm definition, as state in the [official documentation](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.factorization_recommender.create.html#turicreate.recommender.factorization_recommender.create) (`random_seed` parameter).  
For this reason, the reader is advised that **different runs with the same parameters, and also different runs of CV can produce slightly different results**.  
The KNN algorithm is instead fully reproducible.

## Authors
- Manuel Leone
- Gabriele Macchi
- Marco Vicentini  

*Mentor*: Marco Baity-Jesi
