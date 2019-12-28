# -*- coding: utf-8 -*-
'''Helper functions for the preprocessing phase on the dataset'''

import pandas as pd
import numpy as np
from .smiles_proc import *
from sklearn.preprocessing import MinMaxScaler

# Turning off helpless warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)


def load_original_data(DATA_RESULTS_PATH, DATA_TEST_PATH, DATA_SPECIES_PATH):
    '''Load raw ECOTOX dataset
        Inputs:
            - DATA_RESULTS_PATH (string): Path to the results.txt dataset.
            - DATA_TEST_PATH (string): Path to the test.txt dataset.
            - DATA_SPECIES_PATH (string): Path to the species.txt dataset.
        Outputs:
            - results (Pandas dataframe): dataframe with test results loaded.
            - test (Pandas dataframe): dataframe with test information loaded.
            - species (Pandas dataframe): dataframe with species information loaded.'''
    
    # Loading datasets
    results = pd.read_csv(DATA_RESULTS_PATH, sep='\|', engine='python')
    print("Results dataset loaded!")
    
    test = pd.read_csv(DATA_TEST_PATH, sep='\|', engine='python')
    print("Test dataset loaded!")
    
    species = pd.read_csv(DATA_SPECIES_PATH, sep='\|', engine='python')
    print("Species dataset loaded!")
    
    # Return
    return results, test, species

def prefilter(results, test, species):
    '''Filter the original datasets by retaiing only LC50 effect on Mortality, no embryons, only Fish.
       Merge the information in only a single dataset.
        Inputs:
            - results (Pandas dataframe): dataframe with test results.
            - test (Pandas dataframe): dataframe with test information.
            - species (Pandas dataframe): dataframe with species information.
        Outputs:
            - results_prefilter (Pandas dataframe): dataframe prefiltered with information from all the tree used as inputs.'''
    
    # Retaining only LC50 or EC50
    res_50 = results[(results.endpoint.str.contains("LC50")) | (results.endpoint.str.contains("EC50"))]
    
    # Only the ones inside mortality group
    res_50_mor = res_50[res_50.effect.str.contains("MOR")]
    
    # No embrions in the test
    test_no_EM = test[test.organism_lifestage != "EM"]
    
    # No Fish
    species = species[~species.ecotox_group.isnull()]
    species_fish = species[species.ecotox_group.str.contains("Fish")]

    # Merge
    test_fish_only = test.merge(species_fish, on="species_number")
    
    # Produce filtered dataset
    results_prefilter = res_50_mor.merge(test_fish_only, on = "test_id")
    
    return results_prefilter


def filter_best_features(results_prefilter):
    '''Retain only usable features from the ECOTOX dataset and impute them (see report for detail).
    Inputs:
        - results_prefilter (Pandas dataframe): dataframe with all the features.
    Outputs:
        - results_filter (Pandas dataframe): dataframe with only best features correctly imputed.'''
    
    # Considering the best features selected during the data analysis (Read report for more details)
    columns_final = [
                    'exposure_type',
                    'obs_duration_unit',
                    'obs_duration_mean',
                    'conc1_type',
                    'conc1_mean',
                    'conc1_unit',
                    'test_cas',
                    'class',
                    'tax_order',
                    'family',
                    'genus',
                    'species']
    
    print("The best features from the ECOTOX dataset are:", columns_final)
    
    results_filter = results_prefilter.copy()
    results_filter = results_filter[columns_final]
    
    # Imputing phase
    print ("Imputing the concentration labels/features...")
    results_filter = _impute_conc(results_filter, results_prefilter)
    print ("Concentrarion labels/features imputed")
    
    print("Imputing exposure type feature...")
    results_filter = _impute_exp_type(results_filter)
    print ("Exposure type imputed")
    
    print("Imputing observation duration feature...")
    results_filter = _impute_obs_duration(results_filter)
    print ("Observation duration imputed")
    
    print("Imputing species characteristics features...")
    results_filter = _impute_species(results_filter)
    print ("Species characteristics imputed")
    
    # Check imputing worked well
    tot_na = results_filter.isnull().sum().sum()
    if (tot_na==0):
        print("No missing values found in the dataset! Imputing worked")
    else:
        print("Something went wrong during the ECOTOX imputing. Please check.")
        raise Exception('Wrong imputing')
    
    return results_filter
    
    
def _impute_conc(results_filter, results_prefilter):
    '''Impute the columns relative to concentration inside the dataset.
    Inputs:
        - results_filter (Pandas dataframe): dataframe filtered on the best features.
        - results_prefilter (Pandas dataframe): dataframe with all the features (used for imputing).
    Outputs:
        - results_filter_conc (Pandas dataframe): dataframe filtered on the best features with imputed concentrations.'''  
    
    print("Imputing on the LABEL: concentration value...")
    # Filtering on the top 10 unit for concentrations (See report for details)
    conc_unit_counts = results_filter.conc1_unit.value_counts(dropna=False)
    conc_unit_best = conc_unit_counts.head(10).index.values
    
    results_filter_conc = results_filter.copy() 
    results_filter_conc = results_filter_conc[results_filter_conc.conc1_unit.isin(conc_unit_best)]
    
    # Imputing AI (Active Ingredient unit of measure)
    results_filter_conc.conc1_unit = results_filter_conc.conc1_unit.apply(lambda x: x.replace("AI ", "") if 'AI' in x else x)
    
    # Imputing/Removing all the missing values
    # Finding the index where conc1_mean (used as best) is NR and try to impute with conc2 (it has a lot of NaN, but it can be useful)
    conc1_NR_index = results_filter_conc[results_filter_conc.conc1_mean =="NR"].index
    list_conc = ["conc2_mean", "conc2_unit"]
    other_conc = results_prefilter[list_conc].copy()
    other_conc = other_conc.loc[conc1_NR_index]
    other_conc = other_conc.dropna()
    # Using the results from this imputing
    conc2_impute_index = other_conc[other_conc.conc2_mean!="NR"].index
    results_filter_conc.conc1_unit.loc[conc2_impute_index] = other_conc.conc2_unit.loc[conc2_impute_index]
    results_filter_conc.conc1_mean.loc[conc2_impute_index] = other_conc.conc2_mean.loc[conc2_impute_index]
    # Dropping all the non-imputable
    drop_NR_mean = results_filter_conc[results_filter_conc.conc1_mean =="NR"].index
    results_filter_conc.drop(drop_NR_mean, inplace=True)
    
    # Strange values to fix: some values with * at the end and a strange value > 100000 (dropped as doesn't mean much)
    results_filter_conc.conc1_mean = results_filter_conc.conc1_mean.apply(lambda x: x.replace("*", "") if "*" in x else x)
    drop = results_filter_conc[results_filter_conc.conc1_mean ==">100000"].index
    results_filter_conc.drop(drop, inplace=True)
    
    # Converting all the possible measure unit in mg/L
    results_filter_conc.conc1_mean = results_filter_conc.conc1_mean.astype(float)
    # Converting ug/L and ppb (1 ppb = 1 ug/L = 10^-3 mg/L)
    results_filter_conc.conc1_mean = np.where((results_filter_conc['conc1_unit'] == "ug/L") | (results_filter_conc['conc1_unit'] == "ppb"), \
                                        results_filter_conc.conc1_mean/1000, results_filter_conc.conc1_mean)
    # Converting ng/L (1 ng/L = 10^-6 mg/L)
    results_filter_conc.conc1_mean = np.where(results_filter_conc['conc1_unit'] == "ng/L", \
                                              results_filter_conc.conc1_mean*(10**(-6)), results_filter_conc.conc1_mean)
    # Merging concentrations
    results_filter_conc.conc1_unit = results_filter_conc.conc1_unit.apply(lambda x: "mg/L" if x in ["ug/L", "ppm", "ppb", "ng/L"] else x)
    # Dropping the non convertible units (ul/L and uM)
    drop_concs = results_filter_conc[results_filter_conc.conc1_unit.str.contains("u")].index
    results_filter_conc.drop(drop_concs, inplace=True)

    # Dropping no longer needed column of unit
    results_filter_conc.drop(columns="conc1_unit", inplace=True)
    print("Imputing on the LABEL: concentration value finished!")
    
    # Drop concentrations with 0 as value (helpless)
    drop0 = results_filter_conc[results_filter_conc.conc1_mean==0].index
    results_filter_conc.drop(drop0, inplace=True)
    
    # Dropping value on concentration type
    print("Dropping missing value on concentration type feature...")
    drop_conc1_type = results_filter_conc[(results_filter_conc.conc1_type == "NC")|(results_filter_conc.conc1_type == "NR")].index
    results_filter_conc.drop(drop_conc1_type, inplace=True)  
    
    return results_filter_conc


def _impute_exp_type(results_filter):
    '''Impute the columns relative to exposure type inside the dataset.
    Inputs:
        - results_filter (Pandas dataframe): dataframe filtered on the best features.
    Outputs:
        - results_filter_exp_type (Pandas dataframe): dataframe filtered on the best features with imputed exposure type.'''
    
    results_filter_exp_type = results_filter.copy() 
    
    # Removing value with a / in the end
    results_filter_exp_type.exposure_type = results_filter_exp_type.exposure_type.apply(lambda x: x.replace("/", ""))
    
    # Converting the NR in AQUA (We know Fish are in water if doesn't specified)
    results_filter_exp_type.exposure_type = results_filter_exp_type.exposure_type.apply(lambda x: 'AQUA' if 'NR' in x else x)
    
    return results_filter_exp_type


def _impute_obs_duration(results_filter):
    '''Impute the columns relative to observation duration inside the dataset.
    Inputs:
        - results_filter (Pandas dataframe): dataframe filtered on the best features.
    Outputs:
        - results_filter_obs_dur (Pandas dataframe): dataframe filtered on the best features with imputed observation duration.'''
    
    results_filter_obs_dur = results_filter.copy() 
    
    # Dropping the missing values in the observation duration
    drop_NR_dur_unit = results_filter_obs_dur[results_filter_obs_dur.obs_duration_unit =="NR"].index
    results_filter_obs_dur.drop(drop_NR_dur_unit, inplace=True)
    
    # Filtering on the useful observation unit (covering at most 96 hours)
    good_obs_unit = ["h", "d", "mi", "wk"] # week also considered in case of fraction of week
    results_filter_obs_dur = results_filter_obs_dur[results_filter_obs_dur.obs_duration_unit.isin(good_obs_unit)] 
    
    # Drop missing unit in observation duration unit
    drop_NR_obs_mean = results_filter_obs_dur[results_filter_obs_dur.obs_duration_mean =="NR"].index
    results_filter_obs_dur.drop(drop_NR_obs_mean, inplace=True)
    
    # Converting all duration in hours
    results_filter_obs_dur.obs_duration_mean = results_filter_obs_dur.obs_duration_mean.astype(float)
    results_filter_obs_dur.obs_duration_mean = np.where(results_filter_obs_dur['obs_duration_mean'] == "d", \
                                   results_filter_obs_dur.obs_duration_mean*24, results_filter_obs_dur.obs_duration_mean)
    results_filter_obs_dur.obs_duration_mean = np.where(results_filter_obs_dur['obs_duration_mean'] == "mi", \
                                    results_filter_obs_dur.obs_duration_mean/60, results_filter_obs_dur.obs_duration_mean)
    results_filter_obs_dur.obs_duration_mean = np.where(results_filter_obs_dur['obs_duration_mean'] == "wk", \
                                    results_filter_obs_dur.obs_duration_mean*7*24, results_filter_obs_dur.obs_duration_mean)
    results_filter_obs_dur.obs_duration_mean = np.where(results_filter_obs_dur['obs_duration_mean'] == "mo", \
                                   results_filter_obs_dur.obs_duration_mean*30*24, results_filter_obs_dur.obs_duration_mean)
    
    # Drop no longer needed column unit
    results_filter_obs_dur = results_filter_obs_dur.drop(columns="obs_duration_unit")
    
    # Retaining only durations 24, 48, 72, 96 hours
    results_filter_obs_dur = results_filter_obs_dur[results_filter_obs_dur.obs_duration_mean.isin([24, 48, 72, 96])]
    
    return results_filter_obs_dur


def _impute_species(results_filter):
    '''Impute the columns relative to species inside the dataset.
    Inputs:
        - results_filter (Pandas dataframe): dataframe filtered on the best features.
    Outputs:
        - results_filter_exp_type (Pandas dataframe): dataframe filtered on the best features with imputed species features.'''
    
    results_filter_species = results_filter.copy() 
    
    # Dropping missing values relative to species (same values are missing for genus)
    drop_species = results_filter_species[results_filter_species.species.isnull()].index
    results_filter_species.drop(drop_species, inplace=True)
    
    # Dropping missing values relative to family
    drop_family = results_filter_species[results_filter_species.family.isnull()].index
    results_filter_species.drop(drop_family, inplace=True)
    
    return results_filter_species

def to_cas(cas_num):
    '''Convert cas in integer format to XXXXX-XX-X used for SMILES matching.
    Inputs:
        - cas_num (int): cas as integer
    Outputs:
        - cas (string): cas as formatted string'''
    cas = str(cas_num)
    cas = cas[:-3]+ '-' + cas[-3:-1] +'-' + cas[-1]
    return cas

def add_smiles_features(results_best_col, DATA_CAS_TO_SMILES_PATH):
    '''Add SMILES to the filtered ECOTOX dataset.
    Inputs:
        - results_best_col (Pandas dataframe): dataframe filtered with the selected best features.
        - DATA_CAS_TO_SMILES_PATH (string): path to .csv file containing information to match the CAS number to the SMILES
    Outputs:
        - base_db (Pandas dataframe): dataframe with SMILES features added'''
    
    # Loading cas to smiles dataset
    smiles = pd.read_csv(DATA_CAS_TO_SMILES_PATH, error_bad_lines=False).drop(columns="Unnamed: 0")
    smiles = smiles.rename(columns={'cas':'test_cas'})
    
    # Merging ECOTOX dataset with cas to smiles
    base_db = results_best_col.copy()
    base_db['test_cas'] = base_db['test_cas'].apply(to_cas)
    base_db = pd.merge(base_db, smiles, on='test_cas')
    
    # Drop null smiles
    null_smiles = base_db[base_db['smiles'].isnull()].index.tolist()
    base_db = base_db.drop(null_smiles,axis=0)
    
    # Computing new features from SMILES
    print("Computing new features using SMILES...")
    
    print("Finding atom number...")
    base_db['atom_number'] = base_db['smiles'].apply(atom_number)
    
    print("Finding number of alone atoms...")
    base_db['alone_atom_number'] = base_db['smiles'].apply(alone_atom_number)

    print("Finding single bounds number...")
    base_db['bonds_number'] = base_db['smiles'].apply(bonds_number)
    
    print("Finding double bounds number...")
    base_db['doubleBond'] = base_db['smiles'].apply(count_doubleBond)
    
    print("Finding triple bounds number...")
    base_db['tripleBond'] = base_db['smiles'].apply(count_tripleBond)
    
    print("Finding ring number...")
    base_db['ring_number'] = base_db['smiles'].apply(ring_number)
    
    print("Finding mol number...")
    base_db['Mol'] = base_db['smiles'].apply(Mol)
    
    print("Finding morgan density...")
    base_db['MorganDensity'] = base_db['smiles'].apply(MorganDensity)
    
    print("Finding partition number (LogP)...")
    base_db['LogP'] = base_db['smiles'].apply(LogP)
    
    # Drop rows with missing SMILES features (if a SMILES feature is missing, all are missed for the same row)
    to_drop_smiles = base_db[base_db.bonds_number=="NaN"].index
    base_db = base_db.drop(to_drop_smiles,axis=0)
    
    # Drop no longer needed SMILES
    base_db.drop(columns="smiles", inplace=True)
    
    # Converting columns into float
    base_db['bonds_number'] = base_db['bonds_number'].apply(lambda x: float(x))
    base_db['ring_number'] = base_db['ring_number'].apply(lambda x: float(x))
    base_db['Mol'] = base_db['Mol'].apply(lambda x: float(x))
    base_db['MorganDensity'] = base_db['MorganDensity'].apply(lambda x: float(x))
    base_db['LogP'] = base_db['LogP'].apply(lambda x: float(x))

    return base_db


def process_smiles_features(base_db):
    '''Preprocess SMILES features (numerical).
    Inputs:
        - base_db (Pandas dataframe): dataset with ECOTOX and SMILES features.
    Outputs:
        - base_db_processed (Pandas dataframe): dataframe with SMILES numerical features processed.'''
    
    # Applying log transformation + MinMax on all the numerical features except MorganDensity and LogP
    base_db_processed = base_db.copy()
    
    # Bounds number
    base_db_processed.bonds_number = base_db_processed.bonds_number.apply(lambda x: np.log1p(x))
    # MinMax scale
    minmax = MinMaxScaler()
    minmax.fit(base_db_processed[["bonds_number"]])
    base_db_processed[["bonds_number"]] = minmax.transform(base_db_processed[["bonds_number"]])
    
    # Atom number
    base_db_processed.atom_number = base_db_processed.atom_number.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(base_db_processed[["atom_number"]])
    base_db_processed[["atom_number"]] = minmax.transform(base_db_processed[["atom_number"]])
    
    # Mol Quantity
    base_db_processed.Mol = base_db_processed.Mol.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(base_db_processed[["Mol"]])
    base_db_processed[["Mol"]] = minmax.transform(base_db_processed[["Mol"]])
    
    return base_db_processed


def repeated_exp(base_db_processed):
    '''Merge repeated experiments to only one label by taking the median of the experiments results
    Inputs:
        - base_db_processed (Pandas dataframe): dataset with ECOTOX and SMILES features.
    Outputs:
        - final_db (Pandas dataframe): Final dataset ready to be used for the next implementations, with except for the categorical features.'''

    # Save species information in a separate dataset
    db_species = base_db_processed[["species", 'class', 'tax_order', 'family', 'genus']]
    db_species = db_species.groupby("species").first()
    
    # Merge experiments and take the median
    final_db = base_db_processed.groupby(by=["test_cas", "species", "conc1_type", "exposure_type", "obs_duration_mean"]).agg("median").reset_index()
    
    # Add again experiments information
    final_db = final_db.merge(db_species, on="species")
    
    return final_db

    
    
