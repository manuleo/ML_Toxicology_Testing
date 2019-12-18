from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDConfig
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdchem

def atom_number(smile):
    return sum(1 for c in smile if c.isupper())

def alone_atom_number(s):
    return s.count('[') 
    
def bonds_number(smile):
    m = Chem.MolFromSmiles(smile)
    try:
        return rdchem.Mol.GetNumBonds(m)
    except:
        return 'NaN'
    
def ring_number(smile):
    m = Chem.MolFromSmiles(smile)
    try:
        f = rdchem.Mol.GetRingInfo(m)
        return f.NumRings()
    except:
        return 'NaN'

def Mol(smile):
    smile = str(smile)
    try:
        m = Chem.MolFromSmiles(smile)
        return Descriptors.MolWt(m)
    except:
        return 'NaN'    
    
def MorganDensity(smile):
    smile = str(smile)
    m = Chem.MolFromSmiles(smile)
    try:
        return Descriptors.FpDensityMorgan1(m)
    except:
        return 'NaN'

def LogP(smile):
    smile = str(smile)
    try:
        m = Chem.MolFromSmiles(smile)
        return Crippen.MolLogP(m)
    except:
        return 'NaN'    
    
def to_cas(num):
    s = str(num)
    s = s[:-3]+ '-' + s[-3:-1] +'-' + s[-1]
    return s

def count_doubleBond(s):
    return s.count('=') 

def count_tripleBond(s):
    return s.count('#') 