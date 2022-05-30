from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import Chem
import numpy as np


def fp_func(mol, fp_type="avalon"):
    """
    calculate fingerprint. if failed, it will return fingerprint of "C"

    Parameters
    ----------
    mol : mol object of rdkit
        mol object of rdkit
    fp_type: string
        type of fingerprint
        
    Returns
    -------
    return : fingerprint object
        
    """
        
    while True:
        try:
            if fp_type == "avalon":
                return GetAvalonFP(mol, nBits=512)
            elif fp_type == "rdkit":
                return Chem.RDKFingerprint(mol, fpSize=2048)
            elif fp_type == "morgan":
                return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            else:
                return GetAvalonFP(mol, nBits=512)
        except:
            mol = Chem.MolFromSmiles('C')


def fp_similarity(sm1, sm2):
    """
    calculate tanimoto similarity of two molecles. if failed, return 10^-5

    Parameters
    ----------
    sm1: string
        smiles
    sm2: string
        smiles        
        
    Returns
    -------
    return : float
        Tanimoto similarity
        
    """
        
    try:
        fp1 = fp_func(Chem.MolFromSmiles(sm1))
        fp2 = fp_func(Chem.MolFromSmiles(sm2))

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    except:
        return 10**-5


"""
descriptor utilities

"""

descriptor_names = [descriptor_name[0]
                    for descriptor_name in Descriptors._descList]
descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(
    descriptor_names)
calculator = descriptor_calculation.CalcDescriptors


def calc_descriptor(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptor = np.array(calculator(mol))
    descriptor = np.nan_to_num(descriptor, nan=0)
    return descriptor


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cos_sim_smiles(sm1, sm2):
    try:
        v1 = calc_descriptor(sm1)
        v2 = calc_descriptor(sm2)
        return cos_sim(v1, v2)
    except:
        return 0
