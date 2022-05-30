from rdkit import Chem
from rdkit.Chem import Descriptors

"""
Calculate specific molecular descriptors using RDKit
"""


def MolLogP(sm):
    """
    calculate mollogp for a molecule

    Parameters
    ----------
    sm : string
        SMILES

    Returns
    -------
    Descriptors.MolLogP(Chem.MolFromSmiles(sm)) : float
        mollogp
    """

    try:
        return Descriptors.MolLogP(Chem.MolFromSmiles(sm))
    except:
        print("error! ", sm)
        return -1234


def TPSA(sm):
    """
    calculate TPSA for a molecule

    Parameters
    ----------
    sm : string
        SMILES

    Returns
    -------
    Descriptors.TPSA(Chem.MolFromSmiles(sm)) : float
        TPSA
    """
    try:
        return Descriptors.TPSA(Chem.MolFromSmiles(sm))
    except:
        print("error! ", sm)
        return -1234
