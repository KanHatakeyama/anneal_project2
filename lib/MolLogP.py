"""from rdkit import Chem
from rdkit.Chem import Descriptors

def MolLogP(sm):
    try:
        return Descriptors.MolLogP(Chem.MolFromSmiles(sm))
    except:
        print("error! ", sm)
        return -1234"""