#from MultiPropScorer import MultiPropScorer
import joblib
import numpy as np

class multiprop_scoring():

    kwargs = ["model_path"]
    #k = 4.0
    model_path="../data/model.bin"

    def __init__(self):
        self.model=joblib.load(self.model_path)

    def __call__(self, smiles):
        return self.model.predict(smiles)


    
