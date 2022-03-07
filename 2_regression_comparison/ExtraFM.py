from scipy.sparse import csr_matrix
from pyfm import pylibfm
import numpy as np
from numba import jit

default_model = pylibfm.FM(task="regression", num_iter=30,initial_learning_rate=10**-3,
                      num_factors=10,
                   verbose=False
                    )


"""
For extrapolating prediction
The library caps the predicted value: y=min(y,max of y in the train data)
The phenomena can avoided by temporaliry predicting values under classification mode
However,the prediction accuracy is never good, basically
"""
    
    
class ExtraFM:
    def __init__(self,model=None,extra_mode=False):
        if model is None:
            model=default_model
        self.model=model
        self.extra_mode=extra_mode
        
    def fit(self,X,y):
        sparse_X=csr_matrix(X.astype("double"))
        self.model.fit(sparse_X, y)
        #self.qubo=calc_qubo(self.model.v,self.model.v[0].shape[0],self.model.v.shape[0])+np.diag(self.model.w)
        
        #calc offset
        self.b=self.model.predict(csr_matrix(np.zeros(X[0].shape[0]).astype("double")))        

        #self.y_max=max(y)
        self.y_max=0
        #self.y_max=max(y)
        self.y_min=0
        
    def predict(self,X):
        
        # normal prediction by the library
        if not self.extra_mode:
            sparse_X=csr_matrix(X.astype("double"))
            return self.model.predict(sparse_X)


        y=self.original_predict(X,reg_mode=False)
        
        #print(X.shape,y.shape)
        y=-np.log((1-y)/y)
        
        # fill nan
        nan_ids=np.where(y==np.inf)
        y[nan_ids]=self.y_max
        nan_ids=np.where(y==-np.inf)
        y[nan_ids]=self.y_min

        return y
    
    def original_predict(self,X,reg_mode=True):
        if reg_mode:
            self.model.fm_fast.task=0
        else:
            self.model.fm_fast.task=1
        sparse_X=csr_matrix(X.astype("double"))
        return self.model.predict(sparse_X)
    
@jit
def calc_qubo(v,dim1,dim2):
    qubo=np.zeros((dim1,dim1))

    for k in range(dim2):
        for i in range(dim1):
            for j in range(i):
                val=v[k][j]*v[k][i]
                qubo[j,i]+=val
    return qubo



