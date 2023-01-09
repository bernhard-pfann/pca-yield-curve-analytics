import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR


class Model(object):
    """
    This class is fitting and predicting a time series, based on a simple autoregressive-model. Only significant lags are considered.
    """
    def __init__(self, train, test, last_train, alpha, diff):
        """
        The class is initialized to transfored external parameters into class properties.
        """

        self.train, self.test, self.last_train = train, test, last_train
        self.models_init, self.models_param = dict(), dict()
        
        k = len(last_train)
        
        self.idx_orig  = self.train.columns[:-k]
        self.idx_trans = self.train.columns[-k:]
        self.alpha = alpha
        self.diff = diff
        
        for i in self.idx_trans:
            self.models_init[i] = AR(self.train[i].values)
            
    
    def fit(self, lag):
        """
        Required parameters:
        1) df:  dataFrame - consisting of train observations
        2) col: str - column name of respective time series
        3) lag: int - number of lags to be fitted
        """
        
        self.models_fitted = dict()
        
        for i in self.idx_trans:
            self.models_fitted[i] = self.models_init[i].fit(maxlag = lag)

            print (i.center(80))
            print (self.models_fitted[i].summary().tables[0])
            print (self.models_fitted[i].summary().tables[1])
            print ("\n")

        self.parameters()


    def predict(self, steps, components_pred):
        """
        Required parameters:
        1) df:    dataframe - consisting of train observations
        2) model: statsmodels.tsa.ar_model - fitted model
        3) steps: int - number of prediction steps
        4) diff:  bool - true if differenced time series has been fitted 
        """
        
        self.steps = steps
        self.pred = pd.DataFrame()

        # Iterate through all components ----------------------------------------
        for i,j in zip(self.idx_trans, self.idx_orig):

            # Use AR-model if component is in list
            if i in components_pred:
                
                # Model prediction
                temp = self.predict_component(component = i, component_name = j)


            # Use Naive-forecast if not in list ----------------------------------
            else:
                temp    = self.test[j].iloc[:steps].shift(1)
                temp[0] = self.train[j].iloc[-1]
                temp    = pd.Series(temp, index = self.test.index[:self.steps].strftime("%Y-%m-%d"), name = j)

            # Concatenate prediction of each component to dataframe --------------
            self.pred = pd.concat([self.pred, temp], axis = 1)
            
        self.pred.index = pd.to_datetime(self.pred.index)
        return self.pred
    
    
    def back_transform(self, pc_vect_inv, maturities):
        output = np.matrix(self.pred) * np.matrix(pc_vect_inv)
        output = pd.DataFrame(data = output,
                              columns = maturities,
                              index   = self.test.index[:self.steps].strftime("%Y-%m-%d"))
        
        output.index = pd.to_datetime(output.index)
        return output
    
    
    def parameters(self):
        """
        This function stores the parameters of the fitted model in a dictionary
        """
        for x in self.idx_trans:
        
            self.models_param[x] = dict()

            for i,(j, k) in enumerate(zip(self.models_fitted[x].params, 
                                          self.models_fitted[x].pvalues)):
                
                if i == 0:
                    self.models_param[x]["const"] = {"coef":j,"pval":k}
                else:
                    self.models_param[x]["L"+str(i)] = {"coef":j,"pval":k}
                    
                    
    def predict_component(self, component, component_name):
        """
        """
        output = list()
        component_orig = component.rstrip("_diff")
        
        
        for s in range(self.steps):
            pred = 0

            for i,j in enumerate (self.models_param[component].keys()):
                
                idx  = s-i
                pval = self.models_param[component][j]["pval"]
                coef = self.models_param[component][j]["coef"]

                if (j == "const"):
                    if pval < self.alpha:
                        pred += coef
                else:
                    if pval < self.alpha:
                        if idx >= 0:
                            pred += coef * self.test[component][idx]
                        else:
                            pred += coef * self.train[component][idx]
              
            
            # Transform if time series has been differenced                    
            if self.diff == True:
                if s == 0:
                    pred += self.train[component_orig][-1]
                else:
                    pred += self.test[component_orig][s-1]

            output.append(pred)

        return pd.Series(output, index = self.test.index[:self.steps].strftime("%Y-%m-%d"), name = component_name)