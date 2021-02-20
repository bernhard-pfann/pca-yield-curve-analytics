import numpy as np
import pandas as pd


class PCA(object):
    """
    This class derives the principal components of a given data matrix. Based on the datas' covariance matrix (func:pca_cov), 
    Eigenvectors are calculated (func:pca_eig), which act as coordinate system for PC-scores (func:pca_scores). By only retaining
    a few of the newly established dimensions, the backtransformation (func:pca_backtrans) into original units creates a lower 
    dimensional data model. 
    """
    def __init__(self, spot, maturities, k):
        """
        Required parameters:
        - spot: DataFrame()
        - maturities: List >> Given maturities
        - k: Int >> Number of retained PCs
        """
        self.spot = spot
        self.mat  = maturities
        self.k    = k
        
        self.pca_cov()
        self.pca_eig()
        self.pca_scores()
        self.pca_backtrans()
        
    def pca_cov(self):
        """
        This function calculates the covariance matrix of all given maturities to each other over time whole time horizon.
        """
        self.cov_matr = np.array(self.spot).T
        self.cov_matr = np.cov(self.cov_matr, bias = True)
        self.cov_matr = pd.DataFrame(data    = self.cov_matr, 
                                     columns = self.mat, 
                                     index   = self.mat)
        
        
    def pca_eig(self):
        """
        This function calculates the Eigenvectors. By definition these are the vectors that capture the maximum variance of the
        underlying data, and can be found by minimizing the sum of projection length to the respective vector.
        """
    
        # Eigen decomposition
        eig = np.linalg.eig(self.cov_matr)
        self.idx = list(["PC_"+str(i) for i in range(1, eig[0].shape[0]+1)])


        # Eigen values 
        self.eig_vals = pd.DataFrame(eig[0].real, columns = ["eig_val"], index = self.idx)
        self.eig_vals["eig_val_rel"] = self.eig_vals["eig_val"].apply(lambda x: x/self.eig_vals["eig_val"].sum())
        self.eig_vals["eig_val_abs"] = self.eig_vals["eig_val_rel"].cumsum()

        
        # Eigen vectors
        self.eig_vect = pd.DataFrame(eig[1].real, index = self.mat, columns = self.idx)
        self.eig_vect_k = self.eig_vect.iloc[:,:self.k]

        
    def pca_scores(self):
        """
        This function transforms the underlying data into the new dimensionality formed by the Eigenvectors. Transformed datapoints
        can be labeled PC-scores.
        """
    
        # PC scores (all)
        self.eig_scores = np.matrix(self.spot) * np.matrix(self.eig_vect)
        self.eig_scores = pd.DataFrame(data    = self.eig_scores,
                                       columns = self.idx,
                                       index   = pd.to_datetime(self.spot.index))

        # PC scores (retained)
        self.eig_scores_k = self.eig_scores.iloc[:,:self.k]

        
    def pca_backtrans(self):
        """
        This function retains only a limited set of PCs and transformes the PC-scores back to the original coordinate system. Therefore the
        final output is in the same units as the input.
        """
        
        # Inverse transformation (all)
        self.eig_vect_inv = pd.DataFrame (data = np.linalg.inv(np.matrix(self.eig_vect)),
                                          columns = self.mat,
                                          index = self.idx)

        # Inverse transformation (retained)
        self.eig_vect_inv_k = self.eig_vect_inv.iloc[:self.k,:]

        self.yields = np.matrix(self.eig_scores_k) * np.matrix(self.eig_vect_inv_k)
        self.yields = pd.DataFrame(data    = self.yields, 
                                   columns = self.mat,
                                   index   = self.eig_scores_k.index)


    def pca_oos(self, eig_vect_train, spot_test):
        """
        This function derives the dimensionality reduction of out-of-sample data. This means, that Eigenvectors are fitted on train data,
        and are being applied to unseen test data.
        """
        # PC scores (all)
        eig_scores_oos = np.matrix(spot_test) * np.matrix(eig_vect_train)
        eig_scores_oos = pd.DataFrame(data    = eig_scores_oos,
                                      columns = self.idx,
                                      index   = pd.to_datetime(spot_test.index))

        # PC scores (retained)
        eig_scores_oos_k = eig_scores_oos.iloc[:,:self.k]
        
        # Inverse transformation (all)
        eig_vect_inv_oos = pd.DataFrame(data=np.linalg.inv(np.matrix(eig_vect_train)), columns=self.mat, index=self.idx)

        # Inverse transformation (retained)
        eig_vect_inv_oos_k = eig_vect_inv_oos.iloc[:self.k,:]

        yields_oos = np.matrix(eig_scores_oos_k) * np.matrix(eig_vect_inv_oos_k)
        yields_oos = pd.DataFrame(data = yields_oos, 
                                  columns = self.mat,
                                  index = eig_scores_oos_k.index)
        
        return yields_oos