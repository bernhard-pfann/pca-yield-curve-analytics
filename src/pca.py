import numpy as np
import pandas as pd


class PCA(object):
    """
    This class derives the principal components of a given data matrix. Based on 
    the datas' covariance matrix, Eigenvectors are calculated, which act as coordinate 
    system for PC-scores. By only retaining a few of the newly established dimensions, 
    the backtransformation into original units creates lower dimensional data model. 
    """
    def __init__(self, df: pd.DataFrame, k: int):
        self.df = df
        self.maturities = df.columns
        self.k = k
        
        self.get_cov_matrix()
        self.get_eig_vectors()
        self.get_eig_scores()
        self.backtrans()
        
    def get_cov_matrix(self):
        """
        Calculates the covariance matrix of all given maturities to each other over time 
        whole time horizon.
        """
        self.cov_matr = np.array(self.df).T
        self.cov_matr = np.cov(self.cov_matr, bias = True)
        self.cov_matr = pd.DataFrame(
            data    = self.cov_matr, 
            columns = self.maturities, 
            index   = self.maturities
        )
        
    def get_eig_vectors(self):
        """
        Calculates the Eigenvectors. By definition these are the vectors that capture the 
        maximum variance of the underlying data, and can be found by minimizing the sum of 
        projection length to the respective vector.
        """
        # Eigen decomposition
        eig = np.linalg.eig(self.cov_matr)
        self.idx = list(["PC_"+str(i) for i in range(1, eig[0].shape[0]+1)])

        # Eigen values 
        self.eig_vals = pd.DataFrame(eig[0].real, columns = ["eig_val"], index = self.idx)
        self.eig_vals["eig_val_rel"] = self.eig_vals["eig_val"].apply(lambda x: x/self.eig_vals["eig_val"].sum())
        self.eig_vals["eig_val_abs"] = self.eig_vals["eig_val_rel"].cumsum()

        # Eigen vectors
        self.eig_vect = pd.DataFrame(
            data    = eig[1].real, 
            index   = self.maturities, 
            columns = self.idx,
        )
        self.eig_vect_k = self.eig_vect.iloc[:,:self.k]
        
    def get_eig_scores(self):
        """
        This function transforms the underlying data into the new dimensionality formed by the 
        Eigenvectors. Transformed datapointscan be labeled PC-scores.
        """
        # PC scores (all)
        self.eig_scores = np.matrix(self.df) * np.matrix(self.eig_vect)
        self.eig_scores = pd.DataFrame(
            data    = self.eig_scores,
            columns = self.idx,
            index   = pd.to_datetime(self.df.index)
        )

        # PC scores (retained)
        self.eig_scores_k = self.eig_scores.iloc[:,:self.k]
        
    def backtrans(self):
        """
        This function retains only a limited set of PCs and transformes the PC-scores back to the 
        original coordinate system. Therefore the final output is in the same units as the input.
        """
        
        # Inverse transformation (all)
        self.eig_vect_inv = pd.DataFrame(
            data    = np.linalg.inv(np.matrix(self.eig_vect)),
            columns = self.maturities,
            index   = self.idx)

        # Inverse transformation (retained)
        self.eig_vect_inv_k = self.eig_vect_inv.iloc[:self.k,:]

        self.yields = np.matrix(self.eig_scores_k) * np.matrix(self.eig_vect_inv_k)
        self.yields = pd.DataFrame(
            data    = self.yields, 
            columns = self.maturities,
            index   = self.eig_scores_k.index)

    def backtrans_oos(self, df_oos):
        """
        This function derives the dimensionality reduction of out-of-sample data. This means, that 
        Eigenvectors are fitted on train data, and are being applied to unseen test data.
        """
        # PC scores (all)
        eig_scores = np.matrix(df_oos) * np.matrix(self.eig_vect)
        eig_scores = pd.DataFrame(
            data    = eig_scores,
            columns = self.idx,
            index   = df_oos.index
        )

        # PC scores (retained)
        eig_scores_k = eig_scores.iloc[:,:self.k]
        
        # Inverse transformation (all)
        eig_vect_inv = pd.DataFrame(
            data    = np.linalg.inv(np.matrix(self.eig_vect)), 
            columns = self.maturities, 
            index   = self.idx
        )

        # Inverse transformation (retained)
        eig_vect_inv_k = eig_vect_inv.iloc[:self.k,:]

        yields = np.matrix(eig_scores_k) * np.matrix(eig_vect_inv_k)
        yields = pd.DataFrame(
            data    = yields, 
            columns = self.maturities,
            index   = eig_scores_k.index
        )
        
        return yields