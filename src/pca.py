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
        self.maturities = self.df.columns
        self.pc_names = list(["PC_"+str(i) for i in range(1, len(self.maturities)+1)])
        self.k = k

        self.cov                 = self.get_covariance(df=self.df, maturities=self.maturities)
        self.eig_values          = self.get_eig_values(cov=self.cov, pc_names=self.pc_names)
        self.eig_vectors         = self.get_eig_vectors(cov=self.cov, pc_names=self.pc_names, maturities=self.maturities)
        self.eig_vectors_inverse = self.get_eig_vectors_inverse(eig_vectors=self.eig_vectors, maturities=self.maturities, pc_names=self.pc_names)
        self.eig_scores          = self.get_eig_scores(df=self.df, eig_vectors=self.eig_vectors, pc_names=self.pc_names)

        self.eig_vectors_k  = self.eig_vectors.iloc[:,:self.k]
        self.eig_scores_k   = self.eig_scores.iloc[:,:self.k]
        self.eig_vect_inv_k = self.eig_vect_inv.iloc[:self.k,:]

        self.backtrans()
        
    @staticmethod
    def get_covariance(df: pd.DataFrame, maturities: list) -> pd.DataFrame:
        """
        Calculates the covariance matrix of all given maturities to each other over time 
        whole time horizon.
        """
        cov = np.array(df).T
        cov = np.cov(cov, bias = True)
        cov = pd.DataFrame(
            data=cov, 
            columns=maturities, 
            index=maturities
        )

        return cov
    
    @staticmethod
    def get_eig_values(cov: pd.DataFrame, pc_names: list) -> pd.DataFrame:
        """Calculate the eigen vectors and return as dataframe"""
        eig = np.linalg.eig(cov)
        df = pd.DataFrame(
            data=eig[0].real, 
            columns=["value"], 
            index=pc_names
        )

        df["relative"] = df["value"] / df["value"].sum()
        df["cumulative"] = df["relative"].cumsum()   
        return df     

    @staticmethod
    def get_eig_vectors(cov: pd.DataFrame, pc_names: list, maturities: list) -> pd.DataFrame:
        """
        Calculates the Eigenvectors. By definition these are the vectors that capture the 
        maximum variance of the underlying data, and can be found by minimizing the sum of 
        projection length to the respective vector.
        """
        eig = np.linalg.eig(cov)
        df = pd.DataFrame(
            data=eig[1].real, 
            index=maturities, 
            columns=pc_names,
        )

        return df 

    @staticmethod
    def get_eig_vectors_inverse(eig_vectors: pd.DataFrame, maturities: list, pc_names: list) -> pd.DataFrame:
        """Calculates the inverse matrix from eigen vectors."""

        df = pd.DataFrame(
            data=np.linalg.inv(np.matrix(eig_vectors)),
            columns=maturities,
            index=pc_names
        )

        return df

    @staticmethod 
    def get_eig_scores(df: pd.DataFrame, eig_vectors: pd.DataFrame, pc_names: list) -> pd.DataFrame:
        """
        This function transforms the underlying data into the new dimensionality formed by the 
        Eigenvectors. Transformed datapointscan be labeled PC-scores.
        """
        eig_scores = np.matrix(df) * np.matrix(eig_vectors)
        eig_scores = pd.DataFrame(
            data=eig_scores,
            columns=pc_names,
            index=pd.to_datetime(df.index)
        )

        return eig_scores

    @staticmethod    
    def get_backtrans_rates(eig_scores_k: pd.DataFrame, eig_vect_inv_k: pd.DataFrame, maturities: list):
        """
        This function retains only a limited set of PCs and transformes the PC-scores back to the 
        original coordinate system. Therefore the final output is in the same units as the input.
        """

        df = np.matrix(eig_scores_k) * np.matrix(eig_vect_inv_k)
        df = pd.DataFrame(
            data=df, 
            columns=maturities,
            index=eig_scores_k.index
        )

        return df

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

        rates = np.matrix(eig_scores_k) * np.matrix(eig_vect_inv_k)
        rates = pd.DataFrame(
            data    = rates, 
            columns = self.maturities,
            index   = eig_scores_k.index
        )
        
        return rates

    def get_stressed_eig_scores(self, sigma: float, n_days: int):
        """Return the eigen scores with added and subtracted rolling standard deviation"""

        std  = self.eig_scores.rolling(n_days).std()*sigma
        up   = (self.eig_scores + std).dropna()
        down = (self.eig_scores - std).dropna()

        return up, down

    @staticmethod
    def univariate_stress(self, pc: str, sigma: float, n_days: int) -> pd.DataFrame:
        """Shocks only one principal component while keeping other constant"""
        
        k_cols = self.idx[:self.k]
        unstressed_cols = [i for i in k_cols if i != pc]

        eig_scores_up, eig_scores_down = self.get_stressed_eig_scores(sigma=sigma, n_days=n_days)

        df_up = pd.concat([self.eig_scores[unstressed_cols], eig_scores_up[pc]], axis=1) \
            .dropna() \
            .reindex(k_cols, axis=1)
        df_down = pd.concat([self.eig_scores[unstressed_cols], eig_scores_down[pc]], axis=1) \
            .dropna() \
            .reindex(k_cols, axis=1)
        

        
        df_up = pd.DataFrame(
            data=np.matrix(df_up) * np.matrix(self.eig_vect_inv_k), 
            columns=self.maturities,
            index=self.eig_scores[n_days-1:].index
        )

        df_down = pd.DataFrame(
            data=np.matrix(df_down) * np.matrix(self.eig_vect_inv_k), 
            columns=self.maturities,
            index=self.eig_scores[n_days-1:].index
        )

        return df_up, df_down

    