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
        
        self.maturities = df.columns
        self.components = list(["PC_"+str(i) for i in range(1, len(self.maturities)+1)])
        self.k = k

        self.cov                 = self.get_covariance(df)
        self.eig_values          = self.get_eig_values(self.cov)
        self.eig_vectors         = self.get_eig_vectors(self.cov)
        self.eig_vectors_inverse = self.get_eig_vectors_inverse(self.eig_vectors)
        self.eig_scores          = self.get_eig_scores(df=df, eig_vectors=self.eig_vectors)
        self.backtrans_rates     = self.get_backtrans_rates(eig_scores=self.eig_scores, eig_vectors_inverse=self.eig_vectors_inverse)
        
    def get_covariance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the covariance matrix of all given maturities to each other over time 
        whole time horizon.
        """
        cov = np.array(df).T
        cov = np.cov(cov, bias = True)
        cov = pd.DataFrame(
            data=cov, 
            columns=self.maturities, 
            index=self.maturities
        )

        return cov
    
    def get_eig_values(self, cov: pd.DataFrame) -> pd.DataFrame:
        """Calculate the eigen vectors and return as dataframe"""
        eig = np.linalg.eig(cov)
        eig_values = pd.DataFrame(
            data=eig[0].real, 
            columns=["value"], 
            index=self.components
        )

        eig_values["relative"] = eig_values["value"] / eig_values["value"].sum()
        eig_values["cumulative"] = eig_values["relative"].cumsum()   
        return eig_values    

    def get_eig_vectors(self, cov: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Eigenvectors. By definition these are the vectors that capture the 
        maximum variance of the underlying data, and can be found by minimizing the sum of 
        projection length to the respective vector.
        """
        eig = np.linalg.eig(cov)
        eig_vectors = pd.DataFrame(
            data=eig[1].real, 
            index=self.maturities, 
            columns=self.components,
        )

        return eig_vectors

    def get_eig_vectors_inverse(self, eig_vectors: pd.DataFrame) -> pd.DataFrame:
        """Calculates the inverse matrix from eigen vectors."""

        eig_vectors_inverse = pd.DataFrame(
            data=np.linalg.inv(np.matrix(eig_vectors)),
            columns=self.maturities,
            index=self.components
        )

        return eig_vectors_inverse

    def get_eig_scores(self, df: pd.DataFrame, eig_vectors: pd.DataFrame) -> pd.DataFrame:
        """
        This function transforms the underlying data into the new dimensionality formed by the 
        Eigenvectors. Transformed datapointscan be labeled PC-scores.
        """
        eig_scores = np.matrix(df) * np.matrix(eig_vectors)
        eig_scores = pd.DataFrame(
            data=eig_scores,
            columns=self.components,
            index=pd.to_datetime(df.index)
        )

        return eig_scores

    def get_backtrans_rates(self, eig_scores: pd.DataFrame, eig_vectors_inverse: pd.DataFrame):
        """
        This function retains only a limited set of PCs and transformes the PC-scores back to the 
        original coordinate system. Therefore the final output is in the same units as the input.
        """
        eig_scores = eig_scores.iloc[:,:self.k]
        eig_vectors_inverse = eig_vectors_inverse.iloc[:self.k,:]

        rates = np.matrix(eig_scores) * np.matrix(eig_vectors_inverse)
        rates = pd.DataFrame(
            data=rates, 
            columns=self.maturities,
            index=eig_scores.index
        )

        return rates

    @staticmethod
    def get_backtrans_rates_oos(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        This function derives the dimensionality reduction of out-of-sample data. This means, that 
        Eigenvectors are fitted on train data, and are being applied to unseen test data.
        """
        eig_scores          = self.get_eig_scores(df=df_test, eig_vectors=self.eig_vectors)
        eig_vectors_inverse = self.get_eig_vectors_inverse(self.eig_vectors)
        rates               = self.get_backtrans_rates(eig_scores=eig_scores, eig_vectors_inverse=eig_vectors_inverse)
        
        return rates

    @staticmethod
    def get_stressed_eig_scores(self, sigma: float, direction: int, n_days: int) -> pd.DataFrame:
        """Return the eigen scores with added and subtracted rolling standard deviation"""

        std  = self.eig_scores.rolling(n_days).std()*sigma
        eig_scores = (self.eig_scores + std*direction).dropna()
        
        return eig_scores

    @staticmethod
    def univariate_stress(self, stressed_eig_scores: pd.DataFrame, pc: str) -> pd.DataFrame:
        """Shocks only one principal component while keeping other constant"""
        
        k_components = self.components[:self.k]
        k_components_not = [i for i in k_components if i != pc]

        stressed_eig_scores = stressed_eig_scores[pc]
        unstressed_eig_scores = self.eig_scores[k_components_not]

        mixed_eig_scores = pd.concat([unstressed_eig_scores, stressed_eig_scores], axis=1) \
            .dropna() \
            .reindex(k_components, axis=1)

        eig_vectors_inverse = self.eig_vectors_inverse.iloc[:self.k,:]

        rates = self.get_backtrans_rates(
            eig_scores=mixed_eig_scores, 
            eig_vectors_inverse=eig_vectors_inverse
        )

        return rates
