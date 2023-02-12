import os
from joblib import dump
from datetime import datetime as dt

from src.utils import download
from src.rates import clean_rates
from src.pca import PCA
import config as conf

def main():

    if not os.path.exists("assets"):
        os.makedirs("assets")
    if not os.path.exists("assets/stress"):
        os.makedirs("assets/stress")

    lol = False

    if lol:
        download(
            target_path = "assets/rates_raw.csv",
            start_date  = conf.start_date,
            end_date    = conf.end_date
        )

    df = clean_rates(
        input_path = "assets/rates_raw.csv", 
        start      = conf.start_date, 
        end        = conf.end_date, 
        maturities = conf.maturities,
        freq       = conf.frequency
    )

    df.to_csv("assets/rates_clean.csv")

    # Train-test split
    test_date = dt.strptime(conf.test_date, "%Y-%m-%d")
    df_train = df[df.index < test_date]
    df_test = df[df.index >= test_date]

    mdl = PCA(df=df_train, k=conf.n_components)
    
    df_train.to_csv("assets/train.csv")
    df_test.to_csv("assets/test.csv")

    # Create scenarios where each principal component is stressed separately
    for i in range(1, conf.n_components+1):
        df_up, df_down = PCA.univariate_stress(
            self=mdl, 
            pc="PC_"+str(i), 
            sigma=conf.sigma_deviation, 
            n_days=conf.n_days
        )
        
        df_up.to_csv("assets/stress/pc_"+str(i)+"_up.csv")
        df_down.to_csv("assets/stress/pc_"+str(i)+"_down.csv")


    dump(mdl, "assets/pca.joblib")
    


    # pc_scores      = mdl.eig_scores_k
    # pc_vectors     = mdl.eig_vect_k
    # pc_vectors_inv = mdl.eig_vect_inv_k
    # pc_back_trans  = mdl.yields
    # pc_idx         = mdl.idx[:conf.n_components]
    
    # df_oos = mdl.backtrans_oos(df_test)

    import pdb; pdb.set_trace()
    print("DONE")

if __name__ == "__main__":
    main()