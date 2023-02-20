import argparse
from joblib import dump
from datetime import datetime as dt

from src.utils import create_folders, download
from src.rates import clean_rates
from src.pca import PCA
import config as conf


def main():
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Clean rates and performs PCA",
        epilog=""
    )
    
    parser.add_argument(
        "-o", 
        "--overwrite",
        action="store_true",
        default=False,
        required=False,
        help="overwrite data"
    )

    args = parser.parse_args()
    overwrite = args.overwrite

    ############
    # DOWNLOAD #
    ############
    
    create_folders()
    download(
        target_path = "assets/rates_raw.csv",
        start_date  = conf.start_date,
        end_date    = conf.end_date,
        overwrite   = overwrite
    )

    ####################
    # DATA PREPARATION #
    ####################

    df = clean_rates(
        input_path = "assets/rates_raw.csv", 
        start      = conf.start_date, 
        end        = conf.end_date, 
        maturities = conf.maturities,
        freq       = conf.frequency
    )

    test_date = dt.strptime(conf.test_date, "%Y-%m-%d")
    df_train = df[df.index < test_date]
    df_test = df[df.index >= test_date]

    df.to_csv("assets/rates_clean.csv")
    df_train.to_csv("assets/train.csv")
    df_test.to_csv("assets/test.csv")


    ########################
    # PRINCIPAL COMPONENTS #
    ########################

    mdl = PCA(df_train=df_train, df_test=df_test, k=conf.n_components)
    dump(mdl, "assets/pca.joblib")

    # Create scenarios where each principal component is stressed separately
    eig_scores_up = PCA.get_stressed_eig_scores(
        self=mdl, 
        sigma=conf.sigma_deviation, 
        direction=1, 
        n_days=conf.n_days
    )
    
    eig_scores_down = PCA.get_stressed_eig_scores(
        self=mdl, 
        sigma=conf.sigma_deviation, 
        direction=-1, 
        n_days=conf.n_days
    )

    eig_scores_up.to_csv("assets/stress/eig_scores/up.csv")
    eig_scores_down.to_csv("assets/stress/eig_scores/down.csv")

    for i in range(1, conf.n_components+1):

        pc = "PC_"+str(i)
        rates_up = PCA.univariate_stress(self=mdl, stressed_eig_scores=eig_scores_up, pc=pc)
        rates_down = PCA.univariate_stress(self=mdl, stressed_eig_scores=eig_scores_down, pc=pc)

        rates_up.to_csv("assets/stress/rates/"+pc+"_up.csv")
        rates_down.to_csv("assets/stress/rates/"+pc+"_down.csv")


    ####################
    # PREDICTIVE MODEL #
    ####################

    print("DONE")

if __name__ == "__main__":
    main()