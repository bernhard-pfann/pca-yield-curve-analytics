# Predictive Yield Curve Modeling in Reduced Dimensionality

The term structure of interest rates (“yield curve”) is a representation that plots bonds of the same type (e.g. credit quality, sector) in terms of their prices, expressed as yields, over different maturity dates. This project sets out to study the yield curve dynamics in reduced dimensionality. In literature Principal Component Analysis (PCA) is a known application to this use case.

<u>After a successful yield curve decomposition the following topics will be tackled:</u>
1) Supporting the interpretation of the first 3 principal components (PCs) in accordance with traditional (shift,slope,curvature) factors
2) Testing out-of-sample fit for model yield curves, generated from reduced principal components set
3) Derivation of non-linear stress scenarios for each component (1-month ahead 95% confidence)
4) Testing predictability with an autoregressive timeseries model

### A1. Data
Underlying data set is sourced from ECB and covers daily Euro area AAA-rated government spot rate yield curves for Jan-2015 to Dec-2020. https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html

<p align="center"> <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/yields-dyn.gif"></p>

### A2. Project Structure
The script can be executed via "main.ipynb" and thereby calls custom modules:
- yieldcurves.py --> Cleaning of raw input put from source
- principalcomponents.py --> Object class that conducts all transformations of PCs
- autoregressive.py --> Object class fits a time series model and returns predictions, based on a simple autoregressive-process

### 1. Interpreting Principal Component Analysis
By deriving the yield curves' underlying principal components (PCs), its can be shown that already the first 3 are able to explain more than 95% of total yield curve variance. Thus for certain applications it might be sufficient to only work with these limited number of factors. Furthermore a connection between the first 3 PCs to the classical yield curve factors "level", "slope", "curvature" can be established.
<p align="center">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-scores-dyn.gif">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-interpret.png"><br>
</p>


### 2. Out-Of-Sample Fit
By comparing the derived model curves with actual yield curves, the goodness-of-fit can be evaluated. It can be shown that especially regular non-inverting curves can be fitted with a limited number of components, while special cases required higher dimensionality, or potential oversampling of these extreme cases.
<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-fit-dyn.gif">
</p>


### 3. Stress Scenarios
Since it has been proven, that few PCs are able to capture the majority of yield curve variance, realistic stress scenarios can be derived from them. By assessing the variation of each of the first 3 PCs, shock scenarios for a 1-month 95% confidence intervall have been derived.
<p align="center"><img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/yields-stress.png"></p>


### 4. Predictive Model
In order to explore any short-term predictability of the PCs, and autoregressive model is derived to forecast the main PCs one-step-ahead. The model is benchmarked against a naive (no change) forecast. The benchmark can be outperformed by a small margin, since lagged coefficients shown statistical significance, but are small in size.

<p align="center"><img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/yields-pred-eval.png"></p><br>

<b>Python Version:</b> 3.7<br>
<b>Packages:</b> pandas, numpy, datetime, sklearn, statsmodels, matplotlib, seaborn, ipywidgets, warnings
