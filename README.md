# Predictive Yield Curve Modeling in Reduced Dimensionality

The term structure of interest rates (“yield curve”) is a representation that plots bonds of the same type (e.g. credit quality, sector) in terms of their prices, expressed as yields, over different maturity dates. This project sets out to study the yield curve dynamics in reduced dimensionality. In literature Principal Component Analysis (PCA) is a known application to this use case.

<u>After a successful yield curve decomposition the following topics will be tackled:</u>
1) Supporting the interpretation of the first 3 principal components (PCs) in accordance with traditional (shift,slope,curvature) factors
2) Testing out-of-sample fit for model yield curves, generated from reduced principal components set
3) Derivation of non-linear stress scenarios for each component (1-month ahead 95% confidence)
4) Testing predictability with an autoregressive timeseries model

### A1. Data
Underlying data set is sourced from ECB and covers daily Euro area AAA-rated government spot rate yield curves. They selected time horizon starts from Jan-2015 to Dec-2020. https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html

<p align="center"> <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/yields-dyn.gif"></p>

### A2. Project Structure
The script can be executed via "main.ipynb" and thereby calls custom modules:
- yieldcurves.py --> Cleaning of raw input put from source
- principalcomponents.py --> Object class that conducts all transformations of PCs
- autoregressive.py --> Object class fits a time series model and returns predictions, based on a simple autoregressive-process

### 1) Interpreting Principal Component Analysis
By deriving the yield curves' underlying principal components (PCs), its can be shown that already the first 3 are able to explain more than 95% of total yield curve variance. Thus for certain applications it might be sufficient to only work with these limited number of factors. Furthermore a connection between the first 3 PCs to the classical yield curve factors "level", "slope", "curvature" can be established.
<p align="center">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-scores-dyn.gif">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-interpret.png", width = "700"><br>
</p>


### 2) Out-Of-Sample Fit
By comparing the derived model curves with actual yield curves, the goodness-of-fit can be evaluated. It can be shown that especially regular non-inverting curves can be fitted with a limited number of components, while special cases required higher dimensionality, or potential oversampling of these extreme cases.

<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-fit-dyn.gif">
</p>


### 3) Stress Scenarios
Since it has been proven, that few PCs are able to capture the majority of yield curve variance, realistic stress scenarios can be derived from them. By assessing the variation of each of the first 3 PCs, shock scenarios for a 1-month 95% confidence intervall have been derived.

<p align="center">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/pc-scores-stress.png">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/img/yields-stress.png">
</p>


### 4) Predictive Models
In order to explore any short-term predictability of the PCs, several models have been tested. All of them forecast PC's which generate full yield curve predictions by back-transforming PC's to their original dimensional form.

- **Auto-regressive (AR) model:** Predicting future PC's from its own past lags. Standard tests for time series regarding stationaritay and causality had to be applied.
- **Dynamic Nelson-Siegel (DNS) model:** Takes the yield curve parameters specified by Nelson-Siegel and runs an VAR-model. This serves as a benchmark.
- **Extreme Gradient Boosting model:** By converting the time series into a cross-sectional format, an ensemble can be applied as well.

<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/stationary-2.png", width = "500"><br>
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/nelson-siegel-forecast.png", width = "500">
</p>

### Conclusion
None of the algorithms were significantly outperforming a naive forecast over a longer period of time. With a few exceptions, the short-term variations in yield curves are too stochastic and small, that any of the models could outperform. However, more research can be done by extending the predicition horizon, or by implementing partial differencing into the time series model.
