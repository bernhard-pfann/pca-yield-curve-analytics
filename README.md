# Predictive Yield Curve Modeling in Reduced Dimensionality
<i>An Application of Principal Component Analysis to euro yield curves</i>

The term structure of interest rates (“yield curve”) is a representation that plots bonds of the same type (e.g. credit quality, sector) in terms of their prices, expressed as yields, over different maturity dates. This project sets out to study the yield curve dynamics in reduced dimensionality. In literature Principal Component Analysis (PCA) is a known application to this use case.

<u>After a successful yield curve decomposition the following topics will be tackled:</u>
- Supporting the interpretation of the first 3 principal components (PCs) in accordance with traditional (shift,slope,curvature) factors
- Testing out-of-sample fit for model yield curves, generated from reduced principal components set
- Derivation of non-linear stress scenarios for each component (1-month ahead 95% confidence)
- Testing predictability with an autoregressive timeseries model


### Project Structure
The script can be executed via "main.ipynb" and thereby calls custom modules:
- yieldcurves.py >> Cleaning of raw input put from source
- principalcomponents.py >> Object class that conducts all transformations of PCs
- autoregressive.py >> Object class fits a time series model and returns predictions, based on a simple autoregressive-process

<p align="center">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/workflow.PNG"> 
</p>


### Data
Underlying data set is sourced from ECB and covers daily Euro area AAA-rated government spot rate yield curves. They selected time horizon starts from 01-01-2015 to 31.12.2020 on a daily basis. https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html

<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/yield-curve.png", width = "500"><br>
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/yields.png", width = "500">
</p>


### Interpreting Principal Component Analysis
By deriving the yield curves' underlying principal components (PCs), its can be shown that already the first 3 are able to explain more than 95% of total yield curve variance. Thus for certain applications it might be sufficient to only work with these limited number of factors. Furthermore a connection between the first 3 PCs to the classical yield curve factors "level", "slope", "curvature" can be established.
<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/pc-loadings.png", width = "500"><br>
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/assets/readme/pc-scores.png", width = "500">
</p>

### Predictive Models
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
