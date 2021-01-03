# Predictive Yield Curve Modeling in Reduced Dimensionality

### Research Goal
The term structure of interest rates (“yield curve”) is a representation that plots bonds of the same type (e.g. credit quality, sector) in terms of their prices, expressed as yields, over different maturity dates.

The aim of this project is to study yield curve dynamics in reduced dimensionality by applying principal components analysis (PCA). An interpretation for the driving factors, as well as the capability to mimic actual yield curves shall be assessed. Furthermore, the short-term predictive power of these extracted factors is explored. While many econometric forecasting models struggle with the vast number of variables that highly increase complexity and thus uncertainty, the suggested approach makes use of the compressed feature set. 

### Project Structure
The script is split into 5 notebooks, representing separate tasks within the project workflow:
- Data preparation
- Principal component analysis
- Time series model
- Time series benchmark model
- Ensemble model

<p align="center">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/workflow.PNG"> 
</p>

### Data
Underlying data set is sourced from ECB and covers daily European AAA-rated government bond yields over the last 15 years. https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html

<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/01_yield_curve.png", width = "500"><br>
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/02_yields.png", width = "500">
</p>

### Principal Component Analysis
By deriving the yield curves' underlying principal components (PCs), its can be shown that already the first 3 are able to explain more than 95% of total yield curve variance. Thus for certain applications it might be sufficient to only work with these limited number of factors. Furthermore a connection between the first 3 PCs to the classical yield curve factors "level", "slope", "curvature" can be established.
<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/04_pc_loadings.png", width = "500"><br>
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/05_pc_scores.png", width = "500">
</p>

### Predictive Models
In order to explore any short-term predictability of the PCs, several models have been tested. All of them forecast PC's which generate full yield curve predictions by back-transforming PC's to their original dimensional form.

- **Auto-regressive (AR) model:** Predicting future PC's from its own past lags. Standard tests for time series regarding stationaritay and causality had to be applied.
- **Dynamic Nelson-Siegel (DNS) model:** Takes the yield curve parameters specified by Nelson-Siegel and runs an VAR-model. This serves as a benchmark.
- **Extreme Gradient Boosting model:** By converting the time series into a cross-sectional format, an ensemble can be applied as well.

<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/08_stationary2.png", width = "500"><br>
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/10_nelson_siegel_forecast.png", width = "500">
</p>

### Conclusion
None of the algorithms were significantly outperforming a naive forecast over a longer period of time. With a few exceptions, the short-term variations in yield curves are too stochastic and small, that any of the models could outperform. However, more research can be done by extending the predicition horizon, or by implementing partial differencing into the time series model.
