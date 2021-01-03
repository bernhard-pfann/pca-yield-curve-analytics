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
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/01_yield_curve.png", height = "310">
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/02_yields.png", height = "310">
</p>

### Principal Component Analysis
By deriving the underlying principal components, its can be shown that already the first 3 are able to explain more than 95% of total yield curve volatility. Thus for certain applications it might be sufficient to work with these limited 
<p align="center"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/04_pc_loadings.png", height = "270"> 
  <img src="https://github.com/bernhard-pfann/pca-yield-curve-analytics/blob/main/05-images/05_pc_scores.png", height = "270">
</p>

