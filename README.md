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

### Data
European AAA-rated government bond yields has been undertaken. Different variations of models are being suggested, and finally tested against appropriate benchmarks.
