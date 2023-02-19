start_date = "2020-01-01"
end_date   = "2022-12-31"
test_date  = "2022-09-01"

maturities = [
    0.25, 0.5, 1, 2, 3 , 4, 5, 6, 7, 8, 9, 
    10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# Frequency of yield curve data ["day", "week", "month"]
frequency = "day" 

# Number of principal components
n_components = 3

# Stress scenario
n_days = 30
sigma_deviation = 2