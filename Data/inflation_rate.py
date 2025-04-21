import pandas as pd
# Source: Bureau of Labor Statistics

# Load the Excel file
df = pd.read_excel("/Users/shaiverma/Documents/CSE4095/DeepLearningProject/SeriesReport-20250417191516_70894a.xlsx")

# Save as CSV
df.to_csv("inflation_rates.csv", index=False)








