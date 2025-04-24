import pandas as pd
# Source: Bureau of Labor Statistics

# Load the Excel file
df = pd.read_excel("/Users/shaiverma/Documents/CSE4095/DeepLearningProject/SeriesReport-20250423210243_1879f5.xlsx")

# Save as CSV
df.to_csv("inflation_rates.csv", index=False)








