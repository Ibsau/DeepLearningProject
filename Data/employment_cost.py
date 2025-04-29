import pandas as pd
# Source: Bureau of Labor Statistics

# Load the Excel file
df = pd.read_excel("/Users/shaiverma/Documents/CSE4095/DeepLearningProject/SeriesReport-20250423213955_06b31d.xlsx")

# Save as CSV
df.to_csv("employment_cost.csv", index=False)