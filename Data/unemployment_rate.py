import pandas as pd
# Source: Bureau of Labor Statistics

# Load the Excel file
df = pd.read_excel("/Users/shaiverma/Documents/CSE4095/DeepLearningProject/unemployment_report.xlsx")

# Save as CSV
df.to_csv("unemployment_rates.csv", index=False)
