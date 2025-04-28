import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your datasets
employment_cost = pd.read_csv('/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/employment_cost.csv')
inflation_rates = pd.read_csv('/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/inflation_rates.csv')
unemployment_rates = pd.read_csv('/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/unemployment_rates.csv')

# Step 1: Fix the employment cost data
employment_cost['Estimate Value'] = pd.to_numeric(employment_cost['Estimate Value'], errors='coerce')
employment_cost_yearly = employment_cost.groupby('Year')['Estimate Value'].mean().reset_index()
employment_cost_yearly.rename(columns={'Estimate Value': 'avg_employment_cost'}, inplace=True)

# Step 2: Fix the inflation rates data
inflation_rates['avg_inflation_rate'] = inflation_rates.loc[:, 'Jan':'Dec'].mean(axis=1)
inflation_yearly = inflation_rates[['Year', 'avg_inflation_rate']]

# Step 3: Fix the unemployment rates data
unemployment_rates['avg_unemployment_rate'] = unemployment_rates.loc[:, 'Jan':'Dec'].mean(axis=1)
unemployment_yearly = unemployment_rates[['Year', 'avg_unemployment_rate']]

# Step 4: Merge all datasets on 'Year'
merged_df = employment_cost_yearly.merge(inflation_yearly, on='Year', how='inner')
merged_df = merged_df.merge(unemployment_yearly, on='Year', how='inner')

# Step 5: Manually add GDP growth rate for 2023 and 2024
gdp_growth = {2023: 2.9, 2024: 2.8}
merged_df['avg_gdp_growth_rate'] = merged_df['Year'].map(gdp_growth)

# Step 6: Filter only for 2023 and 2024
final_df = merged_df[merged_df['Year'].isin([2023, 2024])]

final_df['macro_score'] = (
    (0.4 * final_df['avg_gdp_growth_rate']) -
    (0.3 * final_df['avg_unemployment_rate']) -
    (0.2 * final_df['avg_inflation_rate']) +
    (0.1 * final_df['avg_employment_cost'])
)

# saving to csv format
final_df.to_csv('final_macro_features.csv', index=False)

# Data
years = [2023, 2024]
macro_scores = [0.11583333333333339, -0.040000000000000285]

# Set colors based on whether score is positive or negative
colors = ['green' if score >= 0 else 'red' for score in macro_scores]

# Create the plot
plt.figure(figsize=(8, 5))
bars = plt.bar(years, macro_scores, color=colors)

# Add horizontal line at y=0
plt.axhline(0, color='black', linewidth=0.8)

# Title and labels
plt.title('Macro Score Comparison: 2023 vs 2024')
plt.xlabel('Year')
plt.ylabel('Macro Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate the actual scores on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01 if yval >= 0 else yval - 0.03,
             f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


