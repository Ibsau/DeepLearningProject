# Demand multipliers by main category

# Bureau of Labor Statistics, U.S. Department of Labor:
# "Occupational Outlook Handbook" (2023–2024 edition)
# LinkedIn Workforce Report (2023)
# Indeed Hiring Lab Reports (2024)
# World Economic Forum: Future of Jobs Report (2023)
import pandas as pd

# Define your categories and subfields
categories = {
    "Medical & Life Sciences": ["Healthcare", "Biology", "Pharmaceutical Sciences", "Nursing", "Biomedical Engineering", "Public Health"],
    "Math & Technology": ["Computer Science", "Mathematics", "Software Engineering", "Mechanical Engineering", "Data Science", "Electrical Engineering"],
    "Business & Finance": ["Finance", "Accounting", "Real Estate", "Investment Banking", "Taxation", "Supply Chain Management"],
    "Humanities & Social Sciences": ["Political Science", "Geography", "Sociology", "International Relations", "History", "Psychology"],
    "Arts & Design": ["Fine Arts", "Music", "Photography", "Videography", "Graphic Design", "Fashion Design"]
}

# Demand multipliers for each category
category_demand_multiplier = {
    "Medical & Life Sciences": 1.3,
    "Math & Technology": 1.2,
    "Business & Finance": 1.0,
    "Humanities & Social Sciences": 0.9,
    "Arts & Design": 0.8
}

# Macro scores for each year
macro_scores = {
    2023: 0.11583333333333339,
    2024: -0.040000000000000285
}

# Create rows for each field/year
rows = []

for year, macro_score in macro_scores.items():
    for category, fields in categories.items():
        for field in fields:
            multiplier = category_demand_multiplier.get(category, 1.0)
            adjusted_score = macro_score * multiplier
            rows.append({
                'Category': category,
                'Field': field,
                'Year': year,
                'Adjusted_Score': adjusted_score
            })

# Create a DataFrame
final_df = pd.DataFrame(rows)

# Save it to a CSV file
final_df.to_csv('final_adjusted_macro_scores.csv', index=False)


