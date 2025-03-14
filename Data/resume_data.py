import random
import ollama
import pandas as pd
import multiprocessing

# Define categories of fields and associated companies
categories = {
    "Medical & Life Sciences": ["Healthcare", "Biology", "Pharmaceutical Sciences", "Nursing", "Biomedical Engineering", "Public Health"],
    "Math & Technology": ["Computer Science", "Mathematics", "Software Engineering", "Mechanical Engineering", "Data Science", "Electrical Engineering"],
    "Business & Finance": ["Finance", "Accounting", "Real Estate", "Investment Banking", "Taxation", "Supply Chain Management"],
    "Humanities & Social Sciences": ["Political Science", "Geography", "Sociology", "International Relations", "History", "Psychology"],
    "Arts & Design": ["Fine Arts", "Music", "Photography", "Videography", "Graphic Design", "Fashion Design"]
}

education_progression = {
    "Bachelor's": ["Freshman", "Sophomore", "Junior", "Senior"],
    "Master's": ["Year 1", "Year 2"]
}

# Mapping companies to fields
companies = {
    "Medical & Life Sciences": ["Pfizer", "Johnson & Johnson", "Mayo Clinic", "Moderna", "CDC"],
    "Math & Technology": ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Tesla"],
    "Business & Finance": ["Goldman Sachs", "JP Morgan", "Deloitte", "McKinsey", "Morgan Stanley"],
    "Humanities & Social Sciences": ["United Nations", "Brookings Institution", "National Geographic", "Amnesty International"],
    "Arts & Design": ["Pixar", "Disney", "Adobe", "Spotify", "Warner Bros"]
}

# Function to generate a realistic resume
def generate_resume():
    category = random.choice(list(categories.keys()))
    major = random.choice(categories[category])
    degree_level = random.choice(list(education_progression.keys()))
    year = random.choice(education_progression[degree_level])
    name = f"Candidate {random.randint(100, 999)}"
    skills = ", ".join(random.sample(categories[category], k=random.randint(3, 5)))
    
    # Determine experience and projects based on year
    projects, experience = [], ""
    
    if degree_level == "Bachelor's" and year == "Freshman":
        # Most freshmen have no major projects or internships
        if random.random() < 0.15:  # 15% chance of having something exceptional
            projects.append(f"{major} Personal Project")
            experience = f"Volunteer at {random.choice(companies[category])}"
    else:
        # More advanced students have relevant projects and internships
        projects = [f"{major} Capstone Project", f"Internship Research on {major}", f"Open-source contribution in {major}"]
        experience = f"Intern at {random.choice(companies[category])}"
    
    education = f"{degree_level} in {major}, Year: {year}"
    
    return {
        "Name": name,
        "Field": category,
        "Major": major,
        "Education": education,
        "Skills": skills,
        "Projects": ", ".join(projects) if projects else "N/A",
        "Experience": experience if experience else "N/A"
    }

# Function to generate a short job description
def generate_job_description(dummy_arg=None):
    job_titles = {
        "Medical & Life Sciences": ["Clinical Research Intern", "Lab Assistant", "Medical Data Analyst"],
        "Math & Technology": ["Software Engineer Intern", "Data Analyst Intern", "Cybersecurity Intern"],
        "Business & Finance": ["Investment Analyst Intern", "Marketing Intern", "Accounting Assistant"],
        "Humanities & Social Sciences": ["Policy Research Intern", "Museum Assistant", "Public Relations Intern"],
        "Arts & Design": ["Graphic Designer Intern", "Music Production Assistant", "Video Editing Intern"]
    }
    
    category = random.choice(list(job_titles.keys()))
    company = random.choice(companies[category])
    title = random.choice(job_titles[category])
    
    prompt = f"Generate a **short** job description (max 3 lines) for a {title} at {company}."
    
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    
    return response['message']['content']

# Function to generate dataset and save as CSV
def generate_dataset(num_resumes=10000, num_jobs=600):
    # Generate student resumes
    resumes = [generate_resume() for _ in range(num_resumes)]
    
    # Use multiprocessing for faster job description generation
    with multiprocessing.Pool(processes=4) as pool:
        job_descriptions = pool.map(generate_job_description, range(num_jobs))
    
    # Create DataFrames
    resumes_df = pd.DataFrame(resumes)
    job_descriptions_df = pd.DataFrame(job_descriptions, columns=["Job Description"])
    
    # Save CSV files
    resumes_df.to_csv("synthetic_resumes.csv", index=False)
    job_descriptions_df.to_csv("synthetic_job_descriptions.csv", index=False)
    
    return resumes_df, job_descriptions_df

# Main entry point
if __name__ == '__main__':
    generate_dataset()
