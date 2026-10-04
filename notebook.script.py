# Import necessary libraries
import pandas as pd

# Load the dataset and create a copy
ds_jobs = pd.read_csv("customer_train.csv")
ds_jobs_transformed = ds_jobs.copy()

# EDA to help identify ordinal, nominal, and two-factor categories
for col in ds_jobs.select_dtypes("object").columns:
    print(ds_jobs_transformed[col].value_counts(), '\n')

# Create a dictionary of columns containing ordered categorical data
ordered_cats = {
    'enrolled_university': ['no_enrollment', 'Part time course', 'Full time course'],
    'education_level': ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
    'experience': ['<1'] + list(map(str, range(1, 21))) + ['>20'],
    'company_size': ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'],
    'last_new_job': ['never', '1', '2', '3', '4', '>4']
}

# Create a mapping dictionary of columns containing two-factor categories to convert to Booleans
two_factor_cats = {
    'relevant_experience': {'No relevant experience': False, 'Has relevant experience': True},
    'job_change': {0.0: False, 1.0: True}
}

# Loop through DataFrame columns to efficiently change data types
for col in ds_jobs_transformed:
    
    # Convert two-factor categories to bool
    if col in ['relevant_experience', 'job_change']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].map(two_factor_cats[col])
    
    # Convert integer columns to int32
    elif col in ['student_id', 'training_hours']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('int32')
    
    # Convert float columns to float16
    elif col == 'city_development_index':
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('float16')
    
    # Convert columns containing ordered categorical data to ordered categories using dict
    elif col in ordered_cats.keys():
        category = pd.CategoricalDtype(ordered_cats[col], ordered=True)
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype(category)
    
    # Convert remaining columns to standard categories
    else:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('category')
        
# Filter students with 10 or more years experience at companies with at least 1000 employees
ds_jobs_transformed = ds_jobs_transformed[(ds_jobs_transformed['experience'] >= '10') & (ds_jobs_transformed['company_size'] >= '1000-4999')]