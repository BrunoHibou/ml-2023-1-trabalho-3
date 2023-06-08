import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

new_data = pd.read_csv('ds_salaries.csv')

le = preprocessing.LabelEncoder()

# Dropping columns that are not relevant
to_drop = ['salary_currency', 'remote_ratio', 'work_year', 'salary']
data = new_data.drop(columns=to_drop)

# Converting string labels into numbers
data['experience_level'] = le.fit_transform(data['experience_level'])
data['job_title'] = le.fit_transform(data['job_title'])
data['employment_type'] = le.fit_transform(data['employment_type'])
data['company_size'] = le.fit_transform(data['company_size'])
data['employee_residence'] = le.fit_transform(data['employee_residence'])
data['company_location'] = le.fit_transform(data['company_location'])

# Split dataset into features and target
X = data[['experience_level', 'job_title', 'employment_type', 'company_size', 'employee_residence', 'company_location']]
y = data['salary_in_usd']

# Create model object
regressor = MLPRegressor(hidden_layer_sizes=(16, 32), random_state=5, verbose=True, learning_rate_init=0.01, max_iter=1000, activation='relu')

# Fit data to the model
regressor.fit(X, y)

# User input section to provide feature values for prediction
input_values = [3, 84, 2, 0, 26, 25]
input_values = le.fit_transform(input_values)

'''
for feature in ['experience_level', 'job_title', 'employment_type', 'company_size', 'employee_residence', 'company_location']:
    value = input(f"Enter value for '{feature}': ")
    input_values.append(float(value))
'''

# Make prediction using the input values
predicted_salary = regressor.predict([input_values])

print("Predicted Salary (USD):", predicted_salary[0])
print(data.head())
