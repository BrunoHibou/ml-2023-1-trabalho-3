import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


new_data=pd.read_csv('ds_salaries.csv')

le = preprocessing.LabelEncoder()
scaler = StandardScaler()
to_drop = ['salary_currency', 'remote_ratio', 'work_year', 'salary']
data = new_data.drop(columns=to_drop)

# Converting string labels into numbers.
data['experience_level']=le.fit_transform(data['experience_level'])
data['job_title']=le.fit_transform(data['job_title'])
data['employment_type']=le.fit_transform(data['employment_type'])
data['company_size']=le.fit_transform(data['company_size'])
data['employee_residence']=le.fit_transform(data['employee_residence'])
data['company_location']=le.fit_transform(data['company_location'])

to_normalize = ['experience_level',
                'job_title', 
                'employee_residence',
                'employment_type',
                'company_size',
                'company_location']

scaler.fit(data[to_normalize])

data_normalized = data.copy()  # Create a copy of the original dataset
data_normalized[to_normalize] = scaler.transform(data[to_normalize])

# Spliting data into Feature and
X=data_normalized[['experience_level', 'job_title', 'employment_type', 'company_size', 'employee_residence', 'company_location']]
y=data_normalized['salary_in_usd']

print(data_normalized.head())
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, # 70% training and 30% test
                                                    random_state=42)  

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(16,32),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01,
                    max_iter=1000)

# Fit data onto the model
clf.fit(X_train,y_train)

# Make prediction on test dataset
ypred=clf.predict(X_test)

# Calcuate accuracy
accuracy = accuracy_score(y_test,ypred)
print(f"accuracy: {accuracy*100}")
print("expected accuracy: 90%+")
