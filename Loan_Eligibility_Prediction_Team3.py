#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# # Data Gathering

# In[62]:


df=pd.read_csv('https://raw.githubusercontent.com/DistributedandScalableEngineeringTeam9/Loan-Eligibility-Prediction/main/loan_approval_dataset.csv')
df


# In[63]:


df.isnull().sum()


# In[30]:


df.dtypes


# # Univariate & Bivariate Analysis - Visualization

# In[72]:


df.columns = df.columns.str.strip()

if 'education' in df.columns and 'loan_status' in df.columns:
    education_loan_status_counts = df.groupby(['education', 'loan_status']).size().unstack()

    plt.figure(figsize=(4, 2))
    education_loan_status_counts.plot(kind='bar', stacked=True, figsize = (6,4))
    plt.title('Loan Status by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper left', labels=['Approved', 'Rejected'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.show()


# In[74]:


plt.figure(figsize=(6, 4))
plt.hist(df['loan_amount'], bins=20, color='orange')
plt.title('Histogram of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[81]:


loan_status_counts = df['loan_status'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(loan_status_counts.index, loan_status_counts.values, color=['green', (1.0, 0.8, 0.8)])
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()


# In[85]:


education_counts = df['education'].value_counts()

plt.figure(figsize=(6, 4))
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', colors=['skyblue', 'grey'])
plt.title('Education Distribution')
plt.show()


# In[86]:


plt.figure(figsize=(6, 4))
sns.histplot(data=df,x='cibil_score',bins=4,hue='loan_status')
plt.title('cibil')
plt.show()


# In[36]:


numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[37]:


r=2
c=2
it=1
plt.figure(figsize=(8,6))
for i in ['residential_assets_value', 'commercial_assets_value','luxury_assets_value', 'bank_asset_value']:
    plt.subplot(r,c,it)
    sns.regplot(x=i,y='loan_amount',data=df)
    plt.grid()
    it+=1
plt.tight_layout()
plt.show()


# # Data Preprocessing

# In[38]:


df.columns


# In[39]:


df.drop(columns=['loan_id'], inplace=True)
print(df.columns)
df.rename(columns=lambda x: x.strip(), inplace=True)
print(df.columns)


# In[40]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['education'] = label_encoder.fit_transform(df['education'])
df['self_employed'] = label_encoder.fit_transform(df['self_employed'])
df['loan_status'] = label_encoder.fit_transform(df['loan_status'])
print(df[['education', 'self_employed','loan_status']])


# In[41]:


df.head(5)


# In[42]:


df.dtypes


# In[43]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = df.drop(columns=['loan_status'])  
y = df['loan_status']  # Target variable

numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

scaling_parameters = {
    'mean': scaler.mean_,
    'std': scaler.scale_
}

with open('scaling_parameters.pkl', 'wb') as file:
    pickle.dump(scaling_parameters, file)

print("Scaled Feature Variables (x):")
print(x.head())

print("\nTarget Variable (y):")
print(y.head())


# # Data Modeling & Evaluation

# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=555)


# ## Logistic Regression

# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logistic_reg = LogisticRegression(random_state=42)

logistic_reg.fit(x_train, y_train)
y_pred = logistic_reg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

classification_rep = classification_report(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Log Reg Confusion Matrix')
plt.show()

log_reg_accuracy = accuracy

print("Logistic Regression Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("Classification Report:\n", classification_rep)


# ## DecisionTree Classifier

# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DecisionTreeClassifier instance
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the decision tree model
decision_tree.fit(x_train, y_train)

# Predict on the test set
y_pred = decision_tree.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(5,3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()

dec_tree_accuracy = accuracy

# Print evaluation metrics separately
print("Decision Tree Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:\n", classification_rep)


# ## RandomForest Classifier

# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a RandomForestClassifier instance
random_forest = RandomForestClassifier(random_state=42)

# Train the random forest model
random_forest.fit(x_train, y_train)

# Predict on the test set
y_pred = random_forest.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

rf_accuracy = accuracy 

# Print evaluation metrics separately
print("Random Forest Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:\n", classification_rep)


# ## KNN Classifier

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a KNeighborsClassifier instance with a specified number of neighbors (e.g., n_neighbors=5)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn_classifier.fit(x_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()

knn_accuracy = accuracy

# Print evaluation metrics separately
print("K-Nearest Neighbors Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:\n", classification_rep)


# In[49]:

# ## XGBoost Classifier

# In[50]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create an XGBClassifier instance
xgb_classifier = XGBClassifier(random_state=42)

# Train the XGBoost model
xgb_classifier.fit(x_train, y_train)

# Predict on the test set
y_pred = xgb_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.show()

xgb_accuracy = accuracy

# Print evaluation metrics separately
print("XGBoost Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:\n", classification_rep)


# ## Hybrid - Voting Classifier

# In[51]:


from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create individual classifiers
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
xgb_classifier = XGBClassifier(random_state=42)

# Create an ensemble of classifiers using VotingClassifier
ensemble_classifier = VotingClassifier(estimators=[
    ('decision_tree', decision_tree),
    ('random_forest', random_forest),
    ('xgb_classifier', xgb_classifier)
], voting='hard')  # 'hard' for majority vote

# Train the ensemble model
ensemble_classifier.fit(x_train, y_train)

# Predict on the test set
y_pred = ensemble_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Voting Classifier Confusion Matrix')
plt.show()

voting_accuracy = accuracy 
# Print evaluation metrics separately
print("Ensemble Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:\n", classification_rep)


# In[88]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame with model names and their corresponding accuracy values
import pandas as pd

data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'XGBoost', 'Voting Classifier'],
    'Accuracy': [log_reg_accuracy, dec_tree_accuracy, rf_accuracy, knn_accuracy, xgb_accuracy, voting_accuracy]
}

df = pd.DataFrame(data)

# Creating a vertical bar plot with small font size for values displayed on top using seaborn
plt.figure(figsize=(6, 6))
ax = sns.barplot(x='Model', y='Accuracy', data=df, color='skyblue')

# Displaying values on top of each bar with small font size
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

    
# Adjusting font size of X-axis labels
plt.xticks(rotation='vertical', fontsize=8)

# Adjusting font size of Y-axis labels
plt.yticks(fontsize=8)

plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.show()


# In[53]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

models = [logistic_reg, decision_tree, random_forest, xgb_classifier, knn_classifier]

# Plot AUC-ROC curves for each model
plt.figure(figsize=(8, 6))

for model in models:
    # Train the model
    model.fit(x_train, y_train)
    
    # Get probability scores for the positive class on the test set
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(x_test)[:, 1]  # Probability of positive class
    else:
        y_score = model.decision_function(x_test)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=[0, 1]), y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc:.4f})')

# Plot the random line
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve for Multiple Models')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        print("Data received:", data)

        
        
        education_mapping = {'Not Graduate': 1, 'Graduate': 0}
        data['education'] = education_mapping.get(data['education'], data['education'])

        self_employed_mapping = {'No': 0, 'Yes': 1}
        data['self_employed'] = self_employed_mapping.get(data['self_employed'], data['self_employed'])
        
        data = pd.DataFrame(data, index=[0])
        numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

                # Load the scaling parameters from the file
        with open('scaling_parameters.pkl', 'rb') as file:
            loaded_scaling_parameters = pickle.load(file)

        # Create a new StandardScaler with the loaded parameters
        scaler = StandardScaler()
        scaler.mean_ = loaded_scaling_parameters['mean']
        scaler.scale_ = loaded_scaling_parameters['std']

        # Use the scaler to transform new_data
        data[numerical_columns] = scaler.transform(data[numerical_columns])
        
        print(data)
        data = data.astype(int)
        # new_data = list(data.values())
        # print(new_data
        prediction = xgb_classifier.predict(data)
        
        print(prediction)
        
        if prediction[0] == 0:
            prediction = 'Approved'
        else:
            prediction = 'Rejected'
            
        print("As per the details provided, you loan will be: ", prediction)
        return render_template('result.html', prediction_result=prediction)
    
    except Exception as e:
        print(f"An error occurred: {
e}")
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run()


# In[ ]:


# no_of_dependents                   5
# education                          1
# self_employed                      1
# income_annum                 3400000
# loan_amount                  9400000
# loan_term                         16
# cibil_score                      535
# residential_assets_value     3100000
# commercial_assets_value      5600000
# luxury_assets_value         12900000
# bank_asset_value             3200000
# loan_status                        1


# no_of_dependents                   4
# education                          0
# self_employed                      1
# income_annum                 5900000
# loan_amount                 17300000
# loan_term                         20
# cibil_score                      657
# residential_assets_value     1500000
# commercial_assets_value      5600000
# luxury_assets_value         13000000
# bank_asset_value             4100000
# loan_status                        0


# In[ ]:


df.iloc[1525]


# In[ ]:


x_train.iloc[1525]

