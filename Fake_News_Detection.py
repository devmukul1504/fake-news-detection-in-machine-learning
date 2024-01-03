import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load the datasets
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Display a few rows from each dataset
print(true_data[0:10])
print(fake_data[0:4])

# Add labels to the datasets
fake_data['label'] = 1
true_data['label'] = 0

# Concatenate the datasets and shuffle the rows
all_data = pd.concat([fake_data, true_data])
all_data = all_data.iloc[np.random.permutation(len(all_data))]

# Select relevant columns and create a new feature combining title, text, and subject
filtered_data = all_data.loc[:, ['title', 'text', 'subject', 'label']]
filtered_data['training_feature'] = filtered_data['title'] + ' ' + filtered_data['text'] + ' ' + filtered_data['subject']

# Check for null values
filtered_data.isnull().sum()

# Split the data into features (X) and labels (y)
X = filtered_data['training_feature'].values
y = filtered_data['label']

# Sample a smaller subset for demonstration purposes
l_X = filtered_data['training_feature'].values[0:1000]
l_Y = filtered_data['label'].values[0:1000]

# Create TF-IDF vectorizers for the entire dataset and the smaller subset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

l_vectorizer = TfidfVectorizer()
l_X = l_vectorizer.fit_transform(l_X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
l_X_train, l_X_test, l_Y_train, l_Y_test = train_test_split(l_X, l_Y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)

# Evaluate the Logistic Regression model
test_y_hat_lr = model_lr.predict(X_test)
train_y_hat_lr = model_lr.predict(X_train)

print("Logistic Regression Model:")
print("Accuracy on test set:", accuracy_score(test_y_hat_lr, Y_test))
print("Precision:", precision_score(Y_test, test_y_hat_lr))
print("Recall:", recall_score(Y_test, test_y_hat_lr))
print("F1 Score:", f1_score(Y_test, test_y_hat_lr))
print("Confusion Matrix:\n", confusion_matrix(Y_test, test_y_hat_lr))

# Train a SVM model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(l_X_train, l_Y_train)

# Evaluate the SVM model
y_pred_svm = model_svm.predict(l_X_test)

print("\nSVM Model:")
print("Accuracy on test set:", accuracy_score(y_pred_svm, l_Y_test))
print("Precision:", precision_score(l_Y_test, y_pred_svm))
print("Recall:", recall_score(l_Y_test, y_pred_svm))
print("F1 Score:", f1_score(l_Y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(l_Y_test, y_pred_svm))

# Train a Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(l_X_train, l_Y_train)

# Evaluate the Random Forest model
y_pred_rf = model_rf.predict(l_X_test)

print("\nRandom Forest Model:")
print("Accuracy on test set:", accuracy_score(y_pred_rf, l_Y_test))
print("Precision:", precision_score(l_Y_test, y_pred_rf))
print("Recall:", recall_score(l_Y_test, y_pred_rf))
print("F1 Score:", f1_score(l_Y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(l_Y_test, y_pred_rf))

# Train a Naive Bayes model
model_nb = GaussianNB()
model_nb.fit(l_X_train.toarray(), l_Y_train)

# Evaluate the Naive Bayes model
y_pred_nb = model_nb.predict(l_X_test.toarray())

print("\nNaive Bayes Model:")
print("Accuracy on test set:", accuracy_score(y_pred_nb, l_Y_test))
print("Precision:", precision_score(l_Y_test, y_pred_nb))
print("Recall:", recall_score(l_Y_test, y_pred_nb))
print("F1 Score:", f1_score(l_Y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(l_Y_test, y_pred_nb))import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load the datasets
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Display a few rows from each dataset
print(true_data[0:10])
print(fake_data[0:4])

# Add labels to the datasets
fake_data['label'] = 1
true_data['label'] = 0

# Concatenate the datasets and shuffle the rows
all_data = pd.concat([fake_data, true_data])
all_data = all_data.iloc[np.random.permutation(len(all_data))]

# Select relevant columns and create a new feature combining title, text, and subject
filtered_data = all_data.loc[:, ['title', 'text', 'subject', 'label']]
filtered_data['training_feature'] = filtered_data['title'] + ' ' + filtered_data['text'] + ' ' + filtered_data['subject']

# Check for null values
filtered_data.isnull().sum()

# Split the data into features (X) and labels (y)
X = filtered_data['training_feature'].values
y = filtered_data['label']

# Sample a smaller subset for demonstration purposes
l_X = filtered_data['training_feature'].values[0:1000]
l_Y = filtered_data['label'].values[0:1000]

# Create TF-IDF vectorizers for the entire dataset and the smaller subset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

l_vectorizer = TfidfVectorizer()
l_X = l_vectorizer.fit_transform(l_X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
l_X_train, l_X_test, l_Y_train, l_Y_test = train_test_split(l_X, l_Y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)

# Evaluate the Logistic Regression model
test_y_hat_lr = model_lr.predict(X_test)
train_y_hat_lr = model_lr.predict(X_train)

print("Logistic Regression Model:")
print("Accuracy on test set:", accuracy_score(test_y_hat_lr, Y_test))
print("Precision:", precision_score(Y_test, test_y_hat_lr))
print("Recall:", recall_score(Y_test, test_y_hat_lr))
print("F1 Score:", f1_score(Y_test, test_y_hat_lr))
print("Confusion Matrix:\n", confusion_matrix(Y_test, test_y_hat_lr))

# Train a SVM model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(l_X_train, l_Y_train)

# Evaluate the SVM model
y_pred_svm = model_svm.predict(l_X_test)

print("\nSVM Model:")
print("Accuracy on test set:", accuracy_score(y_pred_svm, l_Y_test))
print("Precision:", precision_score(l_Y_test, y_pred_svm))
print("Recall:", recall_score(l_Y_test, y_pred_svm))
print("F1 Score:", f1_score(l_Y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(l_Y_test, y_pred_svm))

# Train a Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(l_X_train, l_Y_train)

# Evaluate the Random Forest model
y_pred_rf = model_rf.predict(l_X_test)

print("\nRandom Forest Model:")
print("Accuracy on test set:", accuracy_score(y_pred_rf, l_Y_test))
print("Precision:", precision_score(l_Y_test, y_pred_rf))
print("Recall:", recall_score(l_Y_test, y_pred_rf))
print("F1 Score:", f1_score(l_Y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(l_Y_test, y_pred_rf))

# Train a Naive Bayes model
model_nb = GaussianNB()
model_nb.fit(l_X_train.toarray(), l_Y_train)

# Evaluate the Naive Bayes model
y_pred_nb = model_nb.predict(l_X_test.toarray())

print("\nNaive Bayes Model:")
print("Accuracy on test set:", accuracy_score(y_pred_nb, l_Y_test))
print("Precision:", precision_score(l_Y_test, y_pred_nb))
print("Recall:", recall_score(l_Y_test, y_pred_nb))
print("F1 Score:", f1_score(l_Y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(l_Y_test, y_pred_nb))
