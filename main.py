# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib
# Set the matplotlib backend to 'Agg' for non-interactive environments, like script execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset for mobile security logs
data = pd.DataFrame({
    'data_usage': np.random.normal(120, 40, 1000),  # Mean data usage with some standard deviation
    'login_attempts': np.random.poisson(1, 1000),  # Average number of login attempts
    'location_access': np.random.binomial(1, 0.2, 1000),  # 20% chance of location access attempt
    'malicious': np.random.binomial(1, 0.1, 1000)  # 10% are malicious activities
})

# Features and Labels
X = data[['data_usage', 'login_attempts', 'location_access']]
y = data['malicious']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predicting the test set results
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save the model to disk
joblib.dump(model, 'mobile_security_rf_model.pkl')

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Mobile Security Model')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix.png')  # Save the plot as a PNG file
plt.close()
