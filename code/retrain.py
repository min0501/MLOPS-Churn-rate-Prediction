import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from library import EnsemblePipeline

# Load the original training data
original_data = joblib.load('original_training_data.pkl')
X_train_original = original_data['X_train']
y_train_original = original_data['y_train']

# Load new data
data = pd.read_csv('New_고객이탈률데이터.csv')

# Keep the necessary columns
client_nums = data['고객번호']
genders = data['성별']
income_categories = data['소득']
education_levels = data['교육수준']
marital_statuses = data['결혼상태']

# Prepare data
data['이탈여부'] = data['이탈여부'].map({'기존고객': 0, '이탈고객': 1})

# Extract features and target
X_new_data = data.drop(columns=['이탈여부','고객번호'])
y_new_data = data['이탈여부']

# Split new data into training and test sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_data, y_new_data, test_size=0.2, random_state=42)

# Combine original and new training data
X_train_combined = pd.concat([X_train_original, X_train_new], axis=0)
y_train_combined = pd.concat([y_train_original, y_train_new], axis=0)

# Load the initial model
ensemble_model = joblib.load('initial_model.pkl')

# Retrain the model using combined data
ensemble_model.fit(X_train_combined, y_train_combined)

# Evaluate model on the combined training data
y_train_combined_pred = ensemble_model.predict(X_train_combined)
model_performance = classification_report(y_train_combined, y_train_combined_pred['ensemble'].astype(int), output_dict=True)
print("Retrained model evaluation on combined training data")
print(classification_report(y_train_combined, y_train_combined_pred['ensemble'].astype(int), output_dict=True))
# Save the retrained model
joblib.dump(ensemble_model, 'retrained_model.pkl')