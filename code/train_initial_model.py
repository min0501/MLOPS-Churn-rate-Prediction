import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from library import EnsemblePipeline

# Load initial data
data = pd.read_csv('고객이탈률데이터.csv')

# Prepare data
data['이탈여부'] = data['이탈여부'].map({'기존고객': 0, '이탈고객': 1})

X = data.drop(columns=['이탈여부','고객번호'])
y = data['이탈여부']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the original training data
original_data = {
    'X_train': X_train,
    'y_train': y_train
}
joblib.dump(original_data, 'original_training_data.pkl')

# Train model
ensemble_model = EnsemblePipeline()
ensemble_model.fit(X_train, y_train)

get_feature_names = ensemble_model.get_feature_names()
print("Final feature names used in the model:", get_feature_names)

# Evaluate model
y_train_pred = ensemble_model.predict(X_train)
print("Training data evaluation")
print(classification_report(y_train, y_train_pred['ensemble'].astype(int)))

y_test_pred = ensemble_model.predict(X_test)
print("Test data evaluation")
print(classification_report(y_test, y_test_pred['ensemble'].astype(int)))

# Save the trained model
joblib.dump(ensemble_model, 'initial_model.pkl')
