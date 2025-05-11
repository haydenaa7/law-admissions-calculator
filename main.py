# main.py

# Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt # For local testing/visualization
import joblib
import os

print("Starting model training script...")

# --- Cell 1 (Imports) ---
# Already done above

# --- Cell 2 (Load dataset) ---
print("Loading dataset...")
df = pd.read_csv('lsdata1.csv', low_memory=False)
df2 = pd.read_csv('lsdata2.csv', low_memory=False)
df = pd.concat([df, df2])
print(f"Initial df shape: {df.shape}")
# df.head() # For local viewing

# --- Cell 3 (Feature Engineering: Dates) ---
print("Performing date feature engineering...")
df['year'] = df['sent_at'].map((lambda x : x.split('-')[0]), na_action='ignore')
df['month'] = df['sent_at'].map((lambda x : x.split('-')[1]), na_action='ignore')
df['day'] = df['sent_at'].map((lambda x : x.split('-')[2]), na_action='ignore')
df['dayofweek'] = pd.to_datetime(df['sent_at']).dt.dayofweek

# --- Cell 4 (Drop rows where result is unknown) ---
print(f"Shape before dropping NA results: {df.shape}")
df = df.dropna(subset=['result']) # Corrected from 'result' (as in notebook)
print(f"Shape after dropping NA results: {df.shape}")

# --- Cell 5 (Filter for top schools) ---
print(f"Shape before filtering top schools: {df.shape}")
top_schools = df['school_name'].value_counts().iloc[:30].index.tolist()
print("Top 30 schools:", top_schools)
df = df[df['school_name'].isin(top_schools)]
print(f"Shape after filtering top schools: {df.shape}")

# --- Cell 6 (Coerce result values) ---
print(f"Shape before coercing results: {df.shape}")
mask_accepted = df['result'].str.contains('accept', case=False, na=False)
df.loc[mask_accepted, 'result'] = 'Accepted'
mask_rejected = df['result'].str.contains('reject', case=False, na=False)
df.loc[mask_rejected, 'result'] = 'Rejected'
mask_waitlist = df['result'].str.contains('waitlist', case=False, na=False)
df.loc[mask_waitlist, 'result'] = 'Waitlist'
df = df[df['result'].isin(['Accepted', 'Rejected', 'Waitlist'])]
# df = df.dropna(subset=['result']) # Already handled by the isin filter
print(f"Shape after coercing results: {df.shape}")

# --- Cell 7 (Define X and Y) ---
print("Defining X and Y variables...")
X_col = ['year', 'month', 'day', 'dayofweek', 'is_in_state', 'is_fee_waived', 'lsat', 'softs', 'urm', 'non_trad', 'gpa', 'is_international', 'international_gpa', 'years_out', 'is_military', 'is_character_and_fitness_issues', 'school_name']
Y_col = ['result']
X = df[X_col]
Y = df[Y_col].iloc[:, 0] # Ensure Y is a 1D array (ravel() or iloc)

# --- Cell 8 (Split data) ---
print("Splitting data into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Cell 9 (Identify feature types) ---
print("Identifying feature types...")
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
boolean_features = X_train.select_dtypes(include='boolean').columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Manual adjustment as in notebook for 'is_military' and 'is_character_and_fitness_issues'
# These might be loaded as object/category if they contain NaNs and pandas infers them that way
# before boolean conversion. If they are consistently True/False without NaNs then select_dtypes('boolean') would work.
# The safest approach is to explicitly define them if they cause issues.
bool_like_cols_to_move = []
for col in ['is_military', 'is_character_and_fitness_issues']:
    if col in categorical_features:
        categorical_features.remove(col)
        boolean_features.append(col)
    elif col in numerical_features: # Could happen if read as 0/1
        numerical_features.remove(col)
        boolean_features.append(col)


print("Numerical Features:", numerical_features)
print("Boolean Features:", boolean_features)
print("Categorical Features:", categorical_features)

# --- Cell 10 (Build transformers) ---
print("Building transformers...")
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
boolean_transformer = Pipeline(steps=[
    ('caster', FunctionTransformer(lambda x: x.astype(object))), # Cast to object first
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('final_caster', FunctionTransformer(lambda x: x.astype(bool))) # Cast back to bool if needed by model, or leave as object
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('string_caster', FunctionTransformer(lambda x: x.astype(str))),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Changed sparse_output to False for easier debugging, can be True for efficiency
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('bool', boolean_transformer, boolean_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough', # 'drop' if no other columns should go through
    sparse_threshold=0.0 # Ensure dense output from ColumnTransformer if sparse_output=False in OHE
)

# --- Cell 11 (Build pipeline with model) ---
print("Building model pipeline...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# --- Cell 12 (Train model) ---
print("Training model...")
# Y_train needs to be a 1D array for HistGradientBoostingClassifier
model.fit(X_train, Y_train) # Y_train is already 1D from the .iloc[:, 0]
print("Model training complete.")

# --- Cell 13 (Predict on test set - for local validation) ---
print("Predicting on the test set for local validation...")
output_preds = model.predict(X_test)

# --- Cell 14 (Show confusion matrix - for local validation) ---
# This will only work if you run this script in an environment with a display
try:
    print("Generating confusion matrix...")
    cm = confusion_matrix(Y_test, output_preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title("Confusion Matrix (Local Validation)")
    # plt.show() # Uncomment to display plot when running locally
    # Save the plot instead of showing for non-interactive environments
    os.makedirs('model_artifacts', exist_ok=True)
    plt.savefig(os.path.join('model_artifacts', 'confusion_matrix.png'))
    print(f"Confusion matrix saved to model_artifacts/confusion_matrix.png")
    plt.close()
except Exception as e:
    print(f"Could not generate/save confusion matrix plot: {e}")


# --- Cell 15 (Show classification report - for local validation) ---
print("Classification Report (Local Validation):\n")
print(classification_report(Y_test, output_preds))

# --- NEW: Save the model, classes, and top_schools list ---
print("Saving model artifacts...")
ARTIFACTS_DIR = 'model_artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

model_path = os.path.join(ARTIFACTS_DIR, 'law_school_admission_model.joblib')
joblib.dump(model, model_path)
print(f"Model pipeline saved to {model_path}")

classes_path = os.path.join(ARTIFACTS_DIR, 'model_classes.joblib')
joblib.dump(model.classes_, classes_path)
print(f"Model classes saved to {classes_path}")

schools_path = os.path.join(ARTIFACTS_DIR, 'top_schools.joblib')
joblib.dump(top_schools, schools_path)
print(f"Top schools list saved to {schools_path}")

# Save X_col for reference (optional but good for debugging api/predict.py)
# x_col_path = os.path.join(ARTIFACTS_DIR, 'X_col_order.joblib')
# joblib.dump(X_col, x_col_path)
# print(f"X_col (feature order) saved to {x_col_path}")

print("Script finished.")

# --- Cell 16 (Get individual result - this logic is now in api/predict.py) ---
# The individual prediction part is not needed here as it's for the web app.
# You can test it locally like this if you want:
#
# print("\n--- Testing individual prediction locally ---")
# example_data = {
#     'year': [2024], 'month': [9], 'day': [25],
#     'dayofweek': [pd.to_datetime("2024-09-25").dayofweek], # Calculate dayofweek
#     'is_in_state': [False], 'is_fee_waived': [False], 'lsat': [172],
#     'softs': ['T3'], 'urm': [True], 'non_trad': [False], 'gpa': [3.83],
#     'is_international': [False], 'international_gpa': [np.nan], 'years_out': [8],
#     'is_military': [False], 'is_character_and_fitness_issues': [False],
#     'school_name': ['University of California—Los Angeles']
# }
# example_df = pd.DataFrame(example_data)
#
# local_pred_proba = model.predict_proba(example_df)[0]
# output_dict_local = {model.classes_[i]: prob for i, prob in enumerate(local_pred_proba)}
# print("Local prediction probabilities:", output_dict_local)
# most_likely_local = max(output_dict_local, key=output_dict_local.get)
# print(f"Most likely outcome (local test): {most_likely_local}")
