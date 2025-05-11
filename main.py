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
import matplotlib.pyplot as plt
import joblib
import os
from backend_service.custom_transformers import cast_to_object, cast_to_bool, cast_to_string

# Load dataset
df = pd.read_csv('lsdata1.csv', low_memory=False)
df2 = pd.read_csv('lsdata2.csv', low_memory=False)
df = pd.concat([df, df2])
print(df.shape)
# df.head()

# Only consider year and month for sent date
df['year'] = df['sent_at'].map((lambda x : x.split('-')[0]), na_action='ignore')
df['month'] = df['sent_at'].map((lambda x : x.split('-')[1]), na_action='ignore')
df['day'] = df['sent_at'].map((lambda x : x.split('-')[2]), na_action='ignore')
df['dayofweek'] = pd.to_datetime(df['sent_at']).dt.dayofweek

# Drop rows where result is unknown
print(f"Shape before: {df.shape}")
df = df.dropna(subset=['result']) # Corrected from original notebook 'result' to ['result']
print(f"Shape after: {df.shape}")

# Only keep results for top 198 most frequently applied schools -- may not have enough data for others
print(f"Shape before: {df.shape}")
top_schools = df['school_name'].value_counts().iloc[:198].index.tolist()
print(top_schools)
with open('schools.txt', 'w') as f:
    for school in top_schools:
        f.write(school + '\n')
df = df[df['school_name'].isin(top_schools)]
print(f"Shape after: {df.shape}")

# Coerce result values into Accepted, Rejected, Waitlisted -- everything else can be dropped
print(f"Shape before: {df.shape}")
mask_accepted = df['result'].str.contains('accept', case=False, na=False)
df.loc[mask_accepted, 'result'] = 'Accepted'
mask_rejected = df['result'].str.contains('reject', case=False, na=False)
df.loc[mask_rejected, 'result'] = 'Rejected'
mask_waitlist = df['result'].str.contains('waitlist', case=False, na=False)
df.loc[mask_waitlist, 'result'] = 'Waitlist'
df = df[df['result'].isin(['Accepted', 'Rejected', 'Waitlist'])]
# df = df.dropna(subset=['result']) # This line is likely redundant after the .isin filter
print(f"Shape after: {df.shape}")

# Get dependent and independent variables
X_col = ['year', 'month', 'day', 'dayofweek', 'is_in_state', 'is_fee_waived', 'lsat', 'softs', 'urm', 'non_trad', 'gpa', 'is_international', 'international_gpa', 'years_out', 'is_military', 'is_character_and_fitness_issues', 'school_name']
Y_col = ['result']
X = df[X_col]
Y = df[Y_col].iloc[:, 0] # Ensure Y is 1D

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Get numerical and categorical features for preprocessing
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
boolean_features_initial = X_train.select_dtypes(include='boolean').columns.tolist() # Will be refined
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Refine boolean_features: move specific columns if they were misclassified
boolean_candidates = ['is_in_state', 'is_fee_waived', 'urm', 'non_trad', 'is_international', 'is_military', 'is_character_and_fitness_issues']
actual_boolean_features = []

for col in boolean_candidates:
    if col in X_train.columns: # Check if column actually exists
        if col in numerical_features:
            numerical_features.remove(col)
            actual_boolean_features.append(col)
        elif col in categorical_features:
            categorical_features.remove(col)
            actual_boolean_features.append(col)
        elif col in boolean_features_initial: # Already correctly identified
            actual_boolean_features.append(col)
        # If a boolean candidate is not in any list, it might mean it was all NaNs and dropped or not in X_train
        # Add it to boolean list if it was intended to be one and handle NaNs in transformer
        elif col not in actual_boolean_features: # if it's truly missing but should be boolean
             actual_boolean_features.append(col)


# Ensure no overlap and all boolean candidates are handled
boolean_features = list(set(actual_boolean_features)) # Use set to ensure uniqueness

print("Numerical Features:", numerical_features)
print("Boolean Features:", boolean_features)
print("Categorical Features:", categorical_features)


# Build individual transformers and combine
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

boolean_transformer = Pipeline(steps=[
    ('caster_to_object', FunctionTransformer(cast_to_object)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('caster_to_bool', FunctionTransformer(cast_to_bool))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('string_caster', FunctionTransformer(cast_to_string)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('bool', boolean_transformer, boolean_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough',
    sparse_threshold=0.0
)

# Build pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train model
model.fit(X_train, Y_train)

# Predict on the test set
output = model.predict(X_test)

# Show confusion matrix
ARTIFACTS_DIR = 'model_artifacts' # Define ARTIFACTS_DIR earlier
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
try:
    cm = confusion_matrix(Y_test, output, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    # plt.show() # Commented out for non-interactive runs
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to {ARTIFACTS_DIR}/confusion_matrix.png")
except Exception as e:
    print(f"Could not generate/save confusion matrix plot: {e}")


# Show validation statistics
print(classification_report(Y_test, output))


# Save the model, classes, and top_schools list
print("Saving model artifacts...")

model_path = os.path.join(ARTIFACTS_DIR, 'law_school_admission_model.joblib')
joblib.dump(model, model_path)
print(f"Model pipeline saved to {model_path}")

classes_path = os.path.join(ARTIFACTS_DIR, 'model_classes.joblib')
joblib.dump(model.classes_, classes_path)
print(f"Model classes saved to {classes_path}")

schools_path = os.path.join(ARTIFACTS_DIR, 'top_schools.joblib')
joblib.dump(top_schools, schools_path)
print(f"Top schools list saved to {schools_path}")

print("Script finished.")
