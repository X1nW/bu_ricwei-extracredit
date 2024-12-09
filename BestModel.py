import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve
import xgboost as xgb

# Feature Engineering Function
def feature_engineering(df):
    df['trans_hour'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S').dt.hour
    df['trans_day'] = pd.to_datetime(df['trans_date']).dt.day
    df['trans_month'] = pd.to_datetime(df['trans_date']).dt.month
    df['age'] = pd.to_datetime('today').year - pd.to_datetime(df['dob']).dt.year

    # Calculate distance between cardholder and merchant
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of Earth in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    return df

# Load Data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Apply Feature Engineering
train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

# Define Features and Target
features = ['amt', 'category', 'trans_hour', 'trans_day', 'trans_month', 'distance', 'age', 'city_pop', 'gender']
target = 'is_fraud'

X = train_data[features]
y = train_data[target]

# Define Preprocessing Pipeline
categorical_features = ['category', 'gender']
numerical_features = ['amt', 'trans_hour', 'trans_day', 'trans_month', 'distance', 'age', 'city_pop']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define Individual Models
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt'
    ))
])

xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        random_state=42,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'  # Keep this to specify evaluation metric
    ))
])

# Define Ensemble Model
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'
)

# Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(ensemble_model, X, y, cv=skf, scoring='roc_auc')

print("Cross-Validation Scores:", cross_val_scores)
print("Mean ROC AUC:", np.mean(cross_val_scores))

# Train the Ensemble Model on Full Training Data
ensemble_model.fit(X, y)

def find_optimal_threshold_ensemble(model, X, y):
    val_proba = model.predict_proba(X)[:, 1]  # Get probabilities from ensemble
    precision, recall, thresholds = precision_recall_curve(y, val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

optimal_threshold = find_optimal_threshold_ensemble(ensemble_model, X, y)
print(f"Optimal Threshold for Ensemble: {optimal_threshold}")

# Generate Predictions for Test Set with Optimal Threshold
test_preds = (ensemble_model.predict_proba(test_data[features])[:, 1] > optimal_threshold).astype(int)

# Prepare Submission File
submission = test_data[['id']].copy()
submission['is_fraud'] = test_preds
submission.to_csv('submission.csv', index=False)

print(f"Submission file 'submission.csv' created successfully with threshold {optimal_threshold:.2f}!")