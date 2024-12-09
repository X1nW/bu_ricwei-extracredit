import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import xgboost as xgb
from catboost import CatBoostClassifier

# Feature Engineering Function
def feature_engineering(df, is_train=True):
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

    # Interaction Feature
    df['amt_distance_ratio'] = df['amt'] / (df['distance'] + 1)

    # Historical Features
    if is_train:
        user_avg_amt = df.groupby('cc_num')['amt'].transform('mean')
        merchant_fraud_rate = df.groupby('merchant')['is_fraud'].transform('mean')
    else:
        user_avg_amt = train_data.groupby('cc_num')['amt'].mean()
        merchant_fraud_rate = train_data.groupby('merchant')['is_fraud'].mean()

    df['user_avg_amt'] = df['cc_num'].map(user_avg_amt).fillna(0)
    df['merchant_risk'] = df['merchant'].map(merchant_fraud_rate).fillna(0)

    # Time-Based Features
    df['is_weekend'] = pd.to_datetime(df['trans_date']).dt.weekday.isin([5, 6]).astype(int)

    # Add Feature: Transaction amount compared to average user transaction
    df['amt_user_diff'] = df['amt'] - df['user_avg_amt']

    # Add Feature: Ratio of transaction amount to city population
    df['amt_city_ratio'] = df['amt'] / (df['city_pop'] + 1)

    # Frequency Encoding
    category_freq = df['category'].value_counts().to_dict()
    df['category_freq'] = df['category'].map(category_freq).fillna(0)

    return df

# Load Data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Apply Feature Engineering
train_data = feature_engineering(train_data, is_train=True)
test_data = feature_engineering(test_data, is_train=False)

# Define Features and Target
features = ['amt', 'category', 'trans_hour', 'trans_day', 'trans_month', 'distance', 'age', 'city_pop', 'gender']
target = 'is_fraud'

X = train_data[features]
y = train_data[target]

# Define Preprocessing Pipeline for RandomForest and XGBoost
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
        n_estimators=800,  # Increase estimators
        max_depth=30,  # Increase depth
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt'
    ))
])

xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        random_state=42,
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss'
    ))
])

# Train CatBoost Separately
cat_X = train_data[features]
cat_y = train_data[target]
cat_model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    cat_features=['category', 'gender'],
    verbose=0
)
cat_model.fit(cat_X, cat_y)

# Define Voting Ensemble without CatBoost in Pipeline
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

# Find Optimal Threshold for Ensemble
def find_optimal_threshold_ensemble(model, X, y):
    val_proba = model.predict_proba(X)[:, 1]  # Get probabilities from ensemble
    precision, recall, thresholds = precision_recall_curve(y, val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

optimal_threshold = find_optimal_threshold_ensemble(ensemble_model, X, y)
print(f"Optimal Threshold for Ensemble: {optimal_threshold}")

# Generate Predictions for Test Set with Dynamic Threshold
test_data['dynamic_threshold'] = test_data['merchant_risk'].apply(lambda x: 0.4 if x > 0.2 else optimal_threshold)
test_preds = (ensemble_model.predict_proba(test_data[features])[:, 1] > test_data['dynamic_threshold']).astype(int)

# Prepare Submission File
submission = test_data[['id']].copy()
submission['is_fraud'] = test_preds
submission.to_csv('submission.csv', index=False)

print(f"Submission file 'submission.csv' created successfully!")