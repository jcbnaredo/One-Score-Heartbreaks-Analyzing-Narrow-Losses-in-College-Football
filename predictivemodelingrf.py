## PREDICTIVE MODELING 
## DESCRIPTION: THIS SCRIPT BUILDS AND EVALUATES A RANDOM FOREST CLASSIFIER MODEL TO PREDICT OUTCOMES FOR ONE-SCORE GAMES IN COLLEGE FOOTBALL
## USING TEAM STATS FROM PREVIOUS FBS SEASONS

## MAIN SUMMARY OF THE PROCESS
# 1) LOAD HISTORICAL TEAM STATS FOR WINNERS AND LOSERS (2018, 2019, 2021-2023) AS TRAINING DATA
# 2) LOAD TEAM STATS FOR A TARGET SEASON (2024) AS TEST DATA
# 3) SELECT HYPERPARAMETERS FOR THE RANDOM FOREST CLASSIFIER USING OPTUNA
# 4) TRAIN A RANDOM FOREST CLASSIFIER MODEL ON THE TRAINING DATA
# 5) PREDICT AND EVALUATE PERFORMANCE ON THE TEST DATA USING THE FOLLOWING:
# - ACCURACY
# - CONFUSION MATRIX
# - CLASSIFICATION REPORT
# - ROC AUC SCORE
# - DISPLAYS RANDOM FOREST FEATURE IMPORTANCE

import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


# --- CONFIGURATION (YOU MAY NEED TO UPDATE ON YOUR COMPUTER) ---
winner_file = 'C:/Users/naredo.1/Downloads/all_games_team_stats_winners_PreviousSeasons.xlsx'
loser_file = 'C:/Users/naredo.1/Downloads/all_games_team_stats_losers_PreviousSeasons.xlsx'
test_file = 'C:/Users/naredo.1/Downloads/all_games_team_stats_summary.xlsx'

# --- LOAD TRAINING DATA (FBS SEASONS FROM 2018, 2019, AND 2021-2023)
df_winners = pd.read_excel(winner_file)
df_winners['Result'] = 1

df_losers = pd.read_excel(loser_file)
df_losers['Result'] = 0

df_train = pd.concat([df_winners, df_losers], ignore_index=True)

# --- LOAD TESTING DATA (2024 FBS SEASON)
df_test = pd.read_excel(test_file)

# --- MAP THE TESTING RESULT COLUMN (e.g. Winner -> 1 ; Loser -> 0)
if df_test['Result'].dtype == object:
    df_test['Result'] = df_test['Result'].map({'winner': 1, 'loser': 0})

# --- KEY FEATURES 
feature_cols = ['thirdDownEff', 
                'completionAttempts', 
                'yardsPerPass', 
                'rushingYards', 
                'possessionTime']

# --- CONVERT possessionTime (e.g. "30:45") TO TOTAL SECONDS (e.g. "1845")
def time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, str(time_str).split(':'))
        return minutes * 60 + seconds
    except:
        return None

# --- TIME CONVERSION ---
df_train['possessionTime'] = df_train['possessionTime'].apply(time_to_seconds)
df_test['possessionTime'] = df_test['possessionTime'].apply(time_to_seconds)

# --- HANDLE MISSING VALUES ---
for col in feature_cols:
    median_train = df_train[col].median()
    median_test = df_test[col].median()
    df_train[col] = df_train[col].fillna(median_train)
    df_test[col] = df_test[col].fillna(median_test)

# --- SET X AND y ---
X_train = df_train[feature_cols]
y_train = df_train['Result']

X_test = df_test[feature_cols]
y_test = df_test['Result']

# --- Optuna Hyperparameter Tuning ---
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("✅ Best hyperparameters from Optuna:")
print(study.best_params)

# --- Train final model with best params ---
best_model = RandomForestClassifier(**study.best_params)
best_model.fit(X_train, y_train)


# --- PREDICTION ---
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

# --- EVALUATION ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# --- FEATURE IMPORTANCE VISUALIZATION ---
score_df = pd.DataFrame({'feature':X_train.columns,
                            'importance_score': best_model.feature_importances_})

score_df.sort_values('importance_score',ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=score_df['importance_score'], y=score_df['feature'], palette='viridis')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
