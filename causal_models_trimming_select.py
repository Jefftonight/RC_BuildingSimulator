# enhanced_causal_analysis.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from econml.dml import LinearDML
# import nonparametric dml from econml
from econml.dml import NonParamDML
from econml.metalearners import SLearner, TLearner, XLearner
import optuna
import warnings
import numpy as np
warnings.filterwarnings('ignore')
RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
def prepare_causal_data(df, treatment_col, outcome_col, lag_features=12):
    """
    Prepares data, creates treatment/lagged features, and handles NaNs.
    """
    data = df.copy()
    if 'COP' in data.columns:
        data['COP'].fillna(0, inplace=True)
    data['HeatingAction'] = (data['HeatingDemand'] > 0).astype(int)
    features_to_lag = ['IndoorAir', 'OutsideTemp', 'SolarGains', 'HeatingAction']
    for col in features_to_lag:
        for i in range(1, lag_features + 1):
            data[f'{col}_lag{i}'] = data[col].shift(i)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    Y = data[outcome_col]
    T = data[treatment_col]
    cols_to_exclude = [
        'HeatingDemand', 'HeatingEnergy', 'CoolingDemand', 'CoolingEnergy',
        'IndoorAir', 'HeatingAction', 'COP'
    ]
    if 'CoolingAction' in data.columns:
        cols_to_exclude.append('CoolingAction')
    X = data.drop(columns=cols_to_exclude)
    return X, T, Y


# # ============================================================================
# # 2. PROPENSITY SCORE TRIMMING (UPDATED)
# # ============================================================================
# def trim_by_propensity_score(X, T, Y, random_state, lower_bound=0.1, upper_bound=0.9):
#     """
#     Trims the dataset using a propensity model trained on a specific subset of features.
#     """
#     print(f"Trimming observations with propensity scores outside [{lower_bound}, {upper_bound}]...")
#
#     # --- UPDATED: Select a specific, smaller set of features for the propensity model ---
#     propensity_features = [
#         'IndoorAir_lag1',
#         'SolarGains',
#         'OutsideTemp',
#     ]
#     # Ensure all selected features are present in the dataframe
#     propensity_features = [f for f in propensity_features if f in X.columns]
#     print(f"Using these features for propensity model: {propensity_features}")
#
#     X_propensity = X[propensity_features]
#     # -----------------------------------------------------------------------------------
#
#     propensity_model = LogisticRegression(C=10, penalty='l1', solver='liblinear', random_state=random_state)
#     # if indoor air lag1 is larger than 26, then do not use it to train the model and set it to 0
#     if 'IndoorAir_lag1' in X_propensity.columns:
#         X_propensity = X_propensity[X_propensity['IndoorAir_lag1'] <= 26]
#         T = T[X_propensity.index]
#         Y = Y[X_propensity.index]
#         X_propensity = X_propensity.copy()
#
#     propensity_model.fit(X_propensity, T)
#
#     # Predict probabilities using the same subset of features
#     propensity_scores = propensity_model.predict_proba(X_propensity)[:, 1]
#     # if indoor_air lag1 is larger than 26, then propensity score is 0
#
#
#     keep_mask = (propensity_scores > lower_bound) & (propensity_scores < upper_bound)
#     n_original = len(X)
#     n_kept = keep_mask.sum()
#     print(f"Kept {n_kept} of {n_original} observations ({n_kept / n_original:.2%}).")
#
#     # Filter the original datasets to keep all features for the main causal models
#     return X[keep_mask], T[keep_mask], Y[keep_mask]


# ============================================================================
# 2. PROPENSITY SCORE TRIMMING (UPDATED)
# ============================================================================
def trim_by_propensity_score(X, T, Y, random_state, lower_bound=0.1, upper_bound=0.9):
    """
    Applies a domain rule and then trims based on propensity scores.
    """
    print(f"Trimming observations with propensity scores outside [{lower_bound}, {upper_bound}]...")

    # --- YOUR NEW LOGIC, IMPLEMENTED SAFELY ---
    # 1. Create a mask to identify rows to EXCLUDE from propensity model TRAINING.
    if 'IndoorAir_lag1' in X.columns:
        print("Pre-filtering data for propensity model training where IndoorAir_lag1 <= 1000000")
        train_mask = X['IndoorAir_lag1'] <= 1000000
    else:
        train_mask = pd.Series(True, index=X.index)  # Default to using all data if column not found

    # 2. Select features for the propensity model
    propensity_features = ['IndoorAir_lag1', 'OutsideTemp', 'SolarGains', 'IndoorAir_lag2', 'IndoorAir_lag3', 'SolarGains_lag1', 'SolarGains_lag2', 'OutsideTemp_lag1', 'OutsideTemp_lag2']
    # propensity_features = ['IndoorAir_lag1']
    propensity_features = [f for f in propensity_features if f in X.columns]
    X_propensity = X[propensity_features]

    # 3. Train the propensity model ONLY on the pre-filtered data
    # propensity_model = LogisticRegression(C=1, random_state=random_state, solver='lbfgs', penalty='none')
    # give more weight to IndoorAir_lag1 feature, to 2 other set as 1

    propensity_model = LogisticRegression(C=1, random_state=random_state, solver='sag', penalty='none', class_weight={0:5, 1:1})
    # propensity_model = LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=random_state, class_weight={0:5, 1:1})
    propensity_model.fit(X_propensity[train_mask], T[train_mask])

    # 4. Predict propensity scores for ALL data points
    propensity_scores = propensity_model.predict_proba(X_propensity)[:, 1]
    # -----------------------------------------------

    # Create the final trimming mask based on propensity score bounds
    ps_mask = (propensity_scores > lower_bound) & (propensity_scores < upper_bound)

    # The final mask must also respect the initial domain rule
    final_keep_mask = ps_mask & train_mask

    n_original = len(X)
    n_kept = final_keep_mask.sum()
    print(f"Kept {n_kept} of {n_original} observations ({n_kept / n_original:.2%}).")
    # check if the n_kept is too low less than 5%, if so lower bound set to 0.1
    if n_kept / n_original < 0.025:
        print("Too few observations kept after trimming. Lowering lower_bound to 0.025 and reapplying trimming.")
        lower_bound = 0.1
        final_keep_mask = (propensity_scores > lower_bound) & (propensity_scores < upper_bound) & train_mask
        n_kept = final_keep_mask.sum()
        print(f"Now kept {n_kept} of {n_original} observations ({n_kept / n_original:.2%}).")

    return X[final_keep_mask], T[final_keep_mask], Y[final_keep_mask]

# ============================================================================
# 3. OPTUNA HYPERPARAMETER TUNING
# ============================================================================
def tune_lgbm_model(X_train, y_train, X_val, y_val, n_trials=50, random_state=123):
    """Tunes LGBM hyperparameters using Optuna."""

    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'random_state': random_state,
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # l1 regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),    # l2 regularization
            # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),  # Key for preventing overfitting
            # 'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),  # Row sampling
 # Feature sampling

        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds, squared=False)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# ============================================================================
# 4. CAUSAL MODEL TRAINING (using EconML)
# ============================================================================
class CausalModels:
    """A container for the four EconML causal inference models."""

    def __init__(self, lgbm_params=None, random_state=123):
        lgbm_params_with_seed = (lgbm_params or {}).copy()
        lgbm_params_with_seed['random_state'] = random_state

        base_model = lgb.LGBMRegressor(**lgbm_params_with_seed)
        self.s_learner = SLearner(overall_model=base_model)
        self.t_learner = TLearner(models=base_model)
        self.x_learner = XLearner(models=base_model)
        # self.double_ml = LinearDML(model_y=base_model, model_t=base_model, random_state=random_state)
        self.double_ml = NonParamDML(model_y=base_model, model_t=base_model, model_final=base_model)

    def train_all(self, X, T, Y):
        print("Training S-Learner...")
        self.s_learner.fit(Y, T, X=X)
        print("Training T-Learner...")
        self.t_learner.fit(Y, T, X=X)
        print("Training X-Learner...")
        self.x_learner.fit(Y, T, X=X)
        print("Training Double ML...")
        self.double_ml.fit(Y, T, X=X)


# ============================================================================
# 5. CAUSAL EFFECT ESTIMATION & RECORDING
# ============================================================================
def estimate_and_record_effects(causal_models, X_test, T_test, Y_test):
    """Estimates CATE, calculates ATE, and saves detailed results."""
    print("\n--- Estimating Causal Effects ---")
    results_df = X_test.copy()
    results_df['Treatment'] = T_test
    results_df['Outcome'] = Y_test
    # results_df['PropensityScore'] = propensity_scores

    cate_s = causal_models.s_learner.effect(X_test)
    results_df['CATE_S_Learner'] = cate_s
    print(f"S-Learner ATE: {cate_s.mean():.4f}°C")

    cate_t = causal_models.t_learner.effect(X_test)
    results_df['CATE_T_Learner'] = cate_t
    print(f"T-Learner ATE: {cate_t.mean():.4f}°C")

    cate_x = causal_models.x_learner.effect(X_test)
    results_df['CATE_X_Learner'] = cate_x
    print(f"X-Learner ATE: {cate_x.mean():.4f}°C")

    cate_dml = causal_models.double_ml.effect(X_test)
    results_df['CATE_DoubleML'] = cate_dml
    print(f"Double ML ATE: {cate_dml.mean():.4f}°C")

    output_filename = 'test_results_with_cate.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\nDetailed test results saved to '{output_filename}'")

    return results_df