#
# Run_all_models_without_scaler_selected.py
#
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from causal_models_trimming_select import (
    prepare_causal_data,
    trim_by_propensity_score,
    tune_lgbm_model,
    CausalModels,
    estimate_and_record_effects
)
import matplotlib.pyplot as plt
import numpy as np
import joblib  # Added for saving models
import json  # Added for saving feature list

# ============================================================================
#  CONFIGURATION
# ============================================================================
RANDOM_SEED = 123
USE_PROPENSITY_TRIMMING = True
np.random.seed(RANDOM_SEED)

k = 0
# ============================================================================
if k != 1:

    """Main function to run the enhanced causal analysis pipeline."""

    # 1. LOAD & PREPARE DATA
    print("=" * 30 + " 1. Loading and Preparing Data " + "=" * 30)

    df = pd.read_csv('annual_results.csv')
    print("Successfully loaded 'annual_results.csv'.")
    df = df[df['CoolingDemand'] == 0]

    TREATMENT = 'HeatingAction'
    OUTCOME = 'IndoorAir'
    X, T, Y = prepare_causal_data(df, treatment_col=TREATMENT, outcome_col=OUTCOME)

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False)

    print(f"\nTotal data: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")

    # ============================================================================
    # 2. TRAIN AND SAVE THE BASE (T=0) MODEL
    # ============================================================================
    print("\n" + "=" * 30 + " 2. Training Base (T=0) Model " + "=" * 30)

    # Filter for "Heating Off" (T=0) data only
    X_train_T0 = X_train[T_train == 0]
    Y_train_T0 = Y_train[T_train == 0]
    X_test_T0 = X_test[T_test == 0]
    Y_test_T0 = Y_test[T_test == 0]

    print(f"Base model training on {len(X_train_T0)} (T=0) samples.")
    print(f"Base model validating on {len(X_test_T0)} (T=0) samples.")

    # Tune hyperparameters for the base model
    base_model_params = tune_lgbm_model(
        X_train_T0, Y_train_T0, X_test_T0, Y_test_T0, n_trials=50, random_state=RANDOM_SEED
    )
    print(f"Best Base (T=0) Model Params: {base_model_params}")

    # Train the final base model
    model_base_T0 = lgb.LGBMRegressor(**base_model_params, random_state=RANDOM_SEED)
    model_base_T0.fit(X_train_T0, Y_train_T0)

    # Save the base model
    joblib.dump(model_base_T0, 'model_base_T0.pkl')
    print("Base (T=0) model saved to 'model_base_T0.pkl'")

    # ============================================================================
    # 3. SAVE THE FEATURE LIST
    # ============================================================================
    features_list = list(X_train.columns)
    with open('model_features.json', 'w') as f:
        json.dump(features_list, f, indent=2)
    print(f"Model feature list saved to 'model_features.json'")

    # ============================================================================
    # 4. APPLY PROPENSITY SCORE TRIMMING (to train causal model)
    # ============================================================================
    if USE_PROPENSITY_TRIMMING:
        print("\n" + "=" * 30 + " 4. Applying Propensity Score Trimming " + "=" * 30)
        X_train_trim, T_train_trim, Y_train_trim = trim_by_propensity_score(
            X_train, T_train, Y_train, random_state=RANDOM_SEED, lower_bound=0.15, upper_bound=0.95
        )
    else:
        print("\n" + "=" * 30 + " 4. Skipping Propensity Score Trimming " + "=" * 30)
        X_train_trim, T_train_trim, Y_train_trim = X_train, T_train, Y_train

    # ============================================================================
    # 5. TUNE & TRAIN CAUSAL MODELS (on trimmed data)
    # ============================================================================
    print("\n" + "=" * 30 + " 5. Tuning LGBM for Causal Models " + "=" * 30)
    # Note: We tune using the *trimmed* training set but the *original* test set
    best_causal_lgbm_params = tune_lgbm_model(
        X_train_trim, Y_train_trim, X_test, Y_test, n_trials=50, random_state=RANDOM_SEED
    )
    print(f"Best LGBM Params for Causal Models: {best_causal_lgbm_params}")

    print("\n" + "=" * 30 + " 6. Training Causal Models " + "=" * 30)
    causal_models = CausalModels(lgbm_params=best_causal_lgbm_params, random_state=RANDOM_SEED)
    # Train on the *trimmed* data
    causal_models.train_all(X_train_trim, T_train_trim, Y_train_trim.values)

    # ============================================================================
    # 7. SAVE THE CAUSAL MODELS
    # ============================================================================
    joblib.dump(causal_models.s_learner, 'effect_estimation_slearner.pkl')
    joblib.dump(causal_models.t_learner, 'effect_estimation_tlearner.pkl')
    joblib.dump(causal_models.x_learner, 'effect_estimation_xlearner.pkl')
    joblib.dump(causal_models.double_ml, 'effect_estimation_doubleml.pkl')
    print("All causal models saved (e.g., 'effect_estimation_slearner.pkl')")

    # ============================================================================
    # 8. ESTIMATE & RECORD CAUSAL EFFECTS (on untrimmed test set)
    # ============================================================================
    print("\n" + "=" * 30 + " 8. Estimating and Recording Effects " + "=" * 30)
    results_with_cate = estimate_and_record_effects(
        causal_models, X_test, T_test, Y_test
    )

    print("\n--- First 5 rows of test results with CATE ---")
    print(results_with_cate.head())

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # ... (Plotting code remains the same) ...
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [results_with_cate['CATE_S_Learner'], results_with_cate['CATE_T_Learner'],
         results_with_cate['CATE_X_Learner'], results_with_cate['CATE_DoubleML']],
        labels=['S-Learner', 'T-Learner', 'X-Learner', 'Double ML']
    )
    ax.set_title('CATE Distributions by Model (Trained on Trimmed Data)')
    ax.set_ylabel('CATE (°C)')
    plt.grid(True)
    plt.show()

print("Balabalabalalaaaaaaa enhanced causal analysis...")