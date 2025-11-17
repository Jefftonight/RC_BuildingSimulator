# run_enhanced_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from causal_models_trimming_select import (
    prepare_causal_data,
    trim_by_propensity_score,
    tune_lgbm_model,
    CausalModels,
    estimate_and_record_effects
)
import matplotlib.pyplot as plt
import numpy as np
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
    # df = pd.read_csv('annual_results_hotter.csv')
    print("Successfully loaded 'annual_results.csv'.")
    # fillter out where cooling demand is not zero because we focus on heating only
    df = df[df['CoolingDemand'] == 0]


    TREATMENT = 'HeatingAction'
    OUTCOME = 'IndoorAir'
    X, T, Y = prepare_causal_data(df, treatment_col=TREATMENT, outcome_col=OUTCOME)

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False)

    if USE_PROPENSITY_TRIMMING:
        print("\n" + "=" * 30 + " 2. Applying Propensity Score Trimming " + "=" * 30)
        X_train, T_train, Y_train = trim_by_propensity_score(
            X_train, T_train, Y_train, random_state=RANDOM_SEED, lower_bound=0.15, upper_bound=0.95
        )
    # if USE_PROPENSITY_TRIMMING:
    #     print("\n" + "=" * 30 + " 2. Applying Propensity Score Trimming " + "=" * 30)
    #     X_train, T_train, Y_train = trim_by_propensity_score(
    #         X_train, T_train, Y_train, random_state=RANDOM_SEED, lower_bound=0.1, upper_bound=0.95
    #     )


    # 3. TUNE HYPERPARAMETERS
    print("\n" + "=" * 30 + " 3. Tuning LGBM Base Model " + "=" * 30)
    best_lgbm_params = tune_lgbm_model(
        X_train, Y_train, X_test, Y_test, n_trials=50, random_state=RANDOM_SEED
    )
    print(f"Best LGBM Params found: {best_lgbm_params}")

    # 4. TRAIN CAUSAL MODELS
    print("\n" + "=" * 30 + " 4. Training Causal Models " + "=" * 30)
    causal_models = CausalModels(lgbm_params=best_lgbm_params, random_state=RANDOM_SEED)
    causal_models.train_all(X_train, T_train, Y_train.values)

    # 5. ESTIMATE & RECORD CAUSAL EFFECTS
    print("\n" + "=" * 30 + " 5. Estimating and Recording Effects " + "=" * 30)
    results_with_cate = estimate_and_record_effects(
        causal_models, X_test, T_test, Y_test
    )

    print("\n--- First 5 rows of test results with CATE ---")
    print(results_with_cate.head())

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [results_with_cate['CATE_S_Learner'], results_with_cate['CATE_T_Learner'],
         results_with_cate['CATE_X_Learner'], results_with_cate['CATE_DoubleML']],
        labels=['S-Learner', 'T-Learner', 'X-Learner', 'Double ML']
    )
    # ax.boxplot(
    #     [results_with_cate['CATE_S_Learner'], results_with_cate['CATE_T_Learner'],
    #      results_with_cate['CATE_X_Learner']],
    #     labels=['S-Learner', 'T-Learner', 'X-Learner']
    # )
    ax.set_title('CATE Distributions by Model (Without Scaler)')
    ax.set_ylabel('CATE (°C)')
    plt.grid(True)
    plt.show()

print("Balabalabalalaaaaaaa enhanced causal analysis...")