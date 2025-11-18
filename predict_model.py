#
# predictor_model.py
#
"""
This module defines the CausalLGBMPredictor class.

This class loads a pre-trained "base model" (trained on T=0 data)
and a pre-trained "causal effect model" (e.g., an S-Learner trained
on trimmed data).

It combines their predictions to estimate the next-hour temperature
for a given set of features and a control action.
"""

import joblib
import json
import pandas as pd
import numpy as np


class CausalLGBMPredictor:
    """
    A predictive model that combines a T=0 baseline model with a
    causal effect (CATE) model for T=1.
    """

    def __init__(self,
                 base_model_path='model_base_T0.pkl',
                 causal_model_path='effect_estimation_slearner.pkl',
                 features_path='model_features.json'):
        """
        Loads the trained models and feature list from disk.
        """
        print("Loading CausalLGBMPredictor...")
        try:
            self.model_base = joblib.load(base_model_path)
            print(f"  ...loaded base model: {base_model_path}")

            self.model_causal = joblib.load(causal_model_path)
            print(f"  ...loaded causal model: {causal_model_path}")

            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            print(f"  ...loaded {len(self.feature_columns)} model features from: {features_path}")

        except FileNotFoundError as e:
            print(f"--- ERROR ---")
            print(f"Could not load model file: {e.filename}")
            print("Please run 'Run_all_models_without_scaler_selected.py' first to train and save the models.")
            raise

    def predict_next_temp(self, X_features_dict, heating_action):
        """
        Predicts the next temperature based on a feature dictionary and a heating action.

        Args:
            X_features_dict (dict): A dictionary of all required features.
                e.g., {'OutsideTemp': 10, 'SolarGains': 200, 'IndoorAir_lag1': 21.0, ...}
            heating_action (int): 0 or 1

        Returns:
            float: The predicted next-hour indoor air temperature.
        """

        # 1. Create a DataFrame from the dict, ensuring correct feature order
        # This is critical for the LGBM and EconML models.
        try:
            X_df = pd.DataFrame(X_features_dict, index=[0])[self.feature_columns]
        except KeyError as e:
            print(f"Feature dictionary is missing a required key: {e}")
            print("Required keys are:", self.feature_columns)
            raise

        # 2. Predict the "base" temperature (what happens if T=0)
        base_temp_prediction = self.model_base.predict(X_df)[0]

        # 3. Add the causal effect if heating is ON
        if heating_action == 0:
            return base_temp_prediction
        else:
            # S-Learner's .effect() gives the CATE (T=1 vs T=0)
            causal_effect = self.model_causal.effect(X_df)[0]

            # Final prediction is base + effect
            return base_temp_prediction + causal_effect