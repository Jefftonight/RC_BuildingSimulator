#
# optimization.py
#
import numpy as np
from scipy.optimize import differential_evolution



class CausalTemperatureModel:
    """
    A placeholder for your pre-trained causal (LGBM) model.
    Its job is to predict the *next* air temperature based on the
    current state and a proposed control action.
    """

    def __init__(self):
        # Load your trained LGBM model here
        # self.model = joblib.load('my_lgbm_model.pkl')

        # This is the "effect" trained from your causal analysis.
        # e.g., "running the heat pump for 1 hour adds 1.5°C"
        self.heating_effect_degC = 1.5


    def predict_next_temp(self, t_air_current, t_out, solar_gains, internal_gains, heating_action, cooling_action):
        """
        Predict the next hour's temperature.

        This function should implement your logic:
        "trained on data not heated and plus effect to get when heated"
        """

        # 1. Create the feature vector for your model
        #    (using only non-control features)
        features = np.array([t_air_current, t_out, solar_gains, internal_gains])

        # 2. Predict the "base" temperature (as if no HVAC was used)
        #    This is just an example; use your actual model's prediction
        #    t_base_predicted = self.model.predict(features.reshape(1, -1))[0]

        # --- Placeholder Logic (REPLACE THIS) ---
        # A simple physics-based placeholder for demonstration
        t_base_predicted = t_air_current + (t_out - t_air_current) * 0.1 + (solar_gains + internal_gains) * 0.0001
        # --- End Placeholder Logic ---

        # 3. Add the causal "effect" of the chosen control action
        predicted_temp = t_base_predicted
        if heating_action == 1 and cooling_action == 0:
            predicted_temp += self.heating_effect_degC
        elif cooling_action == 1 and heating_action == 0:
            predicted_temp += self.cooling_effect_degC

        return predicted_temp


# ----------------------------------------------------------------------------
# The Main Optimizer Class
# ----------------------------------------------------------------------------

class Optimizer:


    # Inside optimization.py -> class Optimizer

    def __init__(self, t_set_heating, t_set_cooling, optimization_horizon=2):

        # This now loads your real, trained models
        self.model = CausalLGBMPredictor(
            base_model_path='model_base_T0.pkl',
            causal_model_path='effect_estimation_slearner.pkl',
            features_path='model_features.json'
        )

        self.t_set_heating = t_set_heating
        self.t_set_cooling = t_set_cooling
        self.horizon = optimization_horizon
        self.comfort_weight = 1.0
        self.energy_weight = 1.0

        # Load the feature list to know what features to build
        self.feature_columns = self.model.feature_columns

    def objective_function(self, control_vars, args):
        """
        The cost function that `differential_evolution` will try to minimize.

        - `control_vars`: A 1D array of continuous values [h1, c1, h2, c2, ...]
        - `args`: A tuple containing other necessary data:
            (current_t_air, weather_forecast, price_forecast, gains_forecast)
        """

        # Unpack arguments
        current_t_air, weather_forecast, price_forecast, gains_forecast = args

        # --- 1. Convert continuous control variables to binary actions ---
        # `differential_evolution` gives continuous values (0.0 to 1.0).
        # We round them to get binary 0 or 1.
        binary_actions = np.round(control_vars).astype(int)

        total_cost = 0.0
        predicted_t_air = current_t_air

        # --- 2. Simulate the optimization horizon ---
        for i in range(self.horizon):
            # Get actions for this timestep
            # Note: 2*i for heating, 2*i + 1 for cooling
            heating_action = binary_actions[2 * i]
            cooling_action = binary_actions[2 * i + 1]

            # --- 3. Enforce "no simultaneous heating/cooling" constraint ---
            if heating_action == 1 and cooling_action == 1:
                # Penalize this invalid state heavily
                total_cost += 1000  # Large penalty
                # Force a valid state (e.g., turn off cooling)
                cooling_action = 0

            # --- 4. Get forecasts for this timestep ---
            t_out = weather_forecast['temperature_2m'][i]
            solar_gains = weather_forecast['solar_gains'][i]
            imbalance_price = price_forecast['imbalance_price'][i]
            internal_gains = gains_forecast['internal_gains'][i]

            # --- 5. Predict next temperature using the causal model ---
            predicted_t_air = self.model.predict_next_temp(
                t_air_current=predicted_t_air,
                t_out=t_out,
                solar_gains=solar_gains,
                internal_gains=internal_gains,
                heating_action=heating_action,
                cooling_action=cooling_action
            )

            # --- 6. Calculate cost for this timestep ---

            # Comfort Cost (quadratic penalty for deviation from setpoints)
            comfort_cost = 0.0
            if predicted_t_air < self.t_set_heating:
                comfort_cost = (self.t_set_heating - predicted_t_air) ** 2
            elif predicted_t_air > self.t_set_cooling:
                comfort_cost = (predicted_t_air - self.t_set_cooling) ** 2

            # Energy Cost (Power [kW] * Price [€/kWh] * 1h)
            # You must define your fixed power for heating/cooling
            # This should match the `fixed_heating_power` in `solve_energy_with_binary_control`
            fixed_heating_power_watts = 20 * 35.0  # default 20 W/m2 * 35 m2
            fixed_cooling_power_watts = 20 * 35.0

            energy_cost = 0.0
            if heating_action == 1:
                energy_cost = (fixed_heating_power_watts / 1000.0) * imbalance_price
            elif cooling_action == 1:
                energy_cost = (fixed_cooling_power_watts / 1000.0) * imbalance_price

            # Total cost for this step
            total_cost += (self.comfort_weight * comfort_cost) + (self.energy_weight * energy_cost)

        return total_cost

    def find_optimal_control(self, current_t_air, weather_forecast, price_forecast, gains_forecast):
        """
        Runs the Differential Evolution optimizer for the specified horizon.
        """

        # Number of variables: 2 (heating, cooling) per hour
        num_vars = self.horizon * 2

        # Bounds for each variable: (0.0, 1.0)
        bounds = [(0, 1)] * num_vars

        # Pack the args for the objective function
        args = (current_t_air, weather_forecast, price_forecast, gains_forecast)

        # Run the optimizer
        result = differential_evolution(
            self.objective_function,
            bounds,
            args=args,
            popsize=10,  # Small population for speed
            maxiter=50,  # Few iterations for speed
            tol=0.1,
            recombination=0.7,
            mutation=(0.5, 1.0),
            workers=1  # Use 1 for simple problems, -1 for parallel
        )

        # --- Post-process the result ---
        # Get the best continuous variables
        optimal_continuous_vars = result.x

        # Convert to the final binary actions
        optimal_binary_actions = np.round(optimal_continuous_vars).astype(int)

        # Enforce "no simultaneous" one last time to be safe
        for i in range(self.horizon):
            if optimal_binary_actions[2 * i] == 1 and optimal_binary_actions[2 * i + 1] == 1:
                optimal_binary_actions[2 * i + 1] = 0  # Prioritize heating, for example

        # Reshape to a more useful format: [[h1, c1], [h2, c2], ...]
        optimal_sequence = optimal_binary_actions.reshape(-1, 2)

        return optimal_sequence