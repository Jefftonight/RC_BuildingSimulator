#
# control.py
#
import os
import pandas as pd
import numpy as np

from copy import deepcopy
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# Import your project's modules
from building_physics import Zone
from Optimization_module import Optimizer, CausalTemperatureModel
from radiation import Location, Window

mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mainPath)


from building_physics import Zone  # Importing Zone Class
import supply_system
import emission_system
from radiation import Location
from radiation import Window


class MPCController:

    def __init__(self, main_path, optimization_horizon=2):
        self.main_path = main_path
        self.optimization_horizon = optimization_horizon

        # --- 1. Load all simulation data ---
        self.weather_data = self.load_weather_data()
        self.occupancy_profile = self.load_occupancy()
        self.price_profile = self.load_prices()  # Imbalance prices

        # --- 2. Initialize Building Physics (The "Real" Model) ---
        self.office_zone = Zone(
            window_area=4.0,
            floor_area=35.0,
            t_set_heating=20.0,
            t_set_cooling=26.0,

        )

        # --- 3. Initialize Windows (for solar/lighting) ---
        self.south_window = Window(
            azimuth_tilt=0, alititude_tilt=90, glass_solar_transmittance=0.7,
            glass_light_transmittance=0.8, area=4.0
        )
        self.location = Location(epwfile_path=os.path.join(
            self.main_path, 'rc_simulator', 'auxiliary', 'Zurich-Kloten_2013.epw'))

        # --- 4. Initialize Optimizer ---
        # Initialize your causal model
        causal_model = CausalTemperatureModel()

        # Pass the model and setpoints to the optimizer
        self.optimizer = Optimizer(
            causal_model=causal_model,
            t_set_heating=self.office_zone.t_set_heating,
            t_set_cooling=self.office_zone.t_set_cooling,
            optimization_horizon=self.optimization_horizon
        )

    def load_weather_data(self):
        # Using the Zurich EPW for this example, as in annualSimulation
        weather_df = self.location.weather_data
        # Pre-calculate solar gains for the whole year
        solar_gains_list = []
        for hour in range(8760):
            Altitude, Azimuth = self.location.calc_sun_position(
                latitude_deg=47.480, longitude_deg=8.536, year=2015, hoy=hour)
            self.south_window.calc_solar_gains(
                sun_altitude=Altitude, sun_azimuth=Azimuth,
                normal_direct_radiation=weather_df['dirnorrad_Whm2'][hour],
                horizontal_diffuse_radiation=weather_df['difhorrad_Whm2'][hour])
            solar_gains_list.append(self.south_window.solar_gains)

        weather_df['solar_gains'] = solar_gains_list
        weather_df['temperature_2m'] = weather_df['drybulb_C']
        print("Weather data loaded.")
        return weather_df

    def load_occupancy(self):
        profile = pd.read_csv(os.path.join(
            self.main_path, 'rc_simulator', 'auxiliary', 'schedules_el_OFFICE.csv'))

        # Pre-calculate internal gains
        gain_per_person = 100  # W per person
        appliance_gains = 14  # W per sqm
        max_occupancy = 3.0

        internal_gains_list = []
        for hour in range(8760):
            occupancy = profile.loc[hour, 'People'] * max_occupancy
            gains = occupancy * gain_per_person + appliance_gains * self.office_zone.floor_area
            internal_gains_list.append(gains)

        profile['internal_gains'] = internal_gains_list
        print("Occupancy data loaded.")
        return profile

    def load_prices(self):
        # TODO: Load your actual imbalance price forecast
        # For now, creating a placeholder
        print("Using placeholder imbalance prices.")
        price_data = {'imbalance_price': np.random.uniform(20, 200, 8760)}  # [€/MWh] or [€/kWh]? Assuming €/MWh
        # Convert to €/kWh, as used in optimizer
        price_df = pd.DataFrame(price_data)
        price_df['imbalance_price'] = price_df['imbalance_price'] / 1000.0
        return price_df

    def run_mpc_simulation(self):
        """
        The main simulation loop (replaces annualSimulation.py loop).
        """

        # --- 1. Initialize simulation ---
        t_m_prev = 20.0  # Starting thermal mass temp

        results_data = {
            'real_t_air': [],
            't_out': [],
            'heating_action_applied': [],
            'cooling_action_applied': [],
            'heating_energy': [],
            'cooling_energy': [],
            'total_cost': []  # TODO: calculate and store the "real" cost
        }

        print("Starting 8760-hour MPC simulation...")

        # --- 2. Run the Receding Horizon Loop ---
        for hour in range(8760 - self.optimization_horizon):

            if hour % 100 == 0:
                print(f"Simulating hour {hour}...")

            # --- A. Get Current State & Forecasts ---

            # Get real current air temp from the RC model
            # Note: At hour 0, t_air is not set, so we use t_m_prev as a guess
            current_t_air = self.office_zone.t_air if hasattr(self.office_zone, 't_air') else t_m_prev

            # Get forecasts for the *optimization horizon* (next 2 hours)
            # These are "perfect" forecasts for this simulation
            end_hour = hour + self.optimization_horizon
            weather_forecast = self.weather_data.iloc[hour:end_hour].reset_index(drop=True)
            price_forecast = self.price_profile.iloc[hour:end_hour].reset_index(drop=True)
            gains_forecast = self.occupancy_profile.iloc[hour:end_row].reset_index(drop=True)

            # --- B. Call Optimizer ---
            # Find the best 2-hour plan
            optimal_sequence = self.optimizer.find_optimal_control(
                current_t_air=current_t_air,
                weather_forecast=weather_forecast,
                price_forecast=price_forecast,
                gains_forecast=gains_forecast
            )

            # --- C. Apply First Action ---
            # Get *only* the first action from the optimal sequence
            action_to_apply = optimal_sequence[0]
            heating_action = action_to_apply[0]
            cooling_action = action_to_apply[1]

            # --- D. "Loop Back" to Real Model ---
            # Get the real inputs for this *single* hour
            t_out_real = self.weather_data.loc[hour, 'temperature_2m']
            solar_gains_real = self.weather_data.loc[hour, 'solar_gains']
            internal_gains_real = self.occupancy_profile.loc[hour, 'internal_gains']

            # Run the *real* RC model for *one hour* with this action
            self.office_zone.solve_energy_with_binary_control(
                internal_gains=internal_gains_real,
                solar_gains=solar_gains_real,
                t_out=t_out_real,
                t_m_prev=t_m_prev,
                heating_action=heating_action,
                cooling_action=cooling_action,
                fixed_heating_power=None,  # Use defaults in building_physics
                fixed_cooling_power=None
            )

            # --- E. Store Results & Update State ---

            # Store the "real" results
            results_data['real_t_air'].append(self.office_zone.t_air)
            results_data['t_out'].append(t_out_real)
            results_data['heating_action_applied'].append(heating_action)
            results_data['cooling_action_applied'].append(cooling_action)
            results_data['heating_energy'].append(self.office_zone.heating_energy)
            results_data['cooling_energy'].append(self.office_zone.cooling_energy)

            # Get the "real" next thermal mass temp
            t_m_prev = self.office_zone.t_m_next

        print("Simulation complete.")

        # --- 3. Save Results ---
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(self.main_path, 'mpc_results.csv'), index=False)
        print("Results saved to mpc_results.csv")

        # Plot
        try:
            import matplotlib.pyplot as plt
            results_df[['real_t_air', 't_out']].plot(figsize=(15, 5))
            plt.title('MPC Simulation: Real Indoor vs. Outdoor Temp')
            plt.ylabel('Temperature (°C)')
            plt.xlabel('Hour of Year')
            plt.show()
        except ImportError:
            print("Install matplotlib to see a plot of the results.")


# --- Main execution ---
k = 0
if k == 0:
    # Set root folder one level up
    main_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Run the MPC simulation
    controller = MPCController(main_path=main_repo_path, optimization_horizon=2)
    controller.run_mpc_simulation()