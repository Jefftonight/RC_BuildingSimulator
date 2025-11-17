"""
Example of an Annual Simulation
"""
__author__ = "Prageeth Jayathissa"
__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
__credits__ = []
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Prageeth Jayathissa"
__email__ = "jayathissa@arch.ethz.ch"
__status__ = "Production"
from copy import deepcopy
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set root folder one level up, just for this example
mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mainPath)


from building_physics import Zone  # Importing Zone Class
import supply_system
import emission_system
from radiation import Location
from radiation import Window

matplotlib.style.use('ggplot')

# Empty Lists for Storing Data to Plot
ElectricityOut = []
HeatingDemand = []  # Energy required by the zone
HeatingEnergy = []  # Energy required by the supply system to provide HeatingDemand
CoolingDemand = []  # Energy surplus of the zone
CoolingEnergy = []  # Energy required by the supply system to get rid of CoolingDemand
IndoorAir = []
OutsideTemp = []
SolarGains = []
COP = []

import os
print("Current working directory:", os.getcwd())

# Initialise the Location with a weather file
Zurich = Location(epwfile_path=os.path.join(
    mainPath, 'auxiliary', 'Zurich-Kloten_2013.epw'))

# Initialise an instance of the Zone. Empty spaces take on the default
# parameters. See ZonePhysics.py to see the default values
Office = Zone(window_area=4.0,
              walls_area=11.0,
              floor_area=35.0,
              room_vol=105,
              total_internal_area=142.0,
              lighting_load=11.7,
              lighting_control=300.0,
              lighting_utilisation_factor=0.45,
              lighting_maintenance_factor=0.9,
              u_walls=0.2,
              u_windows=1.1,
              ach_vent=1.5,
              ach_infl=0.5,
              ventilation_efficiency=0.6,
              thermal_capacitance_per_floor_area=165000 * 1,
              t_set_heating=20.0,
              t_set_cooling=32.0,
              max_cooling_energy_per_floor_area=-np.inf,
              max_heating_energy_per_floor_area=np.inf,
              heating_supply_system=supply_system.HeatPumpAir,
              cooling_supply_system=supply_system.HeatPumpAir,
              heating_emission_system=emission_system.NewRadiators,
              cooling_emission_system=emission_system.AirConditioning,)

# Define Windows
SouthWindow = Window(azimuth_tilt=0, alititude_tilt=90, glass_solar_transmittance=0.7,
                     glass_light_transmittance=0.8, area=4)

# A catch statement to prevent future coding bugs when modifying window area
if SouthWindow.area != Office.window_area:
    raise ValueError('Window area defined in radiation file doesnt match area defined in zone')


# Define constants for the Zone
gain_per_person = 100  # W per person
appliance_gains = 14  # W per sqm
max_occupancy = 3.0


# Read Occupancy Profile
occupancyProfile = pd.read_csv(os.path.join(
    mainPath, 'auxiliary', 'schedules_el_OFFICE.csv'))

# Starting temperature of the builidng
t_m_prev = 20

# Loop through all 8760 hours of the year
for hour in range(8760):

    # Occupancy for the time step
    occupancy = occupancyProfile.loc[hour, 'People'] * max_occupancy
    # Gains from occupancy and appliances
    internal_gains = occupancy * gain_per_person + \
        appliance_gains * Office.floor_area



    # Instead of using Location class, read your custom CSV directly


    weather_data = pd.read_csv(os.path.join(
        mainPath, 'auxiliary', 'leuven_2023_2025.csv'))

    # Extract outdoor temperature - REPLACE 'temperature' with your column name
    t_out = weather_data['temperature_2m'][hour]  # Change 'temperature' to your actual column name

    # If you have sun position data in CSV, use it directly:
    Altitude, Azimuth = Zurich.calc_sun_position(
        latitude_deg=50.8792, longitude_deg=4.7009, year=2015, hoy=hour)
    # Altitude = weather_data['sun_altitude'][hour]  # Change to your column name
    # Azimuth = weather_data['sun_azimuth'][hour]  # Change to your column name

    # Or if you need to calculate sun position, keep this:
    # from radiation import Location
    # temp_location = Location.__new__(Location)  # Create without EPW file
    # Altitude, Azimuth = temp_location.calc_sun_position(
    #     latitude_deg=50.8798, longitude_deg=4.7005, year=2023, hoy=hour)  # Leuven coordinates

    # Solar radiation - REPLACE column names with yours
    SouthWindow.calc_solar_gains(
        sun_altitude=Altitude,
        sun_azimuth=Azimuth,
        normal_direct_radiation=weather_data['direct_normal_irradiance'][hour]/2,  # Your column name
        horizontal_diffuse_radiation=weather_data['diffuse_radiation'][hour]/2)  # Your column name

    # Illuminance - REPLACE column names with yours
    LUMINOUS_EFFICACY = 110  # [lumens/Watt]

    # Get the irradiance data from your weather data source
    normal_direct_radiation = weather_data['direct_normal_irradiance'][hour]
    horizontal_diffuse_radiation = weather_data['diffuse_radiation'][hour]

    # --- ADD THIS CONVERSION STEP ---
    # Estimate illuminance by multiplying irradiance by the factor
    normal_direct_illuminance_lux = normal_direct_radiation * LUMINOUS_EFFICACY
    horizontal_diffuse_illuminance_lux = horizontal_diffuse_radiation * LUMINOUS_EFFICACY
    # -------------------------------

    # Call the function with your new estimated values
    SouthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                 normal_direct_illuminance=normal_direct_illuminance_lux,
                                 horizontal_diffuse_illuminance=horizontal_diffuse_illuminance_lux)
    # SouthWindow.calc_illuminance(
    #     sun_altitude=Altitude,
    #     sun_azimuth=Azimuth,
    #     normal_direct_illuminance=weather_data['direct_normal_illuminance'][hour],  # Your column name
    #     horizontal_diffuse_illuminance=weather_data['diffuse_illuminance'][hour])  # Your column name







    # # Extract the outdoor temperature in Zurich for that hour
    # t_out = Zurich.weather_data['drybulb_C'][hour]
    #
    # Altitude, Azimuth = Zurich.calc_sun_position(
    #     latitude_deg=47.480, longitude_deg=8.536, year=2015, hoy=hour)
    #
    # SouthWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
    #                              normal_direct_radiation=Zurich.weather_data[
    #                                  'dirnorrad_Whm2'][hour],
    #                              horizontal_diffuse_radiation=Zurich.weather_data['difhorrad_Whm2'][hour])
    #
    # SouthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
    #                              normal_direct_illuminance=Zurich.weather_data[
    #                                  'dirnorillum_lux'][hour],
    #                              horizontal_diffuse_illuminance=Zurich.weather_data['difhorillum_lux'][hour])

    # Office.solve_energy(internal_gains=internal_gains,
    #                     solar_gains=SouthWindow.solar_gains,
    #                     t_out=t_out,
    #                     t_m_prev=t_m_prev)
    # Your control algorithm here - example:
    # Read your control signals from a file or generate them
    # heating when indoorair < 20, cooling when indoorair > 26, in the first step, no indoorair available, assume 20
#     if len(IndoorAir) == 0:
#         current_indoor_temp = t_m_prev
#     else:
#         current_indoor_temp = IndoorAir[-1]
# #   now implete a threhold control
#     heating_action = 1 if current_indoor_temp <= 21 else 0
#     cooling_action = 1 if current_indoor_temp >= 26 else 0




    # replace with model predict values
    # Load control signals (add this before the hour loop, around line 74)
    # control_signals = pd.read_csv(os.path.join(mainPath, 'auxiliary', 'control_signals.csv'))
    # heating_action = control_signals.loc[hour, 'heating_action']
    # cooling_action = control_signals.loc[hour, 'cooling_action']



    # Assuming columns: 'heating_action' and 'cooling_action' with binary values
    # Use the new binary control method
    # Office.solve_energy_with_binary_control(
    #     internal_gains=internal_gains,
    #     solar_gains=SouthWindow.solar_gains,
    #     t_out=t_out,
    #     t_m_prev=t_m_prev,
    #     heating_action=heating_action,
    #     cooling_action=cooling_action,
    #     fixed_heating_power=None,  # Fixed 350W heating when ON (adjust as needed)
    #     fixed_cooling_power=None)  # Fixed 350W cooling when ON (adjust as needed)

    # Office.solve_lighting(
    #     illuminance=SouthWindow.transmitted_illuminance, occupancy=occupancy)

    # Set the previous temperature for the next time step
    # t_m_prev = Office.t_m_next
    # 1. PREDICTIVE STEP: Calculate next hour's temperature with NO HVAC action.
    Office_no_hvac = deepcopy(Office)  # Use a copy to avoid changing the main object's state
    Office_no_hvac.solve_energy_with_binary_control(
        internal_gains=internal_gains,
        solar_gains=SouthWindow.solar_gains,
        t_out=t_out,
        t_m_prev=t_m_prev,
        heating_action=0,  # Assume heating is OFF
        cooling_action=0  # Assume cooling is OFF
    )
    predicted_temp = Office_no_hvac.t_air

    # 2. DECISION STEP: Decide if heating/cooling is needed based on the prediction.
    heating_action = 1 if predicted_temp < Office.t_set_heating else 0
    cooling_action = 1 if predicted_temp > Office.t_set_cooling else 0

    # 3. FINAL CALCULATION: Run the simulation for the hour with the correct action.
    Office.solve_energy_with_binary_control(
        internal_gains=internal_gains,
        solar_gains=SouthWindow.solar_gains,
        t_out=t_out,
        t_m_prev=t_m_prev,
        heating_action=heating_action,
        cooling_action=cooling_action,
        fixed_heating_power=None,  # Use your specific heat pump power here
        fixed_cooling_power=None)

    # ---------------------------------------------------------------------------
    # END: NEW PROACTIVE CONTROL LOGIC
    # ---------------------------------------------------------------------------

    Office.solve_lighting(
        illuminance=SouthWindow.transmitted_illuminance, occupancy=occupancy)

    # Set the previous temperature for the next time step
    t_m_prev = Office.t_m_next

    HeatingDemand.append(Office.heating_demand)
    HeatingEnergy.append(Office.heating_energy)
    CoolingDemand.append(Office.cooling_demand)
    CoolingEnergy.append(Office.cooling_energy)
    ElectricityOut.append(Office.electricity_out)
    IndoorAir.append(Office.t_air)
    OutsideTemp.append(t_out)
    SolarGains.append(SouthWindow.solar_gains)
    COP.append(Office.cop)

annualResults = pd.DataFrame({
    'HeatingDemand': HeatingDemand,
    'HeatingEnergy': HeatingEnergy,
    'CoolingDemand': CoolingDemand,
    'CoolingEnergy': CoolingEnergy,
    'IndoorAir': IndoorAir,
    'OutsideTemp':  OutsideTemp,
    'SolarGains': SolarGains,
    'COP': COP
})


# export to csv file
annualResults.to_csv(os.path.join(mainPath, 'annual_results.csv'), index=False)
# annualResults.to_csv(os.path.join(mainPath, 'annual_results_hotter.csv'), index=False)
# Plotting has been commented out as it can not be conducted in a virtual environment over ssh
annualResults[['HeatingEnergy', 'CoolingEnergy']].plot()
plt.show()
