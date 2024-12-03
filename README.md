# Data-606-Capstone-Project
## Intelligent Grid Machine Learning Driven Optimization For Renewable Energy Stability

### Project Overview

The goal of this project is to employ machine learning to solve problems related to the grid's integration of renewable energy sources, such as wind and solar. Due to their intrinsic variability, these energy sources may cause energy restriction, inefficiencies, and grid instabilities. 
- This research intends to create predictive models to improve grid stability and maximize resource use by examining past energy production, weather, and curtailment records.
- Predicting situations that could cause grid disruptions is known as grid stability prediction.
- Evaluation of Renewable Energy Risk: use weather trends to identify and forecast times of low output of renewable energy.
  
Curtailment minimization is the process of creating plans to lessen energy curtailment in situations with high output and low demand. By facilitating the effective integration of renewable energy, this initiative enhances resource allocation, supports a sustainable future, and more reliable power grid.

### Data Collection and Cleaning

  #### **Data Sources**
  1. **Curtailment Data**:
     - Includes records of wind and solar energy curtailments.
     - Data Period: 2020–2024.
  2. **Production Data**:
     - Captures energy generation by type (solar, wind, nuclear, etc.).
     - Data Period: 2020–2024.
  3. **Weather Data**:
     - Historical weather metrics such as temperature, humidity, and wind speed.
     - Source: Visual Crossing.

  #### Description of Features
  
  **The Production and Curtailment contains the following features:**     
  
      1. datetime: The combindex for hourly load and weather data, representing the date and time for each observation.
      
      2. load: The total electrical load measured in the system, indicating the demand for electricity during the specified time period.
      
      3. solar: The amount of solar energy generated, measured in megawatts (MW) or a similar unit, reflecting the contribution of solar power to the overall energy mix.
      
      4. wind: The amount of wind energy generated, indicating how much energy was produced from wind sources, also measured in MW or a similar unit.
      
      5. net load: The load remaining after accounting for renewable energy generation, calculated as `load - (solar + wind)`. This figure shows the actual demand that must be met by other generation sources.
      
      6. renewables: The total contribution from renewable energy sources, which includes energy from solar, wind, and potentially other renewable sources like geothermal and biomass.
      
      7. nuclear: The contribution from nuclear power generation, measured in MW, indicating the amount of electricity supplied by nuclear plants during the observed period.
      
      8. large hydro: The amount of electricity generated from large hydroelectric sources, reflecting the contribution of large-scale hydropower plants to the energy supply.
      
      9. imports: The amount of electricity imported from neighboring regions or countries, providing insight into how much power is being brought in to meet demand.
      
      10. generation: The total electricity generation from all sources, including fossil fuels, renewables, and nuclear, indicating the overall production capacity during the observed time.
      
      11. thermal: The electricity generation from thermal sources (e.g., coal, gas), showing the contribution from traditional fossil fuel plants to the overall energy generation.
      
      12. load less (generation + imports): The difference between total load and the sum of generation and imports, calculated as `load - (generation + imports)`. This value can indicate whether there is a deficit or              surplus of electricity.
      
      13. wind curtailment: The amount of wind energy that was curtailed, measured in MW or as a total quantity, indicating instances when wind generation was limited due to grid constraints or low demand.
      
      14. solar curtailment: The amount of solar energy that was curtailed, providing insight into how much potential solar generation was not utilized.
      
      15. temp: The temperature recorded in the weather data, measured in degrees Celsius or Fahrenheit, affecting both energy demand and generation.
      
      16. dew: The dew point temperature recorded, indicating the temperature at which air becomes saturated with moisture. This can influence humidity levels and energy consumption patterns.
      
      17. humidity: The humidity level recorded, typically expressed as a percentage, which can affect energy demand, especially for heating and cooling.    
      
      18. windspeed: The wind speed recorded, typically measured in meters per second (m/s) or miles per hour (mph), which can influence both renewable energy generation and energy demand.  
      
      19. visibility: The visibility level recorded, which can impact various operational aspects of energy generation and distribution, particularly in adverse weather conditions.

  **The Weather Data Contains the following Features:**
  
      1. name: Location name (e.g., California).
      
      2. datetime: Timestamp indicating the date and time of the observation.
      
      3. temp: Temperature in degrees Celsius or Fahrenheit.
      
      4. feelslike:	Perceived temperature considering wind chill and humidity.
      
      5. dew: Dew point temperature, indicating the temperature at which air becomes saturated.
      
      6. humidity: Relative humidity as a percentage, reflecting moisture in the air.
      
      7. precip: Precipitation amount (e.g., rainfall) in millimeters or inches.
      
      8. precipprob: Probability of precipitation as a percentage.
      
      9. preciptype: Type of precipitation (e.g., rain, snow).
      
      10. snow:	Amount of snowfall in millimeters or inches.
      
      11. snowdepth: Depth of snow accumulation in millimeters or inches.
      
      12. windgust:	Maximum wind gust speed recorded in the observation period (e.g., in m/s or mph).
      
      13. windspeed: Average wind speed during the observation period (e.g., in m/s or mph).
      
      14. winddir: Wind direction in degrees, where 0° represents north.
      
      15. sealevelpressure:	Atmospheric pressure at sea level, measured in hPa or mbar.
      
      16. cloudcover:	Percentage of sky covered by clouds.
      
      17. visibility:	Visibility distance, typically in kilometers or miles.
      
      18. solarradiation:	Solar radiation intensity, measured in W/m².
      
      19. solarenergy: Total solar energy received, typically in kWh/m².
      
      20. uvindex: UV index indicating the level of ultraviolet radiation.
      
      21. severerisk:	Risk of severe weather conditions (e.g., storms), on a scale or percentage.
      
      22. conditions:	General description of weather conditions (e.g., Clear, Cloudy).
      
      23. icon:	Weather condition icon representation (e.g., clear-night).
      
      24. stations:	Weather stations contributing to the observation data.
        
  
  #### **Data Cleaning Process**
  
      In the final dataset, we retained only the most relevant features for the research and dropped others, including windgust, as they were not required. The data cleaning process was streamlined as follows:
      
        1. Feature Selection:**
        - Retained key features:
            - Production Data: Load, Net Load, Solar, Wind, Renewables, Thermal, Large Hydro, Nuclear, Imports, Generation.
            - Curtailment Data: Solar Curtailment, Wind Curtailment.
            - Weather Data: Temperature, Humidity, Cloud Cover, Solar Energy, Wind Speed.
        - Dropped irrelevant features such as preciptype, uvindex, visibility, and windgust.
        2. Data Merging:
          - Merged production and curtailment data on date, hour, and interval.
          - Integrated weather data using the datetime field.
        3. Handling Missing Values:
          - Dropped columns with excessive missing data.
        4. Imputed remaining missing values using:
          - Mean for continuous variables.
          - Zeros for fields like precipitation.
        5. Resampling:
            Aggregated data to an hourly level for consistency.
        6. Validation:
          - Removed duplicates and verified data consistency.
    This process ensured a clean and concise dataset focused solely on features relevant to the research.

### Exploratory Data Analysis (EDA)

The Exploratory Data Analysis for this project focused on understanding the relationships between renewable energy production, weather conditions, and electricity demand, along with identifying patterns in energy curtailment.
#### Distribution Analysis

  ![image](https://github.com/user-attachments/assets/3a25469a-173e-43d9-8767-e69b9108c07b)

#### Seasonal Decomposition

  ![image](https://github.com/user-attachments/assets/f7efb954-7140-455f-8a11-dba85c9c2e81)

#### Timeseries Analysis

  ![image](https://github.com/user-attachments/assets/58eaf84c-5c48-4e1f-92ef-b5b872ae7978)

#### Overview Of the EDA:
  1. **Distribution Plots of Key Features:**

      - Load: Skewed distribution with peaks indicating typical electricity demand levels. Outliers suggest periods of high demand.          
      - Solar: Heavily skewed, with most observations at low generation levels, reflecting variability in solar energy availability.
      - Wind: Broad distribution, indicating a wider range of wind energy generation compared to solar.
      - Renewables: Displays combined contributions of solar and wind, peaking during optimal renewable generation periods.
      - Temperature: Near-normal distribution, capturing seasonal temperature variations.
      - Windspeed: Skewed distribution, reflecting varying wind speeds across time.
  2. **Seasonal Decomposition of Load:**

       - Observed Data: Captures the hourly variations in electricity load over time.
       - Trend Component: Highlights long-term trends, such as increasing electricity demand over the years.
       - Seasonal Component: Displays recurring seasonal patterns driven by factors like temperature and seasonal energy consumption.
       - Residual Component: Represents anomalies and variations not explained by trends or seasonality.
  3. **Time Series of Load, Renewables, and Imports:**

        - Electricity Load: Shows high variability with pronounced peaks during high-demand periods.
        - Combined Renewables: Fluctuates with weather conditions, showing higher contributions during favorable periods.
        - Imports: Fills the gap between electricity demand and renewable energy supply, increasing during renewable shortfalls.

These visualizations provide critical insights into the variability and trends of energy demand, renewable generation, and the role of imports, forming a solid foundation for modeling and analysis.

### Model Development

#### **Research Question 1:**
**How do weather conditions and energy generation factors influence electricity demand during peak load seasons, and what key drivers significantly impact these fluctuations?**

- **Objective**: Predict electricity load during peak demand using weather conditions and energy generation factors.
- **Approach**:
  - Used regression models to capture non-linear relationships between weather variables (e.g., `Temperature`, `Humidity`) and electricity demand.
  - Key models tested:
    - **Linear Models**: Underperformed significantly (R² ≈ 0.70), failing to account for the complexities of load patterns.
    - **Optimized Random Forest**: Achieved high R² (0.892) and lower MSE, effectively capturing load variability.
    - **Optimized XGBoost and Gradient Boosting**: Delivered strong R² scores (0.87–0.89) with consistent generalization for load predictions.
  - **Final Choice**: Random Forest and XGBoost were selected for their superior accuracy and ability to manage non-linear relationships.
- **Feature Importance**: Highlighted `Temperature`, `Renewables`, and `Imports` as significant predictors of electricity load.

---

#### **Research Question 2:**
**How can the risk of consecutive days with low renewable energy generation be effectively predicted based on weather patterns, and what measures can be implemented to maintain reliability during such events?**

- **Objective**: Predict periods of low renewable energy generation using weather data and renewable generation metrics.
- **Approach**:
  - Combined regression and classification models to assess risks of low renewable days.
  - Key models tested:
    - **Regression Models**:
      - **Random Forest & XGBoost**: Demonstrated high R² (0.89), effectively modeling load during low renewable periods.
      - **Linear Models**: Underperformed (R² ≈ 0.65), unable to handle complex patterns.
    - **Classification Models**:
      - **Balanced Random Forest**: High recall (0.80) for detecting low renewable days, ensuring reliable predictions.
      - **Original Random Forest**: Higher accuracy (90.37%) but lower recall (0.71), risking missed low-generation events.
  - **Final Choice**: Balanced Random Forest for classification and Random Forest/XGBoost for regression.
- **Key Features**: `Temperature`, `Wind Speed`, `Humidity`, and `Solar Energy`.

---

#### **Research Question 3:**
**What strategies can be implemented to minimize renewable energy curtailment during periods of high renewable output and low demand, ensuring grid stability and optimal resource utilization?**

- **Objective**: Predict renewable energy curtailment and identify strategies to optimize resource utilization.
- **Approach**:
  - Regression models were applied to understand curtailment patterns and drivers.
  - Key models tested:
    - **Linear Models**: Underperformed (R² ≈ 0.32), failing to capture curtailment complexities.
    - **Optimized Random Forest**: Selected for its high R² (0.71) and low MAE, effectively modeling curtailment scenarios.
    - **XGBoost**: Strong training R² (0.85) and reasonable generalization, making it suitable for curtailment analysis.
  - **Final Choice**: Random Forest and XGBoost for their superior performance in modeling non-linear interactions.
- **Insights**:
  - High curtailment linked to high renewable output during low demand periods.
  - Suggested strategies:
    - Energy storage systems.
    - Demand-side management to balance supply and demand.

---

These models provided insights into electricity load patterns, renewable generation risks, and curtailment scenarios, enabling data-driven solutions for grid optimization and stability.



     









