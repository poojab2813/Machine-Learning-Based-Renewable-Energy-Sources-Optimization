# Data-606-Capstone-Project

# 🌍 Intelligent Grid Machine Learning Driven Optimization For Renewable Energy Stability

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5.1-red)
![Project Status](https://img.shields.io/badge/Project%20Status-Complete-brightgreen)
![Pandas](https://img.shields.io/badge/Pandas-1.4.3-green)
![NumPy](https://img.shields.io/badge/NumPy-1.23.3-lightgrey)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.2-blueviolet)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellowgreen)

## 📚 Table of Contents
1. [Introduction](#Introduction)
2. [Project Overview](#Project-Overview)
3. [Data Collection and Cleaning](#Data-Collection-and-Cleaning)
4. [EDA](#Exploratory-Data-Analysis)
5. [Models](#Model-Development)
6. [Key Results](#Key-Results)
7. [Key Contributions](#Key-Contributions)
8. [How to Run](#How-to-Run)
9. [Future Directions](#Future-Directions)
10. [References](#References)


## **Introduction**

The **Intelligent Grid: Machine Learning-Driven Renewable Energy Optimization** project aims to address the growing challenges of integrating renewable energy sources, such as solar and wind, into the electricity grid. With the increasing adoption of renewables, grid operators face issues like energy curtailment, variability in generation, and balancing supply with demand during peak periods.

This project leverages **machine learning techniques** to analyze energy generation, weather conditions, and electricity demand. It provides actionable insights to improve grid reliability, minimize renewable energy curtailment, and optimize resource utilization. By predicting electricity load, identifying risks of low renewable generation, and proposing strategies for reducing curtailment, this project offers a robust framework for sustainable energy integration.

Key objectives include:
1. **Electricity Demand Prediction**: Understanding how weather conditions influence peak load seasons and identifying key drivers of demand.
2. **Risk Assessment**: Predicting consecutive low renewable energy generation days and maintaining grid reliability.
3. **Curtailment Mitigation**: Proposing strategies to reduce renewable energy curtailment during high-output, low-demand periods.

This project highlights the potential of data-driven solutions in building smarter, more sustainable energy grids to meet the demands of a renewable-driven future.

## **Project Overview**

The **Intelligent Grid: Machine Learning-Driven Renewable Energy Optimization** project focuses on leveraging advanced machine learning techniques to address challenges in integrating renewable energy sources into the electricity grid. As the world transitions toward sustainable energy, grid operators face challenges like variability in renewable generation, energy curtailment during low demand periods, and meeting peak electricity demand efficiently.

---

### **Key Research Questions**
1. **Electricity Demand Prediction**:
   - **Question**: How do weather conditions and energy generation factors influence electricity demand during peak load seasons, and what key drivers significantly impact these fluctuations?
   - **Goal**: Understand the role of weather and energy metrics in electricity demand and predict load during high-demand periods.

2. **Low Renewable Energy Generation Risk Prediction**:
   - **Question**: How can the risk of consecutive days with low renewable energy generation be effectively predicted based on weather patterns, and what measures can be implemented to maintain reliability during such events?
   - **Goal**: Forecast low renewable energy generation days and propose strategies to prevent energy shortages.

3. **Curtailment Mitigation**:
   - **Question**: What strategies can be implemented to minimize renewable energy curtailment during periods of high renewable output and low demand, ensuring grid stability and optimal resource utilization?
   - **Goal**: Reduce curtailment by identifying key drivers and proposing actionable solutions.

### **Core Features**
- **Data Analysis**:
  - Integrated data from energy production, curtailment records, and weather conditions.
  - Conducted detailed Exploratory Data Analysis (EDA) to identify trends and relationships.

- **Machine Learning Models**:
  - Built and optimized models (Random Forest, XGBoost, etc.) for predicting electricity demand, renewable risks, and curtailments.
  - Emphasized interpretability and accuracy for real-world applicability.

- **Actionable Insights**:
  - Key drivers of electricity load identified as `Temperature`, `Renewables`, and `Imports`.
  - Suggested strategies like energy storage systems and demand-side management to mitigate curtailment.

## Data Collection and Cleaning

  #### **Data Sources**
  1. **Curtailment Data**:
     - Includes records of wind and solar energy curtailments.
     - Data Period: 2020–2024.
     - source: https://www.caiso.com/about/our-business/managing-the-evolving-grid
  2. **Production Data**:
     - Captures energy generation by type (solar, wind, nuclear, etc.).
     - Data Period: 2020–2024.
     - source: https://www.caiso.com/about/our-business/managing-the-evolving-grid
  3. **Weather Data**:
     - Historical weather metrics such as temperature, humidity, and wind speed.
     - Source: Visual Crossing. (https://www.visualcrossing.com/weather-data)

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
      
        1. Feature Selection:
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

## Exploratory Data Analysis
The Exploratory Data Analysis for this project focused on understanding the relationships between renewable energy production, weather conditions, and electricity demand, along with identifying patterns in energy curtailment.
#### Correlation Analysis
![image](https://github.com/user-attachments/assets/b5bf906a-183a-43b7-b606-916f179f2ceb)

       The correlation matrix highlights the relationships between key features in the dataset:
            
         - Strong Positive Correlations:
            - `Load` and `Generation` (`0.92`): High electricity demand corresponds to increased energy generation.
            - `Renewables` and `Solar` (`0.97`): Solar energy is a major contributor to renewable generation.
            
         - Strong Negative Correlations:
            - `Imports` and `Renewables` (`-0.73`): Higher renewable generation reduces reliance on imports.
            - `Humidity` and `Solar Energy` (`-0.56`): High humidity is associated with reduced solar energy output.
            
         - Weather Influence**:
            - `Temperature` strongly correlates with `Load` (`0.62`), indicating increased energy demand during warmer conditions.
            - `Cloud Cover` negatively impacts `Solar Energy` (`-0.56`).
            
         This analysis provides insights into how weather conditions and renewable energy generation factors interact, forming the foundation for feature selection in model development.

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

## Model Development
### 🧠 Models and Methods
![Linear Regression](https://img.shields.io/badge/Model-Linear%20Regression-blue)
![Ridge Regression](https://img.shields.io/badge/Model-Ridge%20Regression-green)
![Lasso Regression](https://img.shields.io/badge/Model-Lasso%20Regression-orange)
![Decision Tree](https://img.shields.io/badge/Model-Decision%20Tree-yellow)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-brightgreen)
![Gradient Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-lightblue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Hyperparameter Tuning](https://img.shields.io/badge/Method-Hyperparameter%20Tuning-lightgreen)

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

## Key Results

This project provided valuable insights into the challenges and solutions for renewable energy integration into the electricity grid. By addressing critical questions around energy demand, generation risks, and curtailment, we achieved the following key outcomes:

1. **Impact of Weather on Peak Electricity Demand**:
   - **Findings**: Weather conditions, especially temperature and humidity, significantly influence electricity demand. Renewable sources like solar and wind are crucial during peak demand periods but often require complementary imports to meet demand variability.
   - **Modeling**: Optimized Random Forest and XGBoost models effectively captured the non-linear complexities of electricity load during peak seasons.

2. **Prediction of Low Renewable Energy Generation Days**:
   - **Findings**: Consecutive low renewable energy days are predictable using weather variables like temperature, wind speed, and solar energy. These predictions help maintain grid reliability and prevent energy shortages.
   - **Modeling**: The Balanced Random Forest classifier provided high recall, ensuring reliable detection of critical low-renewable periods.

3. **Strategies to Minimize Renewable Energy Curtailment**:
   - **Findings**: Renewable energy curtailment occurs during periods of high output and low demand. Effective solutions include energy storage systems and demand-side management to enhance grid stability.
   - **Modeling**: Optimized Random Forest and XGBoost models identified key drivers of curtailment, providing actionable insights for reducing wasted renewable energy.

### Key Contributions:
- Developed predictive models for electricity load, renewable generation risks, and curtailment.
- Provided actionable strategies for grid optimization and stability.
- Highlighted the importance of weather conditions in shaping renewable energy outcomes.

## How to Run
For detailed instructions on running the project, check [How_To_Run.md](How_To_Run.md).

## Future Directions

This project provides a robust foundation for renewable energy integration into the grid using machine learning. However, there are several promising future directions to further enhance the scope and impact of this work:

---

#### **1. Demand Response Strategies**
- **Objective**: Improve grid stability by coordinating energy demand.
- **Future Research Direction**:
  - Extend the current model to provide insights into demand-side management.
  - Identify optimal times to encourage energy reductions during predicted periods of low renewable output.
  - Develop frameworks to work alongside predictive models, enabling effective demand response strategies for utilities and grid operators.

---

#### **2. Policy Impact Assessment and Socioeconomic Factors**
- **Objective**: Evaluate the impact of policies on renewable energy output and predict energy reliability scenarios.
- **Future Research Direction**:
  - Integrate socioeconomic factors and policy-related data into predictive models.
  - Assess the impact of subsidies, tax incentives, and government initiatives on renewable energy adoption and performance.
  - Use these insights to shape future energy policies that support sustainable energy systems and promote reliability.

---

These future directions emphasize the potential for advanced modeling techniques to not only predict and optimize grid operations but also influence energy policy and societal adaptation to renewable energy sources. By incorporating demand response frameworks and policy impact assessments, the project could lead to more comprehensive, actionable insights for stakeholders in the energy sector.


## References
This project is supported by insights from key studies and research in the field of renewable energy integration and machine learning. Below are the references:

1. [Machine learning-based utilization of renewable power curtailments under uncertainty by planning of hydrogen systems and battery storages](https://www.sciencedirect.com/science/article/abs/pii/S2352152X21007210?via%3Dihub)  
   - **Authors**: Mohammad H. Shams, Haider Niaz, Jonggeol Na, Amjad Anvari-Moghaddam, J. Jay Liu  
   - **Description**: This study highlights the application of machine learning to optimize renewable power curtailments through planning of hydrogen systems and battery storages.

2. [Artificial intelligence-based prediction and analysis of the oversupply of wind and solar energy in power systems](https://www.sciencedirect.com/science/article/abs/pii/S0196890421010682?via%3Dihub)  
   - **Authors**: Mohammad H. Shams, Haider Niaz, Behzad Hashemi, J. Jay Liu, Pierluigi Siano, Amjad Anvari-Moghaddam  
   - **Description**: This paper discusses AI-based methodologies for predicting and analyzing oversupply scenarios of wind and solar energy in power systems.

3. [Integration of Renewable Energy Sources in future power systems: The role of storage](https://www.sciencedirect.com/science/article/abs/pii/S096014811400593X)  
   - **Authors**: Stefan Weitemeyer, David Kleinhans, Thomas Vogt, Carsten Agert  
   - **Description**: This research explores the critical role of storage solutions in enabling the integration of renewable energy sources in future power systems.

4. [Forecasting Renewable Energy Generation with Machine Learning and Deep Learning: Current Advances and Future Prospects](https://www.researchgate.net/publication/370208950_Forecasting_Renewable_Energy_Generation_with_Machine_learning_and_Deep_Learning_Current_Advances_and_Future_Prospects?enrichId=rgreq-8b6254a33c9febe5c7b08576ae6fc513-XXX&enrichSource=Y292ZXJQYWdlOzM3MDIwODk1MDtBUzoxMTQzMTI4MTE1MjkzNjg3NEAxNjgyMjYyNTY0MDk0&el=1_x_2&_esc=publicationCoverPdf)  
   - **Description**: This paper provides an overview of current advances and future prospects in forecasting renewable energy generation using machine learning and deep learning approaches.

---















