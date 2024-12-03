### Updated How to Run the Project

Follow these steps to set up, prepare, and run the project, including the usage of **LIME (Local Interpretable Model-Agnostic Explanations)** plots for model interpretability.

---

#### **1. Prerequisites**
Ensure the following libraries are installed:
- **Python**: Version 3.8 or higher
- **Libraries**: Install required libraries using:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn plotly statsmodels lime openpyxl
  ```

---

#### **2. Data Preparation**
Organize the following datasets:
- **Curtailment Data**: Example file: `productionandcurtailmentsdata_2020.xlsx`
- **Weather Data**: Example file: `California hourly data 2020-01-01 to 2024-09-30.csv`

Ensure file paths are updated to match your local or Google Colab environment.

---

#### **3. Running the Project**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Data-606-Capstone-Project.git
   cd Data-606-Capstone-Project
   ```

2. **Open the Notebook or Script**:
   - For Jupyter Notebook:
     ```bash
     jupyter notebook Team8_Data606_Capstone_Project.ipynb
     ```
   - For Google Colab:
     Upload the notebook file to Colab and update file paths.

---

#### **4. File Execution Steps**

##### **Loading Curtailment and Production Data**
For each year, curtailment and production data were loaded using the following method:
1. **Load Curtailment Data**:
   ```python
   curtailment_data_2020 = pd.read_excel(
       r"/content/drive/Shareddrives/DATA606 capstone project/productionandcurtailmentsdata_2020.xlsx", 
       sheet_name="Curtailments"
   )
   curtailment_data_2020.columns = [feature.lower() for feature in curtailment_data_2020.columns]
   ```

2. **Load Production Data**:
   ```python
   production_data_2020 = pd.read_excel(
       r"/content/drive/Shareddrives/DATA606 capstone project/productionandcurtailmentsdata_2020.xlsx", 
       sheet_name="Production"
   )
   production_data_2020.columns = [feature.lower() for feature in production_data_2020.columns]
   ```

3. **Load Weather Data**:
   ```python
   weather_data_2020 = pd.read_csv(r"/content/drive/Shareddrives/DATA606 capstone project/California hourly data 2020-01-01 to 2024-09-30.csv")
   ```

Repeat the above for subsequent years as necessary.

##### **Merge Datasets**
Use a script or function to merge curtailment, production, and weather datasets on a common field like `datetime`.

---

#### **5. Key Steps in Execution**
1. **Data Cleaning**:
   - Drop unnecessary columns (e.g., `stations`, `preciptype`, `reason`).
   - Handle missing values and duplicates.
   - Standardize column names to lowercase.

2. **Feature Engineering**:
   - Derive metrics like `combined_renewables` (sum of solar and wind) and `total_curtailment` (solar_Curtailment + wind_Curtailment).

3. **EDA**:
   - Perform visualizations for time-series analysis, correlations, and feature distributions.

4. **Modeling**:
   - Use regression models such as Linear Regression, Random Forest, XGBoost, and Gradient Boosting for predictions.

5. **Model Interpretability with LIME**:
   - Use **LIME plots** to explain predictions from complex models like Random Forest and XGBoost:
   - Example:
     ```python
                 import lime
                import lime.lime_tabular
                import numpy as np
                
                # Initialize the LIME explainer for the optimized XGBoost Regressor
                explainer_xgb_tuned = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.array(X_train),
                    feature_names=X.columns,  # Ensure this matches the original feature names used for training
                    mode='regression'
                )
                
                # Choose a random instance from the test set to explain (modify index as needed)
                instance_index_xgb_tuned = 5
                instance_to_explain_xgb_tuned = X_test.iloc[instance_index_xgb_tuned].values.reshape(1, -1)
                
                # Explain the prediction for the selected instance using LIME for the optimized XGBoost model
                exp_xgb_tuned = explainer_xgb_tuned.explain_instance(
                    data_row=X_test.iloc[instance_index_xgb_tuned],
                    predict_fn=best_xgb_reg.predict  # Using the optimized XGBoost model
                )
                
                # Display the explanation in notebook or console
                exp_xgb_tuned.show_in_notebook(show_table=True)

     ```
   - Use LIME plots to understand which features most influenced the model’s predictions for individual samples.

---

#### **6. Example Execution on Google Colab**
- Mount Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- Update file paths:
  ```python
  curtailment_data = pd.read_excel(
      '/content/drive/Shareddrives/DATA606 capstone project/productionandcurtailmentsdata_2020.xlsx', 
      sheet_name='Curtailments'
  )
  production_data = pd.read_excel(
      '/content/drive/Shareddrives/DATA606 capstone project/productionandcurtailmentsdata_2020.xlsx', 
      sheet_name='Production'
  )
  ```

---

#### **7. Output**
- **Data Insights**:
  - Cleaned, merged datasets for curtailment, production, and weather data.
- **Visualizations**:
  - Feature distributions, seasonal decomposition, and trend analysis.
- **Model Interpretability**:
  - LIME plots showcasing feature importance for individual predictions.

By incorporating LIME, the project provides transparency into model decisions, enhancing trust in predictions and enabling better decision-making.
