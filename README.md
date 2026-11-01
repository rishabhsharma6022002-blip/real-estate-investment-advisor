# Real Estate Investment Advisor

### Predicting Property Profitability & Future Value

A Machine Learning and Data Analytics project that explores Indian residential property data, evaluates relative property affordability, and estimates future property values using configurable appreciation scenarios.

## 1. Project Overview

The Real Estate Investment Advisor is an interactive data science project designed to help users understand residential property markets and explore potential investment opportunities.

The project combines data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning, experiment tracking, and an interactive Streamlit application.

It aims to provide data-driven insights into property characteristics, pricing patterns, relative affordability, and potential future property values.

## 2. Project Objectives

- Analyze residential property data across multiple Indian cities and states.

- Identify patterns and relationships between property prices and characteristics.

- Perform data cleaning and preprocessing.

- Develop meaningful features for machine learning.

- Build a classification model to identify relatively affordable properties.

- Estimate future property values using configurable appreciation assumptions.

- Develop an interactive dashboard using Streamlit.

- Track machine learning experiments and model performance using MLflow.

## 3. Technologies & Tools

| Technology |  Purpose |

| Python | Programming language |

| Pandas | Data processing |

| NumPy | Numerical operations |

| Matplotlib | Data visualization |

| Seaborn | Statistical visualization

| Scikit-learn | Machine Learning

| Random Forest | Classification algorithm |

| Streamlit |  Interactive web application | 

| MLflow | Experiment tracking | 

| Joblib | Model serialization |

## 4. Dataset Description

The project uses an Indian housing dataset containing:

| Dataset Information | Value | 

| Total Property Records | 250,000 |

| Number of Features | 23 |

| Cities | 42 |

| States | 20 |

#### Important Dataset Features

- BHK – Number of bedrooms, halls, and kitchens.

- Size_in_SqFt – Property size in square feet.

- Price_in_Lakhs – Property price in lakhs.

- Price_per_SqFt – Price per square foot.

- Year_Built – Year of construction.

- Age_of_Property – Property age.

- Floor_No – Property floor number.

- Total_Floors – Total floors in the building.

- Property_Type – Type of residential property.

- Furnished_Status – Furnishing category.

- State – Property state.

- City – Property city.

- Locality – Property locality.

The dataset also contains information about nearby schools, hospitals, amenities, parking, security, and property availability.

## 5. Project Workflow

### Step 1: Data Inspection

- Loaded and inspected the original dataset.

- Examined dataset dimensions and data types.

- Identified numerical and categorical features.

- Checked missing values and duplicate records.

- Investigated invalid floor information.

### Step 2: Data Preprocessing

- Removed duplicate records.

- Cleaned categorical values.

- Identified and handled invalid floor records.

- Applied median imputation to numerical features.

- Created a corrected price-per-square-foot representation.

### Step 3: Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand property market patterns.

#### Visualizations include:

- Property price distribution.

- Property size distribution.

- Relationship between property size and price.

- Property type distribution.

- Correlation heatmap.

### Step 4: Feature Engineering

Created additional features to support analysis and modelling.

#### Good_Investment

A binary classification target:

1 – Property price per square foot is at or below the median price per square foot for its city.

0 – Property price per square foot is above its city's median.

This target represents relative affordability, not independently verified investment profitability.

#### Future_Price_5Y_Lakhs

An illustrative future property value calculated using an assumed annual appreciation rate of 8% over five years.

### Step 5: Machine Learning Model

Implemented a Random Forest Classifier using Scikit-learn.

Model Configuration:

| Parameter             |   Value | 
 

| Algorithm             | Random Forest Classifier | 

| Number of Estimators  | 100 |

| Maximum Depth         | 15 | 

| Minimum Samples Split | 10 |

| Random State          | 42 |

| Class Weight          | Balanced |

| Training Data         | 80% |

| Testing Data          | 20% |

The preprocessing pipeline includes numerical imputation, categorical encoding, and model training.

### Step 6: Model Evaluation

The trained classification model was evaluated using standard classification metrics.

| Evaluation Metric | Result | 

|  Accuracy         | 84.07% |

| Precision         | 84.48% | 

| Recall            | 83.47% |

| F1 Score          | 83.97% |

| ROC-AUC           | 95.65% |

These results reflect performance on the constructed relative-affordability target. They should not be interpreted as proof of actual investment profitability.

### Step 7: MLflow Experiment Tracking

MLflow was integrated to track and manage machine learning experiments.

#### The experiment records:

- Model parameters.

- Evaluation metrics.

- Training runs.

- Trained model artifacts.

#### Experiment Name: Real_Estate_Investment_Advisor

#### Run Name: Random_Forest_Classifier

#### Saved Model: classification_model.pkl

### Step 8: Future Property Price Forecasting

The application includes a scenario-based future property price calculator.

#### Users can configure:

- Current property price.

- Annual appreciation rate.

- Forecast duration.

The application calculates an estimated future value using compound annual appreciation.

Note: This feature is an assumption-based financial calculation, not an ML regression model or a guaranteed prediction of actual market prices.

## 6. Streamlit Application Features

The interactive application contains four main sections.

#### Home

- Project introduction.

- Dataset overview.

- Key project statistics.

#### Property Investment

- Property information input.

- Property affordability classification.

- Model-based prediction.

#### Price Forecast

- Current property price input.

- Adjustable annual appreciation rate.

- Customizable forecast duration.

- Estimated future property value.

#### Analytics Dashboard

- City-wise property price analysis.

- Property type distribution.

- Property price distribution.

- Dataset preview.

### 📁 7. Project Structure

```text
REAL_ESTATE_ADVERTISEMENT_ADVISOR/
│
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── feature_engineered_data.csv
│
├── models/
│   └── classification_model.pkl
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── inspect_data.py
│   ├── data_quality_checks.py
│   ├── preprocess_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── classification.py
│   └── regression.py
│
├── app.py
├── README.md
└── .gitignore```

## 8. Installation & Execution

Follow these steps to run the project locally.

#### Prerequisites

- Python 3.10 or later

- pip

- Git

- Visual Studio Code (recommended)

### Step 1: Clone the Repository

`````git clone YOUR_GITHUB_REPOSITORY_URL`````

#### Navigate to the project directory:

``````cd REAL_ESTATE_ADVERTISEMENT_ADVISOR``````

### Step 2: Create a Virtual Environment

``````python -m venv .venv``````

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```.\.venv\Scripts\Activate.ps1 ```

If PowerShell blocks activation, you can run Python using the environment's executable directly:

````.\.venv\Scripts\python.exe````

### Step 4: Install Dependencies

````````pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib mlflow````````

### Step 5: Run the Streamlit Application

``````streamlit run app.py``````

The application will be available in your browser, usually at:

http://localhost:8501

## 9. Limitations

- The classification target is derived from city-level median price per square foot.

- The classification model estimates relative affordability rather than independently verified investment returns.

- The dataset and its patterns require independent validation before real-world use.

- The future price calculator relies on assumed appreciation rates.

- Actual property prices may vary due to market conditions, location, infrastructure, demand, and economic factors.

- Forecasts are illustrative and should not be treated as financial advice.

## 10. Future Enhancements

- Integrate real-time property market datasets.

- Improve model validation and feature selection.

- Compare additional machine learning algorithms.

- Incorporate location-specific appreciation trends.

- Add investment risk assessment.

- Develop property comparison functionality.

- Introduce personalized property recommendations.

- Improve the dashboard with additional interactive visualizations.

## 11. Author

Real Estate Investment Advisor

Machine Learning | Data Analytics | Python | Streamlit

## Disclaimer

This project is developed for educational and machine learning application purposes. Its outputs are estimates based on the available dataset and selected assumptions. They should not be considered guaranteed investment returns or professional financial advice.