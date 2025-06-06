# Used Car Price Prediction

This project aims to predict the price of used cars based on various features extracted from the `autos.csv` dataset. It involves data cleaning, exploratory data analysis, feature engineering, and building a linear regression model to estimate car prices.

## Dataset

The primary dataset used in this project is `autos.csv`.

After cleaning and feature engineering, the key features used for prediction include:
*   `vehicleType`: The type of vehicle (e.g., SUV, Kleinwagen).
*   `yearOfRegistration`: The year the vehicle was first registered.
*   `gearbox`: Type of transmission (manual or automatic).
*   `powerPS`: Engine power in PS (Pferdestärke).
*   `model`: The model name of the vehicle.
*   `kilometer`: Mileage of the car in kilometers.
*   `fuelType`: Type of fuel the vehicle uses (e.g., petrol, diesel).
*   `brand`: The manufacturer/brand of the vehicle.
*   `notRepairedDamage`: Indicates if the vehicle has unrepaired damage (yes/no).
*   `saleDurationDays`: The number of days the car ad was online (engineered feature).

## Methodology

### Data Cleaning
*   Handled missing values by imputing or removing rows.
*   Removed rows with erroneous data (e.g., `price` = 0).
*   Engineered the `saleDurationDays` feature from existing date columns.
*   Standardized categorical values (e.g., ''ja''/''nein'' to ''yes''/''no'').

### Exploratory Data Analysis (EDA)
*   Visualized distributions of numerical features using histograms and boxplots.
*   Analyzed categorical features using countplots.
*   Examined relationships between features and the target variable (`price`).
*   Generated a correlation heatmap for numerical features.

### Preprocessing
*   **Numerical Features:** Missing values were imputed using the median. Features then underwent a log transformation (`np.log1p`), followed by the creation of polynomial features (degree 5), and finally, scaling using StandardScaler.
*   **Categorical Features (`gearbox`, `fuelType`, `notRepairedDamage`):** Missing values were imputed using the most frequent value, then OneHotEncoder was applied.
*   **Other Categorical Features (`vehicleType`, `model`, `brand`):** Missing values were imputed using the most frequent value, followed by BinaryEncoder.

### Modeling
*   A Linear Regression model was employed to predict car prices.

### Hyperparameter Tuning
*   GridSearchCV was utilized to find the optimal `degree` for PolynomialFeatures in the numerical pipeline.

### Evaluation
*   The model''s performance was evaluated using R-squared (R²) and Root Mean Squared Error (RMSE) on the test set. RMSE was also calculated after reversing the log transformation on the price predictions.

## Files in Repository

*   `used cars.ipynb`: The main Jupyter Notebook containing the entire workflow from data loading, cleaning, EDA, preprocessing, model training, and evaluation.
*   `autos.csv`: The raw dataset containing information about used car listings.
*   `linearregression_pipeline.pkl`: The serialized (pickled) machine learning pipeline, which includes the trained Linear Regression model and all preprocessing steps. This can be loaded for making predictions on new data.
*   `data.txt`: A text file found in the repository. Its specific purpose within this project is not detailed in the notebook and may require further investigation.

## How to Use

1.  **Environment Setup:**
    *   Ensure you have Python installed.
    *   Install the necessary libraries. You can typically do this using pip:
        ```bash
        pip install pandas numpy matplotlib seaborn scikit-learn category_encoders joblib
        ```
2.  **Running the Analysis:**
    *   Open and run the `used cars.ipynb` notebook in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab, VS Code with Python extension).
    *   The notebook will load the `autos.csv` data, perform cleaning, EDA, train the model, and save the pipeline to `linearregression_pipeline.pkl`.
3.  **Using the Trained Model (Example):**
    *   To use the saved model for predictions on new data (assuming it has the same features as the training data before preprocessing):
        ```python
        import joblib
        import pandas as pd
        import numpy as np

        # Load the pipeline
        pipeline = joblib.load("linearregression_pipeline.pkl")

        # Create a sample DataFrame with new data (ensure all necessary columns are present)
        # Example: new_data = pd.DataFrame({...})
        # Note: The new_data should have columns like ''vehicleType'', ''yearOfRegistration'', etc.

        # Make predictions (these will be log-transformed prices)
        # log_predictions = pipeline.predict(new_data)

        # To get prices in the original scale, apply inverse transformation
        # actual_price_predictions = np.expm1(log_predictions) # or np.exp(log_predictions) - 1 if log1p was used
                                                              # The notebook used np.log1p, so np.expm1 is appropriate.
        # print(actual_price_predictions)
        ```
    *   *Note: The example for loading and using the model assumes new data is in a pandas DataFrame format with the same original features as `x_train`.*

## Results

The Linear Regression model, after preprocessing and hyperparameter tuning (for polynomial feature degree), achieved the following performance on the test set:

*   **R-squared (R²):** Approximately 0.739 (This is the R² score on the log-transformed prices, `y_preprocessed_test`).
*   **Root Mean Squared Error (RMSE) on log-transformed prices:** Approximately 0.635.
*   **Root Mean Squared Error (RMSE) on original price scale:** The notebook reports a value around 1.887 after applying `np.exp()` to the predictions and true values. This indicates the typical error in the predicted price in the original currency unit.
*   **Confidence Interval (95%) for RMSE on original price scale:** The notebook calculates a 95% confidence interval for the RMSE on the original price scale, which was approximately [6115, 12267] (actual values might vary slightly based on exact notebook output during runs).

These metrics suggest the model provides a reasonable estimation of used car prices, though there is still room for improvement.
