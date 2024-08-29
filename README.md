# Retail-Sales-Prediction

Developed an ANN-based predictive model to forecast department-wide sales integrating ARIMA and ANN for robust forecasting accuracy.Analyzed the impact of markdown strategies on sales during holiday weeks, identifying key patterns and trends that influence sales performance.

## Key Technologies and Skills
-Python
-Scikit-Learn
-Keras
-ARIMA
-Time Series Analysis
-Numpy
-Pandas
-Matplotlib
-Seaborn
-Streamlit
-AWS Deployment


1. Introduction
The objective of this project is to develop robust predictive models for forecasting weekly sales in a retail setting. Given the complexity of the dataset, which includes features such as 'Store', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', and several 'MarkDown' columns, we employed both machine learning and deep learning techniques to maximize predictive accuracy. The challenge of handling significant amounts of missing data, particularly in the 'MarkDown' columns, necessitated a thoughtful approach to model development.

2. Data Cleaning and Preparation
The dataset was first preprocessed to address missing values and ensure appropriate data types. Columns such as 'MarkDown1' to 'MarkDown5' had over 50% null values. To handle this, we used a strategy of filling null values with Time Series Analysis and Machine Learning Model.

Additional preprocessing steps included:

-Standardizing numerical features using StandardScaler.
-Converting date formats to extract features such as Day, Month, and Year.
-Creating lag features, such as Weekly_Sales_Lag1, to capture temporal dependencies.

3. Exploratory Data Analysis (EDA)
A thorough EDA was conducted to understand the relationships between features, sales trends, and the impact of holidays. This included analyzing the correlations between different variables and examining how markdowns and other factors like fuel prices and unemployment rates influenced sales.

4. Feature Engineering
New features were engineered to enhance the model's predictive power:

Lag features to capture temporal patterns.
Holiday-specific features to differentiate regular sales weeks from holiday weeks.
Interaction features between markdowns and holiday variables to capture combined effects.

5. Deep Learning Models
Given the substantial missing data in the 'MarkDown' columns, we adopted a dual-model approach:

Model 1: Trained using all features, including 'MarkDown' columns.
Model 2: Trained without 'MarkDown' columns to assess the influence of these features on predictive accuracy.
Several types of Artificial Neural Network (ANN) architecture with multiple hidden layers, LeakyReLU activation, and Dropout for regularization were applied, The deep learning model underwent several iterations, including adjustments to layer sizes and the addition of L2 regularization to improve the R² score, which ultimately reached 0.85. The model's architecture was optimized to reduce overfitting while maintaining predictive power.This model exhibited strong performance with minimal overfitting. 

Key evaluation metrics included:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)

6. Model Evaluation and Comparison
There were two models were built with and without markdown. Metrics as follows:

For with Markdown
'R2': 0.81
'Mean Absolute Error': 16.8
'Mean Squared Error': 585.3
'Root Mean Squared Error': 24.1

For without Markdown
'R2': 0.83 
'Mean Absolute Error': 15.2
'Mean Squared Error': 493.3
'Root Mean Squared Error': 22.2

7. Model Persistence and Deployment
The final models were saved using the pickle module, ensuring easy deployment in a production environment. The models can be loaded efficiently to make predictions on new data, facilitating real-time decision-making in retail operations. And the entire has been deployed in Amazon Web Service Successfully.

Contact

📧 Email: mmismayil2003@gmail.com

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.
