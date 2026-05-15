# Sales-Amount-Prediction-using-Machine-Learning-Models

🍫 Sales Amount Prediction using Machine Learning
📌 Overview
This project builds a machine learning model to predict sales amount (revenue) based on shipment data, product type, and delivery details.

🎯 Problem Statement
Predict the Amount (sales revenue) using features such as:

Number of boxes shipped
Product type (PID)
Shipment date


🧠 Approach

Data Preprocessing
Removed irrelevant columns (ShipmentID)
Converted Shipdate → year, month, day
Applied one-hot encoding



Models Used
Linear Regression
Random Forest (Best Model ✅)
Gradient Boosting


Evaluation Metrics
R² Score
RMSE (Root Mean Squared Error)

Result:
Linear Regression   R2 0.84  RME 1571
Random Forest       R2 0.94  RME 947
Gradient Boosting   R2 0.93  RME 1013


🔍 Key Insights
-Boxes is the most important feature (~75–80%)
-Product type (PID) influences pricing
-Ensemble models significantly outperform linear regression
-Consistent performance across models improves confidence



---------------------------------------------------------------------------------------------------------------------------------------
🚀 Prediction Demo (NEW FEATURE)

--predict_amount(50, 'PID_P05')

Input :
Product = Mint Chip Choco (P05)
Boxes = 50
Date = 10 Dec 2024


Predicted Amount ≈ 431 (Revenue is primarily driven by order quantity (Boxes), with product pricing acting as a secondary factor.)




