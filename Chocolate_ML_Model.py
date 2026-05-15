import numpy as np
import openpyxl
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_excel('/Users/Bibhanshu.Swain/Documents/Chocolate_data/Chocolate_data.xlsx') ## read the data
y = df["Amount"]  ## I want to assign the main value column  to Y
x = df.drop(columns=["Amount","ShipmentID"]) ##because we are using amount in Y

##in the below ship date is dropped and deleted but we have 3 new coloumns in our dataset
##that is day.month,year
x["Shipdate"] = pd.to_datetime(x["Shipdate"])
x["year"] = x["Shipdate"].dt.year
x["month"] = x["Shipdate"].dt.month
x["day"] = x["Shipdate"].dt.day
x = x.drop(columns=["Shipdate"])

x_encoded = pd.get_dummies(x)
x_encoded_train,x_encoded_test,y_train,y_test = train_test_split(
    x_encoded,y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression() ## blank brain
model.fit(x_encoded_train, y_train) ## feeding info of train data x and their values in Y so that model can train on pattern
Predicted_Thing = model.predict(x_encoded_test) ##based on learning seeing what model learn so X-test
R2 = r2_score(y_test,Predicted_Thing) ## Ytest actual vs Predicted things are results I predicted
MSE = mean_squared_error(y_test,Predicted_Thing) ## Ytest actual vs Predicted things are results i predicted
RMSE = np.sqrt(MSE)
##print("MSE:", MSE,"RMSE:", RMSE)
## MSE: 2470023.6496322653 RMSE: 1571.6308884824914
##Model performs well overall (R² = 0.84) -- Verry good
##But has moderate prediction error (~1571)
##And struggles with some large-value cases
##lets try Random forest and see how it works
model_Random_forest = RandomForestRegressor(random_state=42)
model_Random_forest.fit(x_encoded_train, y_train) ## this is real Learning Stage
pred_randomforest = model_Random_forest.predict(x_encoded_test)
##print("Actual",y_test[:10].values)
##print("Random_forest Prediction",pred_randomforest[:10])
##print("Linear regression prediction",Predicted_Thing[:10])
Random_forest_R2 = r2_score(y_test,pred_randomforest)
##print("R2 Score(Random Forest):", Random_forest_R2) ## we get 0.9430164681713803
MSE_Random_Forest = mean_squared_error(y_test, pred_randomforest)
RMSE_Random_forest = np.sqrt(MSE_Random_Forest)
##print("R2(Linn Regg)",R2,"R2 Score(Random Forest):", Random_forest_R2)
##print("MSE(Linn Regg):", MSE,"RMSE(Linn Regg):", RMSE)
##print("MSE(Random Forest):", MSE_Random_Forest,"RMSE(Random Forest):", RMSE_Random_forest)

##--------------------------------------------------------------------------------------------------------------------##
##output :
##R2(Linn Regg) 0.8432946723736701 R2 Score(Random Forest): 0.9430164681713803
##MSE(Linn Regg): 2470023.6496322653 RMSE(Linn Regg: 1571.6308884824914
##MSE(Random Forest): 898186.8924832543 RMSE(Random Forest): 947.7272247241051

####My views :
####R2 of linn egg was - 0.8432946723736701 while for random forest was 0.94 this is a huge boost
# and if not equal to 1 close to 1 - I think I am not sure using gradient boosting
# I will be seeing this more close to 1 , also note I have not used n_estimators or updated it to 500
# its by default which is 100 , so twisting that might make my r2 and the model a bit better

##now coming to MSE and RMSE :
##lets talk about RMSE as its the main thing
##RMSE(Linn Regg): 1571.6308884824914 --- acc to me its BIG a bit more error
##RMSE(Random Forest): 947.7272247241051 --- a significant decrease which is far better

##--I hope in Gradient Boosting the RMSE will fall down more !!

##Gradient Boosting
model_GB = GradientBoostingRegressor(
    n_estimators = 300, ## no of trees
    learning_rate = 0.05,## How much each tree contributes
    max_depth = 3, ##Tree Complexity
    random_state = 42

)
model_GB.fit(x_encoded_train,y_train)
Pred_GB = model_GB.predict(x_encoded_test)
##print("Actual:", y_test[:10].values)
##print("Linn Regg Predicted:", Predicted_Thing[:10])
##print("Random_forest Prediction",pred_randomforest[:10])
##print("Gradient_boost_pred", Pred_GB[:10])

## after this i might feel Random forest is good just looking at 10 values but Gradient boost is slowling learning
##in jsut first 10 values it might not have completed the full learning
##lets go for Mse and RMSE and then makle a conclusion
GB_R2 = r2_score(y_test,Pred_GB)
##print(GB_R2)
##oky I calculated R2 and Random forest R2 is - 0.9430164681713803for GB - 0.9349007749776602
# again this supports a bit to my intuition Random first is a bit good or GB is close its like double checking
# random forest not relaying on random forest only .. yes MSE RMSe is left but just at this point of time my intuition
MSE_Gradient_Boost = mean_squared_error(y_test, Pred_GB)
RMSE_Gradient_Boost = np.sqrt(MSE_Gradient_Boost)
##print("MSE(Gradient Boost):", MSE_Gradient_Boost,"RMSE(Gradient Boost):", RMSE_Gradient_Boost)
##print("MSE(Random Forest):", MSE_Random_Forest,"RMSE(Random Forest):", RMSE_Random_forest)

##“Random Forest performed best with an RMSE of ~947 and R² of 0.94. Gradient Boosting showed comparable performance (RMSE ~1013),
# indicating that the dataset has strong learnable patterns. Using multiple models helped validate consistency,
# and Random Forest remains the preferred model due to lower prediction error.”


##Feature Importance - It tells you which features your model relies on the most
##For Random Forest :
importance_Random_Forest = model_Random_forest.feature_importances_
feature_name_RF = x_encoded.columns
feature_importance_df_RF = pd.DataFrame(
    {
    'Feature_RS': feature_name_RF,
    'Importance_RS': importance_Random_Forest
    }
)
feature_importance_df_RF = feature_importance_df_RF.sort_values(
    by='Importance_RS',
    ascending=False
)
##print(feature_importance_df_RF.head(10)) ##top 10 factors

## Your model learned:
##Boxes is the main driver of Amount ✅
##Product type (PID) also affects pricing ✅
##Other features are less influential ✅
#Feature importance analysis revealed that ‘Boxes’ was the dominant predictor (~73%),
#indicating a strong direct relationship with the target variable ‘Amount’.
#Product-level features (PID) also contributed, reflecting variations in product pricing.”

##gradient_boost
##Feature Importance - It tells you which features your model relies on the most
##For Random Forest :
importance_Gradient_boost = model_GB.feature_importances_
feature_name_GB = x_encoded.columns
feature_importance_df_GB = pd.DataFrame(
    {
    'Feature_GB': feature_name_GB,
    'Importance_GB': importance_Gradient_boost
    }
)
feature_importance_df_GB = feature_importance_df_GB.sort_values(
    by='Importance_GB',
    ascending=False
)
##print(feature_importance_df_GB.head(10)) ##top 10 factors

##Both models identified the same dominant feature (Boxes) and similar secondary features (PID),
# which confirms that the learned patterns are reliable.”
#Random Forest = parallel learning
#Gradient Boosting = sequential error correction


##Simulation : predicting a new case
##“Create a fake new data point and ask the model to predict Amount”

Box_input = int(input('Enter No of Boxes: '))
year = int(input('Enter year: '))
month = int(input('Enter Month: '))
Day = int(input('Enter Day: '))
print(f"the date you added in(DD/MM/YYYY) :{Day}/{month}/{year}")
Product_id = input('Please enter the Product ID you want to predict format eg (PID_P01) : ')

new_data = pd.DataFrame({   #THIS CREATES A SINGLE ROW DATA i want to predict for 50 boxes in dec 2024
    'Boxes': [Box_input],
    'year': [year],
    'month': [month],
    'day': [Day]
})
new_data_encoded = pd.get_dummies(new_data) ##  Convert to encoded format

# Align with training columns
new_data_encoded = new_data_encoded.reindex(  ## my model is trained on the encoded coloumn like - Boxes, year, month, day,
##PID_P01, PID_P02, PID_P03, ........ PID_P22, but our current new data only have - Boxes, year, month, day !!
    ## . reindex - It forces new_data to look EXACTLY like training data after this -
##Boxes, year, month, day,
##PID_P01 = 0
##PID_P02 = 0
##PID_P03 = 0
##...
##PID_P22 = 0
    columns=x_encoded.columns,
    fill_value=0
)
##new_data_encoded[Product_id] = 1 ## we are choosing the product 5 so pid_p05 = 1 all other zero its like moth boxex and all fo prod 5

if Product_id not in x_encoded.columns:
    print("Invalid Product ID! Please enter correct PID like PID_P01")
else:
    new_data_encoded[Product_id] = 1
predicted_amount = model_Random_forest.predict(new_data_encoded) ##Make prediction
print("Predicted Amount:", predicted_amount)
print("top 10 factors in result",feature_importance_df_GB.head(10)) ##top 10 factors
