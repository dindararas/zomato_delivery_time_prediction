# ZOMATO DELIVERY TIME PREDICTION
## 🛵PROBLEM STATEMENT
Zomato is India's #1 food delivery app with market share in India reaching 58% in Q1 2025. While Swiggly - main competitor - offers 10-minute food delivery service,  this company has been struggling with delivery time. In 2022, customers had to wait for more than 25 minutes to get their food. This could led to customer dissatisfaction and increases the risk of customer churn. Check streamlit [here](https://dindararas-zomato-delivery-time-prediction-app-t9er5m.streamlit.app/)

## 🎯 OBJECTIVES
This project aims to :
1. Identify key risk factors influencing delivery time
2. Develop ML-based models to predict delivery time
3. Derive Actionable Insights

## 📊 DATASET OVERVIEW
Dataset can be downloaded on [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset/data). Dataset provides a comprehensive view of delivery operations, including delivery person details, order timestamps, weather conditions, and road traffic density, and more

Features overview :
| Column Name | Description |
|------|------|
| `ID` | Unique identifier for each delivery |
| `Delivery_person_ID` |  Unique identifier for each delivery person |
| `Delivery_person_Age` |  Age of the delivery person |
| `Delivery_person_Ratings` | Ratings assigned to the delivery person |
| `Restaurant_latitude` | Latitude of the restaurant  |
| `Restaurant_longitude` |  Longitude of the restaurant |
| `Delivery_location_latitude` |  Latitude of the delivery location  |
| `Delivery_location_longitude` | Longitude of the delivery location |
| `Order_Date` |  Date of the order |
| `Time_Ordered` | Time the order was placed  |
| `Time_Order_picked` | Country where the customer made the purchase  |
| `Weather_conditions` |  Weather conditions at the time of delivery |
| `Road_traffic_density` | Density of road traffic during delivery|
| `Vehicle_condition` | Condition of the delivery vehicle |
| `Type_of_order` | Type of order (e.g., dine-in, takeaway, delivery)  |
| `Type_of_vehicle` |Type of vehicle used for delivery |
| `Multiple_deliveries` |  Indicator of whether multiple deliveries were made in the same trip |
| `Festival` | Indicator of whether the delivery coincided with a festival|
| `City` | City where the delivery took place|
| `Time_taken (min)` |  Time taken for delivery in minutes|

## 🔎 KEY FINDINGS
### Delivery Time
<img width="298" height="212" alt="image" src="https://github.com/user-attachments/assets/0037b4c0-e0e9-44d5-9e79-afc4d550e4be" />

**Insights :**
1. Overall, **delivery person riding a motorcyle took longer time to deliver food regardless road traffic conditions**
2. During traffic jam, bicycle is the best option as vehicle for delivery. On average, bicycle deliveries took 24.7 minutes in traffic jams compared to 28–32 minutes for other vehicles. This is likely due to the flexibility of bicycle to navigate tight spaces and its ability to access certain streets that may be restricted for motorcycles (e.g. bike paths)
3.  In contrast, scooter and electric scooter emerged as the most efficient vehicles during low traffic.

### Peak Traffic & Peak Orders
<img width="553" height="182" alt="image" src="https://github.com/user-attachments/assets/0f9d3013-a102-4079-b751-116befee380c" />

**Insights :**
1. The peak ordering hours occur at 19.00 and 22.00. Since the high probability of traffic jam also occurred at 19.00, it could lead to delivery issues
2. Number of orders increases in the evening. This makes sense because most of people have finished their day and spend their time with family or friends. Ordering food for dinner becomes a natural part of this routine

## 🤖 PREDICTION MODEL
### Model Comparison
<img width="546" height="182" alt="image" src="https://github.com/user-attachments/assets/36dbb80e-698d-47c6-b8ec-6fee02826951" />

**Insights :**
1. Random Forest and Decision Tree are overfitting to the training data, shown by large gap between RMSE training and testing data
2. Although Linear Regression good for data generalization, this model is possibly underfitting
3. There is a very slight difference in model performance between scaled and unscaled datasets

### Hyperparameter Tuning
<img width="268" height="155" alt="image" src="https://github.com/user-attachments/assets/c9b19879-bfe2-44f4-ade7-3e669de4669c" />

**Insights:**
Hyperparameter tuning has improved LightGBM performance while making it not overfitting

### Model Intrepretability
<img width="902" height="572" alt="image" src="https://github.com/user-attachments/assets/671505c0-ab1a-4f63-bd6b-cc9962b153d3" />

Features importance from LightGBM model's prediction :
* `Road_traffic_density` : this feature has the highest importance (+2.78) for predicting delivery time. This result highlights the neccessary to deal with road traffic during delivery
* `Delivery_person_Ratings` : this feature ranks second for the highest importance (+2.42). As it can be seen in the correlation analysis, this feature has a moderate negative correlation with delivery time. This means that the higher ratings the faster delivery time
* `Delivery_person_Age` : it can be seen that older delivery person, longer delivery person
* `distance_km` : as far distance between restaurant and delivery location, the slower delivery time
* All the encoded features of `Type_of_vehicle`, `Weather_conditions`, and `City` have very little impact on model

## ✍ BUSINESS RECOMMENDATIONS
**1. Route Optimization**

* Optimize routes during high traffic density/traffic jam
* Reduce the number of deliveries at the same time
* Inform real-time road conditions to delivery person


**2. Leverage High-Rated Delivery Person**

* Prioritize high-rated delivery person for time-sensitive orders
* Provide incentives to high-rated delivery person

**3. Manage Delivery Person by Age**
* Assign shorter distance or less traffic to old delivery person
* Provide training for adults to elderly people to speed up their delivery time

**4. Give Real-Time  ETA**
* Update the expected ETA to customers based on real-time road and delivery person conditions

**5. Maintain Vehicle Conditions**
* Do regular checking and maintenance on all the vehicles
* Partner with vehicle service center

**6. Allocate delivery person who is close to restaurant to the orders**


