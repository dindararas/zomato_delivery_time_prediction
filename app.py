# import libraries
import streamlit as st
from streamlit_option_menu import option_menu
from geopy.distance import geodesic 
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import pickle
from lightgbm import LGBMRegressor

# Set page configuration
st.set_page_config(
    page_title='Food Delivery Predictive Analytics',
    page_icon='🛵',
    layout='wide',
    initial_sidebar_state='expanded')

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

# ------------ LOAD DATASET ------------
@st.cache_data
def load_data():
    return pd.read_csv('dataset/processed_data.csv', delimiter=',', encoding='latin-1')

df = load_data()

# add new columns
day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
               4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['Day'] = df['DayofWeek'].map(day_mapping)

# function to separate delivery person age
def age_group(age):
    if age < 25:
        return 'Teenage'
    elif age < 65:
        return 'Adult'
    else:
        return 'Elderly'

df['age_group'] = df['Delivery_person_Age'].apply(age_group)

# ----------------- LOAD MODEL -------------
@st.cache_resource
def load_model():
    with open('model/lgbm_best.pkl', 'rb') as f:
        model, feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()

# --------- TITLE & SIDEBAR ---------
st.sidebar.title('Website Pages')

# Sidebar buttons
if st.sidebar.button('🏠 Home'):
    st.session_state['page'] = 'Home'
if st.sidebar.button('📊 Dashboard'):
    st.session_state['page'] = 'Dashboard'
if st.sidebar.button('🛵 Prediction Model'):
    st.session_state['page'] ='Prediction Model'

# ---------- PAGE 1 : HOME -------
if st.session_state["page"] == 'Home':
    st.title('📱 Zomato Food Delivery Predictive Analytics')
    st.subheader('Welcome to **Zomato Delivery Analytics**👋')
    st.subheader('Business Understanding')
    st.write("""
             **Zomato is India's #1 food delivery app** with over 3 million restaurants and 3 billion+ orders delivered.
            With the mission **'Better food for more people'**, Zomato becomes a successful on-demand food delivery platform that helps users discover food places and get it delivered to their doorstep. 
            As the demand for food delivery increased, competition in the Indian market has intensified. **Swiggy emerged as Zomato's primary rival**. Both companies compete, not only through promotional offers but also in ensuring faster delivery time which is a critical factor in determining customer satisfaction and retention. 
    """)
    st.subheader('Business Challenges')
    st.write("""
             In general, food delivery industry has been struggling with **delivery delays**. Customers expect their food to arrive on time and any delay can lead to dissatisfaction [1](https://www.kopatech.com/blog/how-to-overcome-10-challenges-in-the-food-delivery-industry).
             All food delivery apps are competing to provide quick food delivery services [2](https://www.businesstoday.in/technology/news/story/zomato-joins-quick-food-delivery-race-with-15-minute-service-feature-spotted-on-app-459953-2025-01-08). Zomato also launched a 15-minute delivery service, but unfortunately, it was shut down after only four months as it failed to meet customer expectations [3](https://www.livemint.com/companies/news/zomato-shuts-15-minute-food-delivery-service-quick-four-months-after-launch-confirms-ceo-deepinder-goyal-in-q4-report-11746096973747.html).
             """)
    st.subheader('Problem Statement')
    st.write("""
             To sustain their position as the #1 food delivery app, Zomato should focus on accurately predicting delivery time. 
             """)
    st.subheader('Objectives')
    st.markdown("""
                This project aims to :
                
                🔎 Identify key drivers of delivery time 
                
                🛵 Predict delivery time 
                """)

# ------------- PAGE 2 : DASHBOARD -------
elif st.session_state["page"] == 'Dashboard':
    st.header('📊 Analysis Dashboard')

    # KPI Metrics
    st.subheader('Key Performance Indicators')
    col1, col2, col3, col4 = st.columns([4, 4, 3, 3])

    avg_delivery_time = round(df['Time_taken (min)'].mean(), 1)
    avg_pickup_time = round(df['pickup_time (min)'].mean(), 1)
    avg_ratings = round(df['Delivery_person_Ratings'].mean(), 2)
    total_orders = df['ID'].nunique()

    with col1:
        st.metric(label='Average Delivery Time', value=f'{avg_delivery_time} minutes')
    with col2:
        st.metric(label='Average Pick-up Time', value=f'{avg_pickup_time} minutes')
    with col3:
        st.metric(label='Average Ratings', value=f'{avg_ratings}/6.0')
    with col4:
        st.metric(label='Total Orders', value=f'{round(total_orders/1000,1)}K')

    st.markdown('---')

    st.subheader('Delivery Time vs Vehicles')
    col_vehicle, col_boxplot = st.columns([3, 5])

    with col_vehicle:
        vehicle_proportion = df['Type_of_vehicle'].value_counts().reset_index()
        vehicle_proportion.columns = ['Type_of_vehicle', 'Count']
        pie_vehicle = px.pie(vehicle_proportion, names='Type_of_vehicle', values='Count',
                             title='Distribution of Vehicles')
        st.plotly_chart(pie_vehicle, use_container_width=True)

    with col_boxplot:
        box_delivery = px.box(df, x='Type_of_vehicle', y='Time_taken (min)',
                              color='Road_traffic_density', title='Delivery Time vs Vehicle')
        box_delivery.update_xaxes(title='Vehicle Type')
        box_delivery.update_yaxes(title='Delivery Time (minutes)')
        st.plotly_chart(box_delivery, use_container_width=True)

    st.markdown('---')

    st.subheader('Delivery Person vs Delivery Time')
    col_age, col_age_time = st.columns([3, 5])

    with col_age:
        age_proportion = df['age_group'].value_counts().reset_index()
        age_proportion.columns = ['age_group', 'Count']
        pie_age = px.pie(age_proportion, names='age_group', values='Count',
                         title='Distribution of Age Group')
        st.plotly_chart(pie_age, use_container_width=True)

    with col_age_time:
        box_age = px.box(df, x='age_group', y='Time_taken (min)',
                         title='Delivery Time vs Age Group')
        box_age.update_xaxes(title='Age Group')
        box_age.update_yaxes(title='Delivery Time (minutes)')
        st.plotly_chart(box_age, use_container_width=True)

    st.markdown('---')

    st.subheader('Hour vs Road Traffic Density')
    bar_hour = px.histogram(df, x='Hour', color='Road_traffic_density')
    bar_hour.update_xaxes(title='Hour')
    bar_hour.update_yaxes(title='Count')
    st.plotly_chart(bar_hour, use_container_width=True)

    st.markdown('---')

    st.subheader('Hour vs Number of Orders')
    hour_order = df.groupby('Hour')['ID'].nunique().reset_index()
    bar_order = px.bar(hour_order, x='Hour', y='ID')
    bar_order.update_xaxes(title='Hour')
    bar_order.update_yaxes(title='Number of Orders')
    st.plotly_chart(bar_order, use_container_width=True)
    st.markdown('---')

# --------PAGE 3 : PREDICTION MODEL ----------
elif st.session_state["page"] == 'Prediction Model':
    india_boundaries = {'latitude_min': 8.4, 'latitude_max': 37.6,
                        'longitude_min': 68.7, 'longitude_max': 97.25}

    def is_within_india(lat, lon):
        return (india_boundaries['latitude_min'] <= lat <= india_boundaries['latitude_max']) and \
               (india_boundaries['longitude_min'] <= lon <= india_boundaries['longitude_max'])

    st.header('🛵 Delivery Time Prediction')
    st.markdown('Use this machine learning model to predict delivery time')

    st.subheader('Input your data')
    col_input1, col_input2 = st.columns(2)

    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    vehicle_options = {
        'Bicycle': 'bicycle',
        'Scooter': 'scooter',
        'Electric Scooter': 'electric_scooter',
        'Motorcycle': 'motorcycle'}

    with col_input1:
        delivery_age = st.number_input('Delivery Person Age', min_value=10, max_value=80, value=30)
        delivery_ratings = st.number_input('Delivery Person Ratings', min_value=1.0, max_value=6.0, value=4.6)
        order_day = st.selectbox('Order Day', list(day_mapping.keys()))
        order_day_val = day_mapping[order_day]
        order_hour = st.slider('Order Hour', 0, 23, 0)
        vehicle_type = st.selectbox('Vehicle Type', list(vehicle_options.keys()))
        vehicle_type_val = vehicle_options[vehicle_type]
        vehicle_conditions = st.number_input('Vehicle Condition (0=Poor)', min_value=0, max_value=3)
        traffic_level = st.selectbox('Traffic Level', df['Road_traffic_density'].unique())
        num_deliveries = st.slider('Number of Delivery', 0, 10, 1)

    with col_input2:
        rest_lat = st.number_input('Restaurant Latitude', value=28.61394)
        rest_lon = st.number_input('Restaurant Longitude', value=77.20902)
        del_lat = st.number_input('Delivery Latitude', value=28.61394)
        del_lon = st.number_input('Delivery Longitude', value=77.20902)
        city_type = st.selectbox('City', df['City'].unique())
        festival = st.selectbox('Festival', df['Festival'].unique())
        order_type = st.selectbox('Order Type', df['Type_of_order'].unique())
        weather_conditions = st.selectbox('Weather', df['Weather_conditions'].unique())

    distance_km = None
    if not is_within_india(rest_lat, rest_lon) or not is_within_india(del_lat, del_lon):
        st.warning('⚠️ Coordinates must be within India boundaries!')
    else:
        distance_km = geodesic((rest_lat, rest_lon), (del_lat, del_lon)).km

    input_df = None
    if distance_km is not None:
        input_df = pd.DataFrame([{
            'Delivery_person_Age': delivery_age,
            'Delivery_person_Ratings': delivery_ratings,
            'DayofWeek': order_day_val,
            'Hour': order_hour,
            'Type_of_vehicle': vehicle_type_val,
            'Vehicle_condition': vehicle_conditions,
            'Road_traffic_density': traffic_level,
            'multiple_deliveries': num_deliveries,
            'City': city_type,
            'Festival': festival,
            'Type_of_order': order_type,
            'Weather_conditions': weather_conditions,
            'distance_km': distance_km
        }])

    st.markdown('### Data Input')
    if input_df is not None:
        st.dataframe(input_df)

    if st.button('Predict Delivery Time'):
        if input_df is None:
            st.error("⚠️ Cannot predict because coordinates are invalid.")
        else:
            X = pd.get_dummies(input_df, drop_first=False)
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X.reindex(columns=feature_names, fill_value=0)
            pred_time = model.predict(X)[0]
            st.subheader('Prediction Result')
            st.metric('⏱️ Predicted Delivery Time', f"{pred_time:.1f} minutes")

    if st.button('Show Map'):
        st.session_state['show_map'] = True

    if st.session_state['show_map'] and distance_km is not None:
        m = folium.Map(location=[rest_lat, rest_lon], zoom_start=8)
        folium.Marker([rest_lat, rest_lon], tooltip='Restaurant',
                      icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([del_lat, del_lon], tooltip='Delivery Location',
                      icon=folium.Icon(color='red')).add_to(m)
        folium.PolyLine([[rest_lat, rest_lon], [del_lat, del_lon]],
                        color='blue', weight=2.5).add_to(m)
        st_folium(m, width=700, height=500)

