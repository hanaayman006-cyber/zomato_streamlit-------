import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Function to safely load files
# ----------------------------
def safe_load_joblib(file_name):
    import joblib, os
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.error(f"❌ File '{file_name}' not found. Please upload it in the repo.")
        return None

def safe_load_csv(file_name):
    import os
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        st.error(f"❌ CSV file '{file_name}' not found. Please upload it in the repo.")
        return None

# ----------------------------
# Load Data and Models
# ----------------------------
df = safe_load_csv("zomato_sample.csv")
best_model = safe_load_joblib("best_model.pkl")      # XGBoost
nn_model = safe_load_joblib("nn_model.h5")          # Neural Network
scaler = safe_load_joblib("scaler.pkl")
encoders = safe_load_joblib("encoders.pkl")
feature_cols = safe_load_joblib("feature_cols.pkl")

# ----------------------------
# Only continue if essential files loaded
# ----------------------------
if df is not None and best_model is not None and nn_model is not None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ----------------------------
    # Streamlit Sidebar
    # ----------------------------
    st.sidebar.title("Zomato Bangalore App")
    page = st.sidebar.selectbox("Choose Page", ["Analysis", "Prediction"])

    # ----------------------------
    # Analysis Page
    # ----------------------------
    if page == "Analysis":
        st.title("Zomato Bangalore - EDA Analysis")
        
        st.write("**Top 10 Cuisines:**")
        top_cuisines = df['cuisines'].value_counts().head(10)
        st.bar_chart(top_cuisines)

        st.write("**Top 10 Locations:**")
        top_locations = df['location'].value_counts().head(10)
        st.bar_chart(top_locations)

        st.write("**Distribution of Ratings:**")
        plt.figure(figsize=(6,4))
        sns.histplot(df['rate'])
        st.pyplot(plt)

        st.write("**Online Order %:**")
        st.write(df['online_order'].value_counts(normalize=True)*100)

        st.write("**Table Booking %:**")
        st.write(df['book_table'].value_counts(normalize=True)*100)

    # ----------------------------
    # Prediction Page
    # ----------------------------
    else:
        st.title("Predict Restaurant Rating Category")
        st.write("Enter details of a new restaurant:")

        # Input form
        with st.form(key="prediction_form"):
            online_order = st.selectbox("Online Order", ["Yes", "No"])
            book_table = st.selectbox("Book Table", ["Yes", "No"])
            votes = st.number_input("Votes", min_value=0, value=100)
            location = st.text_input("Location", "Koramangala")
            rest_type = st.text_input("Restaurant Type", "Cafe")
            dish_liked = st.text_input("Popular Dish", "Pasta")
            cuisines = st.text_input("Cuisines", "Italian")
            approx_cost = st.number_input("Approx Cost for Two", min_value=0, value=700)

            submit_button = st.form_submit_button(label="Predict")

        if submit_button:
            # Prepare input
            input_dict = {
                'online_order': online_order,
                'book_table': book_table,
                'votes': votes,
                'location': location,
                'rest_type': rest_type,
                'dish_liked': dish_liked,
                'cuisines': cuisines,
                'avg_cost_log': np.log1p(approx_cost)
            }

            # Encode text
            input_df = pd.DataFrame([input_dict])
            if encoders is not None:
                for col, le in encoders.items():
                    if col in input_df.columns:
                        input_df[col] = le.transform(input_df[col])
            # Add missing columns
            if feature_cols is not None:
                for col in feature_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_cols]

            # Standardize
            if scaler is not None:
                num_cols = input_df.select_dtypes(include=np.number).columns
                input_df[num_cols] = scaler.transform(input_df[num_cols])

            # Predictions
            pred_xgb = best_model.predict(input_df)[0] if best_model else "N/A"
            pred_nn = np.argmax(nn_model.predict(input_df), axis=1)[0] if nn_model else "N/A"

            st.write(f"**Prediction from XGBoost:** {pred_xgb}")
            st.write(f"**Prediction from Neural Network:** {pred_nn}")

else:
    st.warning("🚨 Essential files missing. App cannot run properly.")
