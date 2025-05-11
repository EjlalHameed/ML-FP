import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Global_AI_Content_Impact_Dataset.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(inplace=True)

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("AI Adoption Rate (%)", axis=1)
y = df_encoded["AI Adoption Rate (%)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

st.title("🌐 AI Adoption Rate Predictor")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted AI Adoption Rate: {round(prediction, 2)} %")

    # Save history
    history_df = input_df.copy()
    history_df["Predicted AI Adoption Rate (%)"] = round(prediction, 2)

    if os.path.exists("search_history.csv"):
        existing = pd.read_csv("search_history.csv")
        history_df = pd.concat([existing, history_df], ignore_index=True)

    history_df.to_csv("search_history.csv", index=False)

if st.button("View History"):
    if os.path.exists("search_history.csv"):
        history = pd.read_csv("search_history.csv")
        st.subheader("📚 User Search History")
        st.dataframe(history)
    else:
        st.info("No history found yet.")
