import streamlit as st
from pymongo import MongoClient
from datetime import datetime
from api_config import MONGO_URI

uri = MONGO_URI

# Create a new client and connect to the server
client = MongoClient(uri)
db = client['temp_moisture'] 
collection = db['c1']  

def get_latest_stats():
    # Fetch the latest data (no need to count documents or sort, just fetch the most recent entry)
    latest_data = collection.find_one(sort=[('timestamp', -1)])  # Sort by timestamp descending and get the most recent
    return latest_data

st.title("Plant Care Monitor")

# Button to fetch the latest data
if st.button("Give Me Stats Update"):
    data = get_latest_stats()

    if data:
        temperature = data["temperature"]
        moisture_value = data["moisture_value"]
        timestamp = data["timestamp"]

        # Convert timestamp to a readable format
        timestamp = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        st.write(f"**Temperature**: {temperature}°F")
        st.write(f"**Moisture Level**: {moisture_value}")
        st.write(f"**Last Updated**: {timestamp}")

        if moisture_value < 400:  # Example threshold for moisture
            st.warning("Your plant needs water!")
        elif moisture_value < 600:
            st.info("Your plant is doing okay!")
        else:
            st.success("Your plant is happy and hydrated!")
    else:
        st.error("No data available.")
