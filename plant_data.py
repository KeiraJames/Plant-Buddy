import json

def load_plant_data():
    with open('plant_care_instructions.json') as f:
        return json.load(f)

def get_plant(plant_name):
    plants = load_plant_data()
    return next((p for p in plants if p["Plant Name"].lower() == plant_name.lower()), None)