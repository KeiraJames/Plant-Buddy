import streamlit as st
st.set_page_config(page_title="Plant Buddy", page_icon="🌿", layout="wide")

from PIL import Image
import os
import json
import requests
import base64
import tempfile
from io import BytesIO
import pytz
from datetime import datetime, timedelta, timezone
import random
import re
import pandas as pd # For charts

from fuzzywuzzy import process
from pymongo import MongoClient

# --- Project Imports ---
from plant_net import PlantNetAPI
from api_config import PLANTNET_API_KEY, GEMINI_API_KEY, MONGO_URI


# --- Constants ---
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
EASTERN_TZ = pytz.timezone('US/Eastern')

# Load plant care data (ensure this file exists)
try:
    with open(os.path.join(os.path.dirname(__file__), 'plants_with_personality3_copy.json')) as f:
        SAMPLE_PLANT_CARE_DATA = json.load(f)
except FileNotFoundError:
    st.error("Error: `plants_with_personality3_copy.json` not found. Please ensure it's in the same directory as the app.")
    SAMPLE_PLANT_CARE_DATA = []
except json.JSONDecodeError:
    st.error("Error: `plants_with_personality3_copy.json` is not valid JSON.")
    SAMPLE_PLANT_CARE_DATA = []

# --- MongoDB Client (Optional) ---
mongo_client = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # 5 sec timeout
        mongo_client.admin.command('ping') # Verify connection
        db = mongo_client['temp_moisture'] # Or your specific DB name
        sensor_collection = db['c1']       # Or your specific collection name
        st.sidebar.success("MongoDB Connected", icon="🍃")
    except Exception as e:
        st.sidebar.warning(f"MongoDB connection failed: {e}. Sensor stats will be simulated.", icon="⚠️")
        mongo_client = None
        sensor_collection = None
else:
    st.sidebar.info("MongoDB URI not set. Sensor stats will be simulated.", icon="ℹ️")
    sensor_collection = None

# =======================================================
# ===== IMAGE DISPLAY HELPER FUNCTION =====
# =======================================================
def display_image_with_max_height(image_source, caption="", max_height_px=300, min_height_px=0, use_container_width=False, fit_contain=False):
    img_data_url = None
    if isinstance(image_source, str) and image_source.startswith('data:image'):
        img_data_url = image_source
    elif isinstance(image_source, bytes):
        try:
            img = Image.open(BytesIO(image_source))
            mime_type = Image.MIME.get(img.format) or f"image/{img.format.lower() if img.format else 'jpeg'}"
            b64_img = base64.b64encode(image_source).decode()
            img_data_url = f"data:{mime_type};base64,{b64_img}"
        except Exception as e: st.error(f"Error processing image bytes: {e}"); return
    elif isinstance(image_source, Image.Image):
        try:
            buffer = BytesIO()
            img_format = image_source.format or 'PNG'
            image_source.save(buffer, format=img_format)
            mime_type = Image.MIME.get(img_format) or f"image/{img_format.lower()}"
            b64_img = base64.b64encode(buffer.getvalue()).decode()
            img_data_url = f"data:{mime_type};base64,{b64_img}"
        except Exception as e: st.error(f"Error processing PIL image: {e}"); return
    else: st.error("Invalid image source type."); return

    if img_data_url:
        img_styles = [f"max-height: {max_height_px}px", "display: block", "margin-left: auto", "margin-right: auto", "border-radius: 8px"]
        if use_container_width:
            img_styles.append("width: 100%")
            img_styles.append(f"object-fit: {'contain' if fit_contain else 'cover'}") # 'cover' or 'contain'
        else:
            img_styles.append("width: auto")
        
        if min_height_px > 0: img_styles.append(f"min-height: {min_height_px}px")
        img_style_str = "; ".join(img_styles)
        
        html_string = f"""
<div style="display: flex; justify-content: center; flex-direction: column; align-items: center; margin-bottom: 10px;">
    <img src="{img_data_url}" style="{img_style_str};" alt="{caption or 'Uploaded image'}">
    {f'<p style="text-align: center; font-size: 0.9em; color: grey; margin-top: 5px;">{caption}</p>' if caption else ""}
</div>"""
        st.markdown(html_string, unsafe_allow_html=True)

# =======================================================
# ===== PLANT STATS RING DISPLAY FUNCTIONS =====
# =======================================================
MOISTURE_COLOR = "#FF2D55"; MOISTURE_TRACK_COLOR = "#591F2E"
TEMPERATURE_COLOR = "#A4E803"; TEMPERATURE_TRACK_COLOR = "#4B6A01"
FRESHNESS_COLOR = "#00C7DD"; FRESHNESS_TRACK_COLOR = "#005C67"
WHITE_COLOR = "#FFFFFF"; LIGHT_GREY_TEXT_COLOR = "#A3A3A3"; WATCH_BG_COLOR = "#000000"
MOISTURE_MAX_PERCENT = 100; TEMP_DISPLAY_MAX_F = 100; TEMP_DISPLAY_MIN_F = 50
FRESHNESS_MAX_MINUTES_AGO = 120 # Max minutes for freshness ring to show 100% fresh

def get_ring_html_css():
    return f"""<style>
    .watch-face-grid {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
    .watch-face-container {{ background-color: {WATCH_BG_COLOR}; padding: 15px; border-radius: 28px; width: 200px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: {WHITE_COLOR}; text-align: center; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    .watch-header {{ width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 5px; margin-bottom: 8px; }}
    .ring-title {{ font-size: 15px; font-weight: 600; }} .ring-timestamp {{ font-size: 13px; color: {LIGHT_GREY_TEXT_COLOR}; }}
    .ring-outer-circle {{ width: 130px; height: 130px; border-radius: 50%; position: relative; display: flex; align-items: center; justify-content: center; }}
    .ring-progress {{ width: 100%; height: 100%; border-radius: 50%; position: relative; }}
    .ring-inner-content {{ position: absolute; color: {WHITE_COLOR}; text-align: center; }}
    .ring-value {{ font-size: 36px; font-weight: 500; line-height: 1.1; }} .ring-goal-text {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; text-transform: uppercase; }}
    .progress-indicator-dot {{ width: 12px; height: 12px; background-color: {WHITE_COLOR}; border-radius: 50%; border: 2px solid {WATCH_BG_COLOR}; position: absolute; top: 4px; left: 50%; transform-origin: center calc(65px - 4px); }}
    .ring-dots {{ margin-top: 8px; font-size: 16px; }} .ring-dots .dot-dim {{ color: #444; }}
    .ring-description {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; margin-top: 12px; text-align: left; width: 90%; line-height: 1.3; }}
    .home-tab-content {{ background-color: #F0FFF0; padding: 15px; border-radius: 8px; }} /* Honeydew background */
    .health-score-heart {{ font-size: 2em; transition: color 0.5s ease; }}
    .health-good {{ color: #28a745; }}
    .health-medium {{ color: #ffc107; }}
    .health-bad {{ color: #dc3545; }}
    </style>"""

def generate_ring_html(title, value_text, goal_text, progress_percent, color, track_color, timestamp_str, description, dot_index=0):
    progress_capped = max(0, min(progress_percent, 100))
    dot_rotation = (progress_capped / 100) * 360
    dots_html = "".join([f'<span style="color:{color};">•</span> ' if i == dot_index else '<span class="dot-dim">•</span> ' for i in range(3)])
    ring_style = f"background-image: conic-gradient(from -90deg, {color} 0% {progress_capped}%, {track_color} {progress_capped}% 100%); padding: 10px;"
    dot_style = f"transform: translateX(-50%) rotate({dot_rotation}deg);"
    return f"""<div class="watch-face-container"><div class="watch-header"><span class="ring-title" style="color:{color};">{title}</span><span class="ring-timestamp">{timestamp_str}</span></div><div class="ring-outer-circle"><div class="ring-progress" style="{ring_style}"><div class="progress-indicator-dot" style="{dot_style}"></div></div><div class="ring-inner-content"><div class="ring-value">{value_text}</div><div class="ring-goal-text">{goal_text}</div></div></div><div class="ring-dots">{dots_html}</div><div class="ring-description">{description}</div></div>"""

def parse_temp_range(temp_range_str):
    if not isinstance(temp_range_str, str): return None, None
    match_f = re.search(r'(\d+)\s*-\s*(\d+)\s*°F', temp_range_str)
    if match_f: return int(match_f.group(1)), int(match_f.group(2))
    match_single_f = re.search(r'(\d+)\s*°F', temp_range_str)
    if match_single_f: val = int(match_single_f.group(1)); return val, val
    return None, None

# =======================================================
# ===== API Functions =====
# =======================================================
plantnet_api_client = PlantNetAPI(api_key=PLANTNET_API_KEY)

def identify_plant_wrapper(image_bytes, filename="uploaded_image.jpg"):
    """Wrapper for PlantNetAPI to use bytes."""
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        # Simulate for demo if API key is missing
        # st.warning("PlantNet API Key not configured. Using demo identification.", icon="🔑")
        common_names_list = ["Monstera deliciosa", "Fiddle Leaf Fig", "Snake Plant", "Aloe Vera", "Peace Lily"]
        sci_names_list = ["Monstera deliciosa", "Ficus lyrata", "Dracaena trifasciata", "Aloe barbadensis miller", "Spathiphyllum wallisii"]
        idx = random.randint(0, len(common_names_list)-1)
        return {
            'scientific_name': sci_names_list[idx],
            'common_name': common_names_list[idx],
            'confidence': random.uniform(70, 95),
            'raw_data': {"message": "Demo mode"}
        }
    return plantnet_api_client.identify_plant_from_bytes(image_bytes, filename)

def create_personality_profile(care_info):
    default = {"title": "Standard Plant", "traits": "observant", "prompt": "You are a plant. Respond factually."}
    if not isinstance(care_info, dict): return default
    p_data = care_info.get("Personality")
    if not isinstance(p_data, dict): return {"title": f"The {care_info.get('Plant Name', 'Plant')}", "traits": "resilient", "prompt": "Respond simply."}
    traits = p_data.get("Traits", ["observant"]); traits = [str(t) for t in traits if t] if isinstance(traits, list) else ["observant"]
    return {"title": p_data.get("Title", care_info.get('Plant Name', 'Plant')), "traits": ", ".join(traits) or "observant", "prompt": p_data.get("Prompt", "Respond in character.")}

def send_message_to_gemini(messages_for_api, image_bytes=None, image_type="image/jpeg"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Chat disabled: Gemini API Key missing or invalid."

    payload_contents = []
    # Handle existing messages (system prompt, history)
    for msg in messages_for_api:
        payload_contents.append(msg)

    # Add image to the last user message if provided
    if image_bytes and payload_contents and payload_contents[-1]["role"] == "user":
        last_user_message_parts = payload_contents[-1]["parts"]
        # Ensure 'parts' is a list
        if not isinstance(last_user_message_parts, list):
            last_user_message_parts = [{"text": str(last_user_message_parts)}] # Convert to list of dicts if it's just text

        img_base64 = base64.b64encode(image_bytes).decode()
        last_user_message_parts.append({
            "inline_data": {
                "mime_type": image_type,
                "data": img_base64
            }
        })
        payload_contents[-1]["parts"] = last_user_message_parts
    
    payload = {"contents": payload_contents, "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7}}
    headers = {"Content-Type": "application/json"}

    try:
        r = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if data.get('candidates') and data['candidates'][0].get('content', {}).get('parts'):
            return data['candidates'][0]['content']['parts'][0]['text']
        # Check for safety ratings or blocked responses
        if data.get('promptFeedback', {}).get('blockReason'):
            return f"Sorry, my response was blocked. Reason: {data['promptFeedback']['blockReason']}"
        return "Sorry, I received an unexpected response from the chat model."
    except requests.exceptions.Timeout:
        return "Sorry, the request to the chat model timed out."
    except requests.exceptions.RequestException as e:
        err_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_json = e.response.json()
                err_detail = err_json.get('error', {}).get('message', e.response.text)
            except json.JSONDecodeError:
                err_detail = e.response.text
        return f"Sorry, I'm having trouble connecting to the chat model. (Details: {err_detail or str(e)})"
    except Exception as e:
        return f"Oops, something unexpected went wrong while trying to chat: {str(e)}"


def chat_with_plant(care_info, conversation_history, id_result=None, image_bytes_for_chat=None, image_type_for_chat="image/jpeg"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Chat feature disabled: Gemini API Key not set."

    plant_name = "this plant"
    prompt_parts = ["CONTEXT: Short chatbot response (1-3 sentences).", "TASK: Act *exclusively* as the plant. Stay in character. NO mention of AI/model."]
    rules = ["RESPONSE RULES:", "1. First person (I, me, my).", "2. Embody personality.", "3. Concise (1-3 sentences).", "4. **Never break character or mention AI.**"]

    # This specific line in the system prompt might be less effective than modifying the user's actual turn.
    # We can keep it as a general hint or remove if the turn-specific modification is sufficient.
    # Let's keep it for now.
    if image_bytes_for_chat:
        prompt_parts.append("INFO: The user may have provided an image of me along with their current message. If their message refers to 'the image' or asks to 'look at this', please consider the visual context provided in this turn.")

    if care_info and isinstance(care_info, dict):
        p = create_personality_profile(care_info)
        plant_name = care_info.get('Plant Name', 'a plant')
        prompt_parts.extend([f"PERSONALITY: '{p['title']}' (traits: {p['traits']}). Philosophy: {p['prompt']}", "CARE NEEDS (Use ONLY these):",
                             f"- Light: {care_info.get('Light Requirements', 'N/A')}", f"- Water: {care_info.get('Watering', 'N/A')}",
                             f"- Humidity: {care_info.get('Humidity Preferences', 'N/A')}", f"- Temp: {care_info.get('Temperature Range', 'N/A')}"])
    elif id_result and isinstance(id_result, dict) and 'error' not in id_result:
        plant_name = id_result.get('common_name', id_result.get('scientific_name', 'this plant'))
        if plant_name == 'N/A' or not plant_name.strip(): plant_name = 'this plant'
        prompt_parts.extend([f"Identified as '{plant_name}'. No specific stored profile.", f"Answer generally about '{plant_name}' plants.", "If asked specifics, say you lack exact details but can offer general advice."])
    else:
        return "Sorry, not enough info to chat."
    
    sys_prompt = "\n".join(prompt_parts + rules)
    
    messages_for_api = [{"role": "user", "parts": [{"text": sys_prompt}]}, 
                        {"role": "model", "parts": [{"text": f"Understood. I am {plant_name}. Ask away!"}]}]
    
    # `conversation_history` (passed as `api_call_chat_history`) now contains the user's prompt,
    # potentially modified to mention the image.
    for entry in [m for m in conversation_history if isinstance(m, dict) and "role" in m and "content" in m]: 
        api_role = "model" if entry["role"] in ["assistant", "model"] else "user"
        content_text = str(entry["content"]) if entry["content"] is not None else ""
        messages_for_api.append({"role": api_role, "parts": [{"text": content_text}]})
        
    # The image will be added to the last user message by send_message_to_gemini if image_bytes_for_chat is provided
    return send_message_to_gemini(messages_for_api, image_bytes=image_bytes_for_chat, image_type=image_type_for_chat)

# --- MongoDB Sensor Data Helper ---
def get_latest_sensor_stats():
    if sensor_collection is not None:
        try:
            latest_data = sensor_collection.find_one(sort=[('timestamp', -1)])
            if latest_data:
                # Ensure keys exist, provide defaults if not
                return {
                    "temperature": latest_data.get("temperature"),
                    "moisture_value": latest_data.get("moisture_value"),
                    "timestamp": latest_data.get("timestamp") # This should be a Unix timestamp
                }
        except Exception as e:
            st.toast(f"Error fetching from MongoDB: {e}", icon=" M") # Minor error, don't block
            return None
    return None

# =======================================================
# --- Helper Functions ---
# =======================================================
@st.cache_data(show_spinner="Loading plant database...")
def load_plant_care_data(): return SAMPLE_PLANT_CARE_DATA

def find_care_instructions(plant_name_id, care_data_list, threshold=75): 
    if not care_data_list: return None
    sci_name, common_name_str = (None, None)
    if isinstance(plant_name_id, dict): 
        sci_name, common_name_str = plant_name_id.get('scientific_name'), plant_name_id.get('common_name')
    elif isinstance(plant_name_id, str): sci_name = plant_name_id
    
    s_sci = sci_name.lower().strip() if sci_name else None
    s_com = common_name_str.lower().strip() if common_name_str else None

    for p_entry in care_data_list: 
        db_sci_name = p_entry.get('Scientific Name','').lower().strip()
        db_plant_name = p_entry.get('Plant Name','').lower().strip() # Often same as common
        
        if s_sci and (s_sci == db_sci_name or s_sci == db_plant_name): return p_entry
        if s_com and (s_com == db_plant_name): return p_entry
        
        db_common_names_list = p_entry.get('Common Names',[])
        if isinstance(db_common_names_list, list):
            if s_com and s_com in [c.lower().strip() for c in db_common_names_list if isinstance(c, str)]:
                return p_entry
        elif isinstance(db_common_names_list, str): # If it's a single string
             if s_com and s_com == db_common_names_list.lower().strip():
                 return p_entry

    # Fuzzy matching as fallback
    all_names_map = {}
    for p_obj in care_data_list:
        names_to_check = [p_obj.get('Scientific Name',''), p_obj.get('Plant Name','')]
        c_names = p_obj.get('Common Names',[])
        if isinstance(c_names, list): names_to_check.extend(c_names)
        elif isinstance(c_names, str): names_to_check.append(c_names)
        
        for name_str in names_to_check:
            if isinstance(name_str, str) and name_str.strip():
                all_names_map[name_str.lower().strip()] = p_obj
    
    if not all_names_map: return None
    
    best_match_plant, high_score = None, 0 
    for search_term in [s_sci, s_com]:
        if search_term: 
            match_result = process.extractOne(search_term, all_names_map.keys())
            if match_result and match_result[1] >= threshold and match_result[1] > high_score: 
                high_score = match_result[1]
                best_match_plant = all_names_map.get(match_result[0])
    return best_match_plant

def display_identification_result(result):
    st.subheader("🔍 Identification Results")
    if not result or 'error' in result: 
        st.error(f"Identification failed: {result.get('error', 'Unknown error') if result else 'No result available.'}")
        return
    conf = result.get('confidence', 0); color = "#28a745" if conf > 75 else ("#ffc107" if conf > 50 else "#dc3545")
    st.markdown(f"- **Scientific Name:** `{result.get('scientific_name', 'N/A')}`\n- **Common Name:** `{result.get('common_name', 'N/A')}`\n- **Confidence:** <strong style='color:{color};'>{conf:.1f}%</strong>", unsafe_allow_html=True)

def display_care_instructions(care_info, header_level=3):
    if not care_info: st.warning("Care info missing."); return
    name = care_info.get('Plant Name', 'This Plant')
    st.markdown(f"<h{header_level}>🌱 {name} Care Guide</h{header_level}>", unsafe_allow_html=True)
    with st.expander("📋 Care Summary", expanded=True):
        c1,c2=st.columns(2); details=[("☀️ Light",'Light Requirements'),("💧 Water",'Watering'),("🌡️ Temp",'Temperature Range'),("💦 Humidity",'Humidity Preferences'),("🍃 Feeding",'Feeding Schedule'),("⚠️ Toxicity",'Toxicity')]
        for i,(lbl,key) in enumerate(details): 
            col = c1 if i < len(details)/2 else c2
            col.markdown(f"**{lbl}**")
            col.caption(f"{care_info.get(key,'N/A')}")
    if care_info.get('Additional Care','').strip():
        with st.expander("✨ Pro Tips"): st.markdown(care_info['Additional Care'])

def find_similar_plant_matches(id_r, care_data_list, limit=3, score_thresh=60):
    if not id_r or 'error' in id_r or not care_data_list: return []
    
    all_names_map = {} # Map lowercase name to plant object
    for p_obj in care_data_list:
        names_to_check = [p_obj.get('Scientific Name',''), p_obj.get('Plant Name','')]
        c_names = p_obj.get('Common Names',[])
        if isinstance(c_names, list): names_to_check.extend(c_names)
        elif isinstance(c_names, str): names_to_check.append(c_names)
        
        for name_str in names_to_check:
            if isinstance(name_str, str) and name_str.strip():
                 # Prefer Plant Name or Scientific Name as key if available for uniqueness
                key_name = p_obj.get('Plant Name') or p_obj.get('Scientific Name') or name_str
                all_names_map[name_str.lower().strip()] = p_obj 

    if not all_names_map: return []
    
    search_terms = [term.lower().strip() for term in [id_r.get('scientific_name',''), id_r.get('common_name','')] if term]
    
    potential_matches = {} # Store best score for each unique plant object
    for term in search_terms:
        if term:
            fuzz_results = process.extract(term, all_names_map.keys(), limit=limit*3) # Get more initial results
            for match_name_key, score in fuzz_results:
                if score >= score_thresh:
                    matched_plant_obj = all_names_map[match_name_key]
                    # Use a unique identifier for the plant object to store its best score
                    plant_id = id(matched_plant_obj) 
                    if plant_id not in potential_matches or score > potential_matches[plant_id]['score']:
                        potential_matches[plant_id] = {'plant': matched_plant_obj, 'score': score}
    
    # Sort by score and get top unique plants
    sorted_matches = sorted(potential_matches.values(), key=lambda x: x['score'], reverse=True)
    
    final_suggestions = []
    seen_plant_names_for_suggestion = set() # To avoid duplicate suggestions if same plant name appears
    original_id_name = id_r.get('common_name', id_r.get('scientific_name', '')).lower()

    for match_data in sorted_matches:
        p_info = match_data['plant']
        p_sugg_name = p_info.get('Plant Name', p_info.get('Scientific Name',''))
        
        # Avoid suggesting the exact same plant as already identified
        if p_sugg_name.lower() == original_id_name:
            continue

        if p_sugg_name not in seen_plant_names_for_suggestion:
            final_suggestions.append(p_info)
            seen_plant_names_for_suggestion.add(p_sugg_name)
            if len(final_suggestions) >= limit: break
                
    return final_suggestions


def display_suggestion_buttons(suggestions):
     if not suggestions: return
     st.info("🌿 Perhaps one of these is a closer match from our database?")
     # Ensure suggestions is a list before trying to use st.columns
     if not isinstance(suggestions, list) or not suggestions:
         return

     num_cols = len(suggestions)
     if num_cols == 0:
         return
     
     cols = st.columns(num_cols)

     for i, p_info in enumerate(suggestions):
         p_name = p_info.get('Plant Name', p_info.get('Scientific Name', f'Sugg {i+1}'))
         safe_n = "".join(c if c.isalnum() else "_" for c in p_name) # Make key safe
         tip = f"Select {p_name}" + (f" (Sci: {p_info.get('Scientific Name')})" if p_info.get('Scientific Name','') != p_name else "")
         
         # Use a unique and consistent key for each button
         button_key = f"sugg_btn_{safe_n}_{i}"

         if cols[i].button(p_name, key=button_key, help=tip, use_container_width=True):
            # Create a new ID result based on the suggestion
            new_id_result = {
                'scientific_name': p_info.get('Scientific Name','N/A'), 
                'common_name': p_name, 
                'confidence': 100.0, # Assume 100% confidence for user-selected suggestion
                'raw_data': {"message": "Selected from database suggestion", "original_suggestion_data": p_info}
            }
            
            # Update session state to reflect this new "primary" identification
            st.session_state.plant_id_result = new_id_result
            st.session_state.plant_care_info = p_info # Directly use the care info from suggestion
            st.session_state.plant_id_result_for_care_check = new_id_result # Mark as "checked" to prevent re-fetch
            
            st.session_state.suggestions = [] # Clear suggestions as one has been chosen
            st.session_state.chat_history = [] # Reset chat for the new plant context
            st.session_state.current_chatbot_plant_name = None # Will be set by chat interface
            
            # Keep the uploaded image for the chat context with the newly selected plant
            # st.session_state.active_chat_image_bytes remains from original upload
            # st.session_state.active_chat_image_type remains
            st.session_state.send_image_with_next_message = (st.session_state.get('active_chat_image_bytes') is not None)

            st.session_state.suggestion_just_selected = True # Flag that a suggestion was picked
            st.session_state.new_user_message_to_process = False # Reset any pending chat processing
            st.session_state.saving_mode = False # Ensure not in saving mode
            
            st.rerun() # Rerun to reflect the new state

def display_chat_interface(current_plant_care_info=None, plant_id_result=None, uploaded_image_bytes=None, uploaded_image_type=None):
    chatbot_name = "this plant"
    can_chat = False

    if current_plant_care_info and isinstance(current_plant_care_info, dict):
        chatbot_name = current_plant_care_info.get("Plant Name", "this plant")
        can_chat = True
    elif plant_id_result and isinstance(plant_id_result, dict) and 'error' not in plant_id_result:
        name_id = plant_id_result.get('common_name', plant_id_result.get('scientific_name'))
        if name_id and name_id != 'N/A' and name_id.strip(): chatbot_name = name_id
        can_chat = True

    if can_chat and (not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here"):
        st.warning("Chat feature requires a Gemini API key.", icon="🔑")
        return
    if not can_chat:
        st.warning("Cannot initialize chat without valid plant identification or care info.", icon="ℹ️")
        return

    st.subheader(f"💬 Chat with {chatbot_name}")
    st.markdown("""<style>.message-container{padding:1px 5px}.user-message{background:#0b81fe;color:white;border-radius:18px 18px 0 18px;padding:8px 14px;margin:3px 0 3px auto;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.1);animation:fadeIn .3s ease-out}.bot-message{background:#e5e5ea;color:#000;border-radius:18px 18px 18px 0;padding:8px 14px;margin:3px auto 3px 0;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.05);animation:fadeIn .3s ease-out}.message-meta{font-size:.7rem;color:#777;margin-top:3px}.bot-message .message-meta{text-align:left;color:#555}.user-message .message-meta{text-align:right}@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}.stChatInputContainer{position:sticky;bottom:0;background:var(--streamlit-secondary-background-color);padding-top:10px;z-index:99}</style>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    if st.session_state.get("current_chatbot_plant_name") != chatbot_name and not st.session_state.get("viewing_saved_details"):
        st.session_state.chat_history = []
        st.session_state.current_chatbot_plant_name = chatbot_name
        # If navigating to a new plant ID, set its image as the active one for the chat
        st.session_state.active_chat_image_bytes = uploaded_image_bytes 
        st.session_state.active_chat_image_type = uploaded_image_type
    elif st.session_state.get("viewing_saved_details"):
         saved_data = st.session_state.saved_photos.get(st.session_state.viewing_saved_details)
         if saved_data and st.session_state.get("current_chatbot_plant_name") != saved_data.get("nickname"):
            st.session_state.chat_history = saved_data.get('chat_log', [])
            st.session_state.current_chatbot_plant_name = saved_data.get("nickname")
         # For saved plants, we don't automatically assume an image context for ongoing chat unless explicitly re-triggered
         st.session_state.active_chat_image_bytes = None
         st.session_state.active_chat_image_type = None


    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_history:
            role, content, time = msg.get("role"), msg.get("content", ""), msg.get("time", "")
            if role == "user": st.markdown(f'<div class="message-container"><div class="user-message">{content}<div class="message-meta">You • {time}</div></div></div>', unsafe_allow_html=True)
            elif role in ["assistant", "model"]: st.markdown(f'<div class="message-container"><div class="bot-message">🌿 {content}<div class="message-meta">{chatbot_name} • {time}</div></div></div>', unsafe_allow_html=True)

    chat_input_key = f"chat_input_{''.join(c if c.isalnum() else '_' for c in chatbot_name)}"
    
    if prompt := st.chat_input(f"Ask {chatbot_name}...", key=chat_input_key):
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": prompt, "time": timestamp})
        st.session_state.new_user_message_to_process = True
        # Determine if the active image should be sent with this user message
        st.session_state.send_image_with_next_message = (st.session_state.get('active_chat_image_bytes') is not None)
        st.rerun() 

    if st.session_state.get("new_user_message_to_process", False):
        st.session_state.new_user_message_to_process = False 
        
        image_to_send_this_turn = None
        image_type_this_turn = None
        
        # Prepare a copy of chat history for this API call
        # This allows modification for adding image context without altering the displayed history yet
        api_call_chat_history = [msg.copy() for msg in st.session_state.chat_history]

        if st.session_state.get("send_image_with_next_message"):
            image_to_send_this_turn = st.session_state.get('active_chat_image_bytes')
            image_type_this_turn = st.session_state.get('active_chat_image_type')
            
            # Modify the text of the last user message (the current prompt) to cue Gemini
            if api_call_chat_history and api_call_chat_history[-1]["role"] == "user":
                original_prompt_text = api_call_chat_history[-1]["content"]
                api_call_chat_history[-1]["content"] = f"{original_prompt_text} (I've also provided an image of myself. Please refer to it if relevant to my question.)"
            
            # Consume the image for this turn
            st.session_state.send_image_with_next_message = False 
            st.session_state.active_chat_image_bytes = None 
            st.session_state.active_chat_image_type = None

        with st.spinner(f"{chatbot_name} is thinking..."):
            bot_response = chat_with_plant(
                current_plant_care_info, 
                api_call_chat_history, # Pass the potentially modified history
                plant_id_result,
                image_bytes_for_chat=image_to_send_this_turn,
                image_type_for_chat=image_type_this_turn
            )
        
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        # Append the bot's response to the original session state chat_history for display
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "time": timestamp})
        
        if st.session_state.get("viewing_saved_details") and st.session_state.viewing_saved_details in st.session_state.saved_photos:
            st.session_state.saved_photos[st.session_state.viewing_saved_details]['chat_log'] = st.session_state.chat_history
        st.rerun()

# --- Health Score Calculation ---
def calculate_health_score_component(value, ideal_min, ideal_max, lower_is_better=False):
    """Calculates a score (0-100) for a single component."""
    if value is None or ideal_min is None or ideal_max is None:
        return 50 # Neutral score if data is missing

    # Normalize value to be within a 0-100 range for easier scoring
    # This part may need adjustment based on actual sensor value ranges
    
    if lower_is_better: # e.g., for 'minutes ago' for freshness
        if value <= ideal_min: return 100
        if value >= ideal_max: return 0
        return 100 - ((value - ideal_min) / (ideal_max - ideal_min) * 100)
    else: # e.g., for moisture, temperature
        if ideal_min <= value <= ideal_max: return 100
        # Penalize more for being too low than too high, or vice versa, if needed
        # Simple linear penalty for now
        if value < ideal_min:
            # How far below min? Max deviation is ideal_min.
            # If ideal_min is 0, this needs care. Assume ideal_min is not far from 0.
            range_below = ideal_min - (ideal_min * 0.5) # Allow some tolerance, score drops if below 50% of min
            if range_below <= 0: range_below = ideal_min / 2 if ideal_min > 0 else 50 # Fallback
            penalty = ((ideal_min - value) / range_below) * 100
            return max(0, 100 - penalty)
        if value > ideal_max:
            range_above = (ideal_max * 1.5) - ideal_max # Allow up to 50% above max
            if range_above <= 0: range_above = ideal_max / 2 if ideal_max > 0 else 50 # Fallback
            penalty = ((value - ideal_max) / range_above) * 100
            return max(0, 100 - penalty)
    return 50 # Should not be reached if logic is correct

def calculate_overall_health(p_data_stats):
    """Calculates an overall health score for a plant."""
    if not p_data_stats: return 0, "No data"

    scores = []
    
    # 1. Moisture
    moisture_val = p_data_stats.get("moisture_level", 50) # Default to 50% if not present
    # Ideal moisture: assume 40-80% for generic plants for now
    # This could be refined if care_info has specific soil moisture preferences
    moisture_score = calculate_health_score_component(moisture_val, 40, 80)
    scores.append(moisture_score)

    # 2. Temperature
    temp_val = p_data_stats.get("temperature_value") # Actual temp value
    ideal_temp_min, ideal_temp_max = None, None
    if p_data_stats.get("care_info") and p_data_stats["care_info"].get("Temperature Range"):
        ideal_temp_min, ideal_temp_max = parse_temp_range(p_data_stats["care_info"]["Temperature Range"])
    
    if temp_val is not None and ideal_temp_min is not None and ideal_temp_max is not None:
        temp_score = calculate_health_score_component(temp_val, ideal_temp_min, ideal_temp_max)
    else: # Fallback if ideal range or current temp is unknown
        temp_score = 50 if temp_val is None else calculate_health_score_component(temp_val, 60, 80) # Assume 60-80F ideal
    scores.append(temp_score)

    # 3. Freshness of Data (Last Check)
    last_check_ts = p_data_stats.get("last_check_timestamp")
    if last_check_ts:
        # If it's a datetime object already (from simulation)
        if isinstance(last_check_ts, datetime):
            # Ensure it's offset-aware for comparison with offset-aware datetime.now()
            if last_check_ts.tzinfo is None:
                last_check_ts = EASTERN_TZ.localize(last_check_ts) # Assuming simulated times are Eastern
            mins_ago = (datetime.now(EASTERN_TZ) - last_check_ts).total_seconds() / 60
        # If it's a Unix timestamp (from MongoDB)
        elif isinstance(last_check_ts, (int, float)):
            last_check_dt = datetime.fromtimestamp(last_check_ts, tz=timezone.utc).astimezone(EASTERN_TZ)
            mins_ago = (datetime.now(EASTERN_TZ) - last_check_dt).total_seconds() / 60
        else: # Fallback if timestamp format is unexpected
            mins_ago = FRESHNESS_MAX_MINUTES_AGO + 1 # Treat as old data

        freshness_score = calculate_health_score_component(mins_ago, 0, FRESHNESS_MAX_MINUTES_AGO, lower_is_better=True)
    else:
        freshness_score = 20 # Low score if no timestamp
    scores.append(freshness_score)

    overall_score = sum(scores) / len(scores) if scores else 0
    
    status_text = "Unknown"
    if overall_score >= 80: status_text = "Excellent"
    elif overall_score >= 60: status_text = "Good"
    elif overall_score >= 40: status_text = "Fair"
    else: status_text = "Needs Attention"
        
    return round(overall_score), status_text

def get_health_score_emoji_html(score):
    heart_class = "health-bad"
    heart_symbol = "💔" # Broken heart for bad
    if score >= 80: 
        heart_class = "health-good"; heart_symbol = "❤️" # Full heart for good
    elif score >= 50: 
        heart_class = "health-medium"; heart_symbol = "💛" # Yellow heart for medium
    
    # Simple pulse for good health
    animation_style = ""
    if score >=80:
        animation_style = "animation: pulse_green 1.5s infinite;"

    return f'<span class="health-score-heart {heart_class}" style="{animation_style}">{heart_symbol}</span> Overall Health: {score:.0f}%'


# --- Initialize Session State ---
def initialize_session_state():
    defaults = {
        "plant_id_result": None, "plant_care_info": None, "chat_history": [],
        "current_chatbot_plant_name": None, "suggestions": None,
        "uploaded_file_bytes": None, "uploaded_file_type": None,
        "saving_mode": False,
        "viewing_saved_details": None, "plant_id_result_for_care_check": None,
        "suggestion_just_selected": False, "viewing_plant_stats_nick": None, # Changed from viewing_plant_stats
        "current_nav_choice": "🏠 Home",
        "new_user_message_to_process": False,
        "saved_photos": {},
        "active_chat_image_bytes": None, # Image to be used in current chat context
        "active_chat_image_type": None,
        "send_image_with_next_message": False,
        "welcome_response_generated": False,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # Add example plants if none are saved
    if not st.session_state.saved_photos:
        try:
            # Example 1: Monstera
            with open(os.path.join(os.path.dirname(__file__), "example_plant_1.jpg"), "rb") as f1:
                img1_bytes = f1.read()
            img1_b64 = base64.b64encode(img1_bytes).decode()
            care_info1 = find_care_instructions("Monstera deliciosa", SAMPLE_PLANT_CARE_DATA)
            
            st.session_state.saved_photos["My Monstera"] = {
                "nickname": "My Monstera",
                "image": f"data:image/jpg;base64,{img1_b64}",
                "id_result": {'scientific_name': 'Monstera deliciosa', 'common_name': 'Monstera', 'confidence': 95.0, 'raw_data':{}},
                "care_info": care_info1 if care_info1 else SAMPLE_PLANT_CARE_DATA[0] if SAMPLE_PLANT_CARE_DATA else {}, # Fallback
                "chat_log": [{"role": "assistant", "content": "Hello! I'm your example Monstera.", "time": ""}],
                "moisture_level": random.randint(40, 70),
                "temperature_value": random.uniform(68.0, 75.0),
                "last_check_timestamp": datetime.now(EASTERN_TZ) - timedelta(hours=random.randint(1,5)),
                "health_history": []
            }

            # Example 2: Snake Plant
            with open(os.path.join(os.path.dirname(__file__), "example_plant_2.jpg"), "rb") as f2:
                img2_bytes = f2.read()
            img2_b64 = base64.b64encode(img2_bytes).decode()
            care_info2 = find_care_instructions("Snake Plant", SAMPLE_PLANT_CARE_DATA)
            st.session_state.saved_photos["Sneaky Snake Plant"] = {
                "nickname": "Sneaky Snake Plant",
                "image": f"data:image/jpg;base64,{img2_b64}",
                "id_result": {'scientific_name': 'Dracaena trifasciata', 'common_name': 'Snake Plant', 'confidence': 92.0, 'raw_data':{}},
                "care_info": care_info2 if care_info2 else SAMPLE_PLANT_CARE_DATA[1] if len(SAMPLE_PLANT_CARE_DATA) > 1 else {},
                "chat_log": [],
                "moisture_level": random.randint(30, 60),
                "temperature_value": random.uniform(70.0, 78.0),
                "last_check_timestamp": datetime.now(EASTERN_TZ) - timedelta(minutes=random.randint(30,180)),
                "health_history": []
            }
        except FileNotFoundError:
            st.toast("Example plant images not found. Skipping example plant setup.", icon="⚠️")
        except Exception as e:
            st.toast(f"Error loading example plants: {e}", icon="🔥")


# --- Main App Logic ---
def main():
    st.markdown(f'<link rel="manifest" href="manifest.json">', unsafe_allow_html=True)
    initialize_session_state()
    st.markdown(get_ring_html_css(), unsafe_allow_html=True)

    st.sidebar.title("📚 Plant Buddy")
    nav_options = ["🏠 Home", "🆔 Identify New Plant", "🪴 My Saved Plants", "📊 Plant Stats"]
    
    try:
        nav_idx = nav_options.index(st.session_state.current_nav_choice)
    except ValueError:
        nav_idx = 0 
        st.session_state.current_nav_choice = "🏠 Home"

    nav_choice = st.sidebar.radio("Navigation", nav_options, key="main_nav", index=nav_idx, label_visibility="collapsed")

    if nav_choice != st.session_state.current_nav_choice:
        st.session_state.current_nav_choice = nav_choice
        if nav_choice != "🪴 My Saved Plants": st.session_state.viewing_saved_details = None
        if nav_choice != "📊 Plant Stats": st.session_state.viewing_plant_stats_nick = None
        
        # More targeted reset only if navigating TO "Identify New Plant" AND no file is active
        if nav_choice == "🆔 Identify New Plant" and not st.session_state.get("uploaded_file_bytes"):
            clear_identification_data_soft() # Use a soft clear
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(f"PlantNet: {'OK' if PLANTNET_API_KEY and PLANTNET_API_KEY != 'your_plantnet_api_key_here' else 'Not Set'}" )
    st.sidebar.caption(f"Gemini: {'OK' if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here' else 'Not Set'}")
    st.sidebar.caption(f"MongoDB: {'Connected' if mongo_client else 'Not Set/Failed'}")
    
    care_data = load_plant_care_data()

    # ==================================== HOME PAGE ====================================
    if st.session_state.current_nav_choice == "🏠 Home":
        # ... (Home page logic remains largely the same) ...
        st.markdown("<div class='home-tab-content'>", unsafe_allow_html=True)
        st.header("🌿 Welcome to Plant Buddy!")
        
        default_welcome_msg = "Welcome to Plant Buddy! Your companion for identifying plants, getting care tips, chatting with your leafy friends, and tracking their health. Happy gardening!"
        if not st.session_state.get("welcome_response_generated"):
            if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
                with st.spinner("Plant Buddy is waking up..."):
                    welcome_payload = [{"role":"user","parts":[{"text":"System: You are Plant Buddy, a friendly and enthusiastic app assistant. Generate a short, cheerful welcome message (2-3 sentences) for users opening the app. Briefly mention key features like plant identification, care advice, plant chat, and health statistics."}]}, {"role":"model","parts":[{"text":"Okay, I'll craft a great welcome!"}]}]
                    st.session_state.welcome_response = send_message_to_gemini(welcome_payload)
            else:
                st.session_state.welcome_response = default_welcome_msg
            st.session_state.welcome_response_generated = True
        
        current_welcome = st.session_state.get("welcome_response", default_welcome_msg)
        if "Sorry" in current_welcome or "disabled" in current_welcome : 
             current_welcome = default_welcome_msg

        st.markdown(f"""<div style="background-color: #e6ffed; padding:20px; border-radius:10px; border-left:5px solid #4CAF50; margin-bottom:20px;"><h3 style="color:#2E7D32;">🌱 Hello Plant Lover!</h3><p style="font-size:1.1em; color:#333333;">{current_welcome}</p></div>""", unsafe_allow_html=True)
        
        st.subheader("🔍 What You Can Do")
        hc1,hc2=st.columns(2)
        with hc1: 
            st.markdown("Spot an unknown plant? Let's identify it!")
            if st.button("📸 Identify My Plant!",use_container_width=True,type="primary"): 
                st.session_state.current_nav_choice="🆔 Identify New Plant"
                st.rerun()
        with hc2: 
            st.markdown("Check on your saved plant family.")
            if st.button("💚 Go to My Plants",use_container_width=True): 
                st.session_state.current_nav_choice="🪴 My Saved Plants"
                st.rerun()

        if st.session_state.saved_photos:
            st.divider(); st.subheader("🪴 Your Recently Saved Plants") 
            recent_keys = list(st.session_state.saved_photos.keys())
            display_keys = list(reversed(recent_keys))[:3]
            if display_keys: 
                cols_home = st.columns(len(display_keys))
                for i, nick in enumerate(display_keys): 
                    p_data = st.session_state.saved_photos[nick]
                    with cols_home[i]:
                        with st.container(border=True): 
                            if p_data.get("image"): 
                                display_image_with_max_height(p_data["image"], caption=nick, max_height_px=200, use_container_width=True, fit_contain=True)
                            else: st.markdown(f"**{nick}**")
                            id_res = p_data.get("id_result", {})
                            com_n = id_res.get('common_name', 'N/A')
                            if com_n and com_n != 'N/A' and com_n.lower() != nick.lower(): st.caption(f"({com_n})")
                            overall_score, health_status = calculate_overall_health(p_data)
                            st.markdown(get_health_score_emoji_html(overall_score), unsafe_allow_html=True)
                            if st.button("View Details", key=f"home_v_{nick.replace(' ','_')}", use_container_width=True):
                                st.session_state.viewing_saved_details = nick
                                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


    # ==================================== IDENTIFY NEW PLANT (Streamlined) ====================================
    elif st.session_state.current_nav_choice == "🆔 Identify New Plant":
        st.header("🔎 Identify a New Plant")

        # File uploader always visible
        up_file = st.file_uploader("Upload a clear photo of your plant:", type=["jpg","jpeg","png"], key="uploader_id_page_streamlined")

        if up_file:
            # If a new file is uploaded, or different from current
            new_file_bytes = up_file.getvalue()
            if not st.session_state.uploaded_file_bytes or st.session_state.uploaded_file_bytes != new_file_bytes:
                # This is a new upload, reset the identification process for this new image
                clear_identification_data_full() # Full clear for new image
                st.session_state.uploaded_file_bytes = new_file_bytes
                st.session_state.uploaded_file_type = up_file.type
                st.session_state.active_chat_image_bytes = new_file_bytes # For chat context
                st.session_state.active_chat_image_type = up_file.type
                st.session_state.needs_identification = True # Flag to trigger identification
                st.rerun() # Rerun to process the new upload state
        
        # --- Displaying Image and Further Steps ---
        if st.session_state.uploaded_file_bytes:
            display_image_with_max_height(st.session_state.uploaded_file_bytes, "Your Plant", 400)
            
            if st.button("🗑️ Clear and Start Over", key="clear_id_flow_button", use_container_width=True):
                clear_identification_data_full()
                st.rerun()
            st.divider()

            # Step 1: Perform Identification if needed
            if st.session_state.get("needs_identification") and not st.session_state.plant_id_result:
                with st.spinner("Identifying your plant... 🌱"):
                    st.session_state.plant_id_result = identify_plant_wrapper(
                        st.session_state.uploaded_file_bytes, 
                        "uploaded_plant_image.jpg" # Generic filename
                    )
                st.session_state.needs_identification = False # Mark identification as done for this upload
                # After ID, we need to check for care info or suggestions
                st.session_state.needs_care_check = True
                st.rerun() # Rerun to display ID results and proceed

            # Step 2: Display ID and Fetch/Display Care/Suggestions
            if st.session_state.plant_id_result:
                current_id_res = st.session_state.plant_id_result
                display_identification_result(current_id_res)

                if 'error' not in current_id_res:
                    # Fetch care info or suggestions if needed
                    if st.session_state.get("needs_care_check"):
                        st.session_state.plant_care_info = find_care_instructions(current_id_res, care_data)
                        if not st.session_state.plant_care_info: # If no direct care info
                            st.session_state.suggestions = find_similar_plant_matches(current_id_res, care_data)
                        else: # Care info found, no need for suggestions initially
                            st.session_state.suggestions = []
                        st.session_state.plant_id_result_for_care_check = current_id_res
                        st.session_state.needs_care_check = False # Mark as checked
                        st.session_state.chat_history = [] # Reset chat for this ID context
                        st.session_state.current_chatbot_plant_name = None
                        st.rerun() # Rerun to display care/suggestions

                    # Display Care Info OR Suggestions
                    care_to_display = st.session_state.get('plant_care_info')
                    if care_to_display:
                        display_care_instructions(care_to_display, header_level=4)
                        if st.button("💾 Save This Plant Profile", key="save_profile_with_care_id", type="primary", use_container_width=True):
                            st.session_state.saving_mode = True; st.rerun()
                    elif st.session_state.get('suggestions'): # Check if suggestions list exists and is not empty
                        st.warning("No exact care guide found. Check these suggestions or save the current ID.")
                        display_suggestion_buttons(st.session_state.suggestions) # This function handles its own rerun on click
                        if st.button("💾 Save Current ID Only", key="save_id_only_no_care_id", use_container_width=True):
                             st.session_state.saving_mode = True; st.rerun()
                    else: # No care, no suggestions (or suggestions already processed and cleared)
                        st.warning("No specific care instructions found, and no close matches in our database.")
                        if st.button("💾 Save Current ID Only", key="save_id_only_no_care_no_sugg_id", use_container_width=True):
                             st.session_state.saving_mode = True; st.rerun()
                    
                    st.divider()
                    # Chat Interface - always available if ID is successful
                    display_chat_interface(
                        current_plant_care_info=st.session_state.get('plant_care_info'), 
                        plant_id_result=current_id_res,
                        uploaded_image_bytes=st.session_state.get("active_chat_image_bytes"), # Pass active image for initial chat
                        uploaded_image_type=st.session_state.get("active_chat_image_type")
                    )
                # else: Error in ID result already handled by display_identification_result

            # --- Saving Mode ---
            if st.session_state.get("saving_mode") and st.session_state.plant_id_result and 'error' not in st.session_state.plant_id_result:
                st.subheader("💾 Save Plant Profile")
                # Nickname form
                default_nick = st.session_state.plant_id_result.get('common_name') or st.session_state.plant_id_result.get('scientific_name', 'My Plant')
                with st.form("save_form_streamlined_id"):
                    plant_nickname = st.text_input("Plant Nickname:", value=default_nick, key="nickname_save_id_stream")
                    submitted_save = st.form_submit_button("✅ Confirm & Save to My Plants")

                    if submitted_save:
                        if not plant_nickname.strip():
                            st.warning("Nickname is required.")
                        elif plant_nickname in st.session_state.saved_photos:
                            st.warning(f"Nickname '{plant_nickname}' already exists.")
                        else:
                            img_b64 = base64.b64encode(st.session_state.uploaded_file_bytes).decode()
                            st.session_state.saved_photos[plant_nickname] = {
                                "nickname": plant_nickname,
                                "image": f"data:{st.session_state.uploaded_file_type};base64,{img_b64}",
                                "id_result": st.session_state.plant_id_result,
                                "care_info": st.session_state.plant_care_info,
                                "chat_log": st.session_state.get("chat_history",[]),
                                "moisture_level": random.randint(30,90),
                                "temperature_value": random.uniform(65.0, 78.0),
                                "last_check_timestamp": datetime.now(EASTERN_TZ) - timedelta(hours=random.randint(1,3)),
                                "health_history": [(datetime.now(EASTERN_TZ).isoformat(), random.randint(60,90))]
                            }
                            st.success(f"'{plant_nickname}' saved! View in 'My Saved Plants'."); st.balloons()
                            clear_identification_data_full() # Clear current ID flow
                            st.session_state.current_nav_choice = "🪴 My Saved Plants"
                            st.session_state.viewing_saved_details = plant_nickname
                            st.rerun()
                if st.button("❌ Cancel Save", key="cancel_save_id_stream"):
                    st.session_state.saving_mode = False; st.rerun()
        else: # No file uploaded yet in this tab
            st.info("Upload an image above to identify a plant and get care tips!")


    # ==================================== MY SAVED PLANTS ====================================
    elif st.session_state.current_nav_choice == "🪴 My Saved Plants":
        # ... (My Saved Plants logic remains largely the same, it's already quite streamlined) ...
        st.header("🪴 My Saved Plant Profiles")
        nick_to_view = st.session_state.get("viewing_saved_details")

        if not st.session_state.saved_photos:
            st.info("You haven't saved any plants yet. Go to 'Identify New Plant' to start your collection!")
        elif nick_to_view and nick_to_view in st.session_state.saved_photos:
            entry = st.session_state.saved_photos[nick_to_view]
            
            if st.button("← Back to All Saved Plants",key="back_to_all_saved_plants_gallery"):
                st.session_state.viewing_saved_details = None
                st.session_state.chat_history = [] 
                st.session_state.current_chatbot_plant_name = None
                st.rerun()

            st.subheader(f"'{nick_to_view}' Details")
            if entry.get("image"): 
                display_image_with_max_height(entry["image"], nick_to_view, max_height_px=400, use_container_width=True, fit_contain=True)
            
            st.divider()
            saved_id_res = entry.get("id_result")
            if saved_id_res: display_identification_result(saved_id_res)
            
            if st.session_state.get("plant_id_result") != saved_id_res: 
                st.session_state.plant_id_result = saved_id_res
            
            saved_care = entry.get("care_info")
            if st.session_state.get("plant_care_info") != saved_care: 
                st.session_state.plant_care_info = saved_care
            
            if st.session_state.get("current_chatbot_plant_name") != entry.get("nickname") or not st.session_state.chat_history : # Load if name changed OR history is empty
                st.session_state.chat_history = entry.get('chat_log', [])
                st.session_state.current_chatbot_plant_name = entry.get("nickname")
                st.session_state.active_chat_image_bytes = None # No image context for saved chats by default
                st.session_state.active_chat_image_type = None

            btn_col1, btn_col2 = st.columns([0.7, 0.3])
            if btn_col1.button(f"📊 View Stats for {nick_to_view}",key=f"stats_btn_saved_det_{nick_to_view.replace(' ','_')}",use_container_width=True):
                st.session_state.viewing_plant_stats_nick = nick_to_view
                st.session_state.current_nav_choice = "📊 Plant Stats"
                st.rerun()
            
            st.divider()
            if saved_care: 
                display_care_instructions(saved_care)
                st.divider()
                display_chat_interface(current_plant_care_info=saved_care, plant_id_result=saved_id_res)
            else:
                st.info("No specific care instructions were saved for this plant.")
                st.divider()
                if saved_id_res: 
                    st.info("Chat will be based on the saved identification.")
                    display_chat_interface(plant_id_result=saved_id_res)
            
            st.divider()
            confirm_key = f"confirm_del_det_{nick_to_view.replace(' ','_')}"
            if confirm_key not in st.session_state: st.session_state[confirm_key] = False
            
            if btn_col2.button(f"🗑️ Delete Profile",key=f"del_btn_saved_det_{nick_to_view.replace(' ','_')}",type="secondary",use_container_width=True, help=f"Delete {nick_to_view}'s profile"):
                st.session_state[confirm_key] = True
                st.rerun()

            if st.session_state[confirm_key]:
                st.error(f"Are you sure you want to permanently delete '{nick_to_view}'?")
                c1d,c2d, _ = st.columns([1,1,2]) 
                if c1d.button("Yes, Delete It",key=f"yes_del_final_det_{nick_to_view.replace(' ','_')}",type="primary", use_container_width=True):
                    del st.session_state.saved_photos[nick_to_view]
                    st.session_state.viewing_saved_details = None
                    clear_identification_data_soft() # Soft clear related states
                    st.session_state[confirm_key]=False
                    st.success(f"Deleted '{nick_to_view}'.")
                    st.rerun()
                if c2d.button("No, Cancel Delete",key=f"no_del_final_det_{nick_to_view.replace(' ','_')}", use_container_width=True):
                    st.session_state[confirm_key]=False
                    st.rerun()
        else: 
            st.info("Select a plant to view its details, or add a new one via 'Identify New Plant'.")
            num_g_cols=3
            sorted_plant_nicks = sorted(list(st.session_state.saved_photos.keys()))
            
            for i in range(0, len(sorted_plant_nicks), num_g_cols):
                cols = st.columns(num_g_cols)
                for j in range(num_g_cols):
                    if i + j < len(sorted_plant_nicks):
                        nick = sorted_plant_nicks[i+j]
                        data = st.session_state.saved_photos[nick]
                        with cols[j]:
                            with st.container(border=True, height=430):
                                if data.get("image"): 
                                    display_image_with_max_height(data["image"], caption=nick, max_height_px=250, use_container_width=True, fit_contain=True)
                                else: 
                                    st.markdown(f"**{nick}**")
                                id_res_g = data.get("id_result",{}); com_n_g=id_res_g.get('common_name','N/A')
                                if com_n_g and com_n_g!='N/A' and com_n_g.lower()!=nick.lower(): 
                                    st.caption(f"({com_n_g})")
                                overall_score, health_status = calculate_overall_health(data)
                                st.markdown(get_health_score_emoji_html(overall_score), unsafe_allow_html=True)
                                gc1,gc2=st.columns(2)
                                if gc1.button("Details",key=f"g_detail_gal_{nick.replace(' ','_')}",use_container_width=True): 
                                    st.session_state.viewing_saved_details=nick
                                    st.rerun()
                                if gc2.button("📊 Stats",key=f"g_stats_gal_{nick.replace(' ','_')}",use_container_width=True): 
                                    st.session_state.viewing_plant_stats_nick=nick
                                    st.session_state.current_nav_choice="📊 Plant Stats"
                                    st.rerun()

    # ==================================== PLANT STATS ====================================
    elif st.session_state.current_nav_choice == "📊 Plant Stats":
        # ... (Plant Stats logic remains largely the same) ...
        p_nick_stats = st.session_state.get("viewing_plant_stats_nick")

        if not p_nick_stats or p_nick_stats not in st.session_state.saved_photos:
            st.header("📊 Plant Health Dashboard")
            st.warning("No specific plant selected for detailed stats. Please select a plant from 'My Saved Plants' to view its dashboard.", icon="🪴")
            latest_mongo_stats = get_latest_sensor_stats()
            if latest_mongo_stats and latest_mongo_stats.get("timestamp"):
                st.subheader("🛰️ Latest General Sensor Reading (MongoDB)")
                ts_val = latest_mongo_stats["timestamp"]
                # Handle if timestamp is already datetime or needs conversion
                if isinstance(ts_val, (int, float)):
                    ts = datetime.fromtimestamp(ts_val, tz=timezone.utc).astimezone(EASTERN_TZ)
                elif isinstance(ts_val, datetime):
                    ts = ts_val.astimezone(EASTERN_TZ) if ts_val.tzinfo else EASTERN_TZ.localize(ts_val)
                else:
                    ts = None
                
                st.write(f"**Temperature**: {latest_mongo_stats.get('temperature','N/A')}°F")
                st.write(f"**Moisture Level**: {latest_mongo_stats.get('moisture_value','N/A')}")
                if ts: st.write(f"**Last Updated**: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                else: st.write(f"**Last Updated**: Timestamp format error")
            else:
                st.info("No live sensor data available from MongoDB for general overview.", icon="📡")

            if st.button("← Back to Saved Plants",key="stats_back_no_plant_sel_main"):
                st.session_state.current_nav_choice="🪴 My Saved Plants"
                st.session_state.viewing_plant_stats_nick=None
                st.rerun()
        else: 
            p_data_stats = st.session_state.saved_photos[p_nick_stats]
            st.header(f"📊 Plant Stats: {p_nick_stats}")

            if st.button("← Back to Details",key="stats_back_to_details_page_main"):
                st.session_state.current_nav_choice="🪴 My Saved Plants"
                st.session_state.viewing_saved_details=p_nick_stats 
                st.session_state.viewing_plant_stats_nick=None
                st.rerun()
            st.divider()
            current_score, health_status_text = calculate_overall_health(p_data_stats)
            MAX_HISTORY = 30
            now_iso = datetime.now(EASTERN_TZ).isoformat()
            if 'health_history' not in p_data_stats: p_data_stats['health_history'] = []
            add_new_score_to_history = True
            if p_data_stats['health_history']:
                last_ts_str, last_score = p_data_stats['health_history'][-1]
                last_dt = datetime.fromisoformat(last_ts_str)
                if last_score == current_score and (datetime.now(EASTERN_TZ) - last_dt).total_seconds() < 3600 : 
                    add_new_score_to_history = False
            if add_new_score_to_history:
                 p_data_stats['health_history'].append((now_iso, current_score))
                 p_data_stats['health_history'] = p_data_stats['health_history'][-MAX_HISTORY:]
                 st.session_state.saved_photos[p_nick_stats] = p_data_stats 

            st.subheader("Overall Health")
            st.markdown(get_health_score_emoji_html(current_score), unsafe_allow_html=True)
            st.markdown(f"**Status:** {health_status_text}")
            st.divider()
            
            moisture_level_plant = p_data_stats.get("moisture_level", random.randint(30, 70))
            last_check_ts_for_ring = p_data_stats.get("last_check_timestamp", datetime.now(EASTERN_TZ))
            if isinstance(last_check_ts_for_ring, str): last_check_ts_for_ring = datetime.fromisoformat(last_check_ts_for_ring)
            if last_check_ts_for_ring.tzinfo is None: last_check_ts_for_ring = EASTERN_TZ.localize(last_check_ts_for_ring)
            
            sim_time_moisture = last_check_ts_for_ring.strftime('%H:%M')

            ring1_moisture = generate_ring_html("Moisture", str(moisture_level_plant), f"OF {MOISTURE_MAX_PERCENT}%", 
                                                moisture_level_plant, MOISTURE_COLOR, MOISTURE_TRACK_COLOR, 
                                                sim_time_moisture, f"Soil moisture at {moisture_level_plant}%. Ideal varies.", 0)
            current_temp_plant = p_data_stats.get("temperature_value", random.uniform(68.0, 78.0))
            care_s = p_data_stats.get("care_info", {})
            temp_rng_str = care_s.get("Temperature Range", "65-85°F")
            min_f, max_f = parse_temp_range(temp_rng_str)
            temp_prog_display = ((current_temp_plant - TEMP_DISPLAY_MIN_F) / (TEMP_DISPLAY_MAX_F - TEMP_DISPLAY_MIN_F)) * 100
            ring2_temp = generate_ring_html("Temperature", str(int(current_temp_plant)), "°F NOW", temp_prog_display, 
                                            TEMPERATURE_COLOR, TEMPERATURE_TRACK_COLOR, sim_time_moisture, 
                                            f"Ambient temp {current_temp_plant:.1f}°F. Ideal: {temp_rng_str or 'N/A'}.", 1)
            mins_ago_plant = int((datetime.now(EASTERN_TZ) - last_check_ts_for_ring).total_seconds() / 60)
            fresh_prog_plant = max(0, (1 - (min(mins_ago_plant, FRESHNESS_MAX_MINUTES_AGO) / FRESHNESS_MAX_MINUTES_AGO))) * 100
            ring3_fresh = generate_ring_html("Last Update", str(mins_ago_plant), "MINS AGO", fresh_prog_plant, 
                                             FRESHNESS_COLOR, FRESHNESS_TRACK_COLOR, sim_time_moisture, 
                                             f"Plant data last updated {mins_ago_plant} mins ago.", 2)
            st.markdown(f'<div class="watch-face-grid">{ring1_moisture}{ring2_temp}{ring3_fresh}</div>', unsafe_allow_html=True)
            st.divider()
            st.subheader("📈 Health Score Over Time")
            health_hist_data = p_data_stats.get("health_history", [])
            if health_hist_data:
                df_hist = pd.DataFrame(health_hist_data, columns=['Timestamp', 'Health Score'])
                df_hist['Timestamp'] = pd.to_datetime(df_hist['Timestamp'])
                df_hist = df_hist.set_index('Timestamp')
                st.line_chart(df_hist['Health Score'])
            else:
                st.info("No health history recorded yet for this plant.")
            st.divider()
            img_c,info_c=st.columns([0.4,0.6])
            with img_c:
                 if p_data_stats.get("image"): 
                     display_image_with_max_height(p_data_stats["image"],max_height_px=250, use_container_width=True, fit_contain=True)
            with info_c:
                st.subheader("Plant Identification Refresher")
                id_res_s = p_data_stats.get("id_result",{})
                st.markdown(f"**Nickname:** {p_nick_stats}")
                st.markdown(f"**Scientific Name:** `{id_res_s.get('scientific_name','N/A')}`")
                st.markdown(f"**Common Name:** `{id_res_s.get('common_name','N/A')}`")
                st.caption(f"Full care guide and chat available on '{p_nick_stats}'s main profile page (under 'My Saved Plants').")

# Helper functions for clearing session state
def clear_identification_data_full():
    """Clears all data related to the current identification flow."""
    keys_to_clear = ["uploaded_file_bytes", "uploaded_file_type", 
                     "plant_id_result", "plant_care_info", "suggestions", 
                     "chat_history", "current_chatbot_plant_name", "saving_mode",
                     "plant_id_result_for_care_check", "suggestion_just_selected",
                     "active_chat_image_bytes", "active_chat_image_type", 
                     "new_user_message_to_process", "needs_identification", "needs_care_check"]
    for key in keys_to_clear:
        if key == "chat_history": st.session_state[key] = []
        elif key == "suggestions": st.session_state[key] = None
        elif key in ["saving_mode", "suggestion_just_selected", "new_user_message_to_process", "needs_identification", "needs_care_check"]: 
            st.session_state[key] = False
        else: st.session_state[key] = None
    # st.toast("Cleared current identification.", icon="🧹") # Optional toast

def clear_identification_data_soft():
    """Clears most ID related data but might keep uploaded_file_bytes if user is just navigating."""
    keys_to_clear = ["plant_id_result", "plant_care_info", "suggestions", 
                     "chat_history", "current_chatbot_plant_name", "saving_mode",
                     "plant_id_result_for_care_check", "suggestion_just_selected",
                     "new_user_message_to_process", "needs_identification", "needs_care_check"]
    # Keep active_chat_image_bytes and uploaded_file_bytes if user might come back
    for key in keys_to_clear:
        if key == "chat_history": st.session_state[key] = []
        elif key == "suggestions": st.session_state[key] = None
        elif key in ["saving_mode", "suggestion_just_selected", "new_user_message_to_process", "needs_identification", "needs_care_check"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# --- Run the App ---
if __name__ == "__main__":
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here": 
        st.sidebar.warning("PlantNet Key missing. ID uses demo data.",icon="🔑")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here": 
        st.sidebar.warning("Gemini Key missing. Chat limited/disabled.",icon="🔑")
    main()
