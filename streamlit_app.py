import streamlit as st
st.set_page_config(page_title="Plant Buddy", page_icon="🌿", layout="wide")

from PIL import Image
import os
import json
import requests
import base64
from io import BytesIO
import pytz
from datetime import datetime, timedelta, timezone
import random
import re
import pandas as pd

from fuzzywuzzy import process
from pymongo import MongoClient

# --- Project Imports ---
from plant_net import PlantNetAPI
from api_config import PLANTNET_API_KEY, GEMINI_API_KEY, MONGO_URI

# --- Constants ---
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
EASTERN_TZ = pytz.timezone('US/Eastern')

# Load plant care data
try:
    with open(os.path.join(os.path.dirname(__file__), 'plants_with_personality3_copy.json')) as f:
        SAMPLE_PLANT_CARE_DATA = json.load(f)
except FileNotFoundError:
    st.error("Error: `plants_with_personality3_copy.json` not found.")
    SAMPLE_PLANT_CARE_DATA = []
except json.JSONDecodeError:
    st.error("Error: `plants_with_personality3_copy.json` is not valid JSON.")
    SAMPLE_PLANT_CARE_DATA = []

# --- MongoDB Client ---
mongo_client = None
sensor_collection = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        db = mongo_client['temp_moisture']
        sensor_collection = db['c1']
        # Removed sidebar success message here to avoid clutter if it appears before sidebar itself
    except Exception as e:
        # st.sidebar.warning(f"MongoDB connection failed: {e}. Sensor stats may be limited.", icon="⚠️")
        mongo_client = None
        sensor_collection = None
# else:
    # st.sidebar.info("MongoDB URI not set. Sensor stats will be limited.", icon="ℹ️")


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
            img_styles.append(f"object-fit: {'contain' if fit_contain else 'cover'}")
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
FRESHNESS_MAX_MINUTES_AGO = 120

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
    .home-tab-content {{ background-color: #273D3D; padding: 15px; border-radius: 8px; }} /* Darker Honeydew for home page content bg */
    .health-score-heart {{ font-size: 2em; transition: color 0.5s ease; }}
    .health-good {{ color: #28a745; }}
    .health-medium {{ color: #ffc107; }}
    .health-bad {{ color: #dc3545; }}
    @keyframes pulse_green {{ 0% {{transform: scale(1);}} 50% {{transform: scale(1.1);}} 100% {{transform: scale(1);}} }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{ font-size: 1.1rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        padding: 0px 15px;
        background-color: #1f2f22; /* secondaryBackgroundColor from theme */
        border-radius: 8px 8px 0 0 !important; /* top rounded corners */
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #d08b00; /* primaryColor from theme */
    }}
    .stChatInputContainer > div {{ background-color: #2a4646; }} /* Match main background for chat input area */
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
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        common_names_list = ["Monstera deliciosa", "Fiddle Leaf Fig", "Snake Plant"]
        sci_names_list = ["Monstera deliciosa", "Ficus lyrata", "Dracaena trifasciata"]
        idx = random.randint(0, len(common_names_list)-1)
        return {
            'scientific_name': sci_names_list[idx],
            'common_name': common_names_list[idx],
            'confidence': random.uniform(70, 95),
            'raw_data': {"message": "Demo mode due to missing PlantNet API Key"}
        }
    return plantnet_api_client.identify_plant_from_bytes(image_bytes, filename)

def create_personality_profile(care_info):
    default = {"title": "Standard Plant", "traits": "observant", "prompt": "You are a plant. Respond factually."}
    if not isinstance(care_info, dict): return default
    p_data = care_info.get("Personality")
    if not isinstance(p_data, dict): return {"title": f"The {care_info.get('Plant Name', 'Plant')}", "traits": "resilient", "prompt": "Respond simply."}
    traits_list = p_data.get("Traits", ["observant"])
    traits = [str(t) for t in traits_list if t] if isinstance(traits_list, list) else ["observant"]
    return {"title": p_data.get("Title", care_info.get('Plant Name', 'Plant')), "traits": ", ".join(traits) or "observant", "prompt": p_data.get("Prompt", "Respond in character.")}

def send_message_to_gemini(messages_for_api, image_bytes=None, image_type="image/jpeg"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Chat disabled: Gemini API Key missing or invalid."

    payload_contents = list(messages_for_api) # Make a copy

    if image_bytes and payload_contents and payload_contents[-1]["role"] == "user":
        last_user_message = payload_contents[-1]
        last_user_message_parts = last_user_message.get("parts", [])
        
        if not isinstance(last_user_message_parts, list): # Ensure parts is a list
            last_user_message_parts = [{"text": str(last_user_message_parts)}]

        img_base64 = base64.b64encode(image_bytes).decode()
        last_user_message_parts.append({
            "inline_data": {"mime_type": image_type, "data": img_base64}
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
        if data.get('promptFeedback', {}).get('blockReason'):
            return f"Response blocked: {data['promptFeedback']['blockReason']}"
        return "Unexpected response from chat model."
    except requests.exceptions.Timeout: return "Request to chat model timed out."
    except requests.exceptions.RequestException as e:
        err_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try: err_detail = e.response.json().get('error', {}).get('message', e.response.text)
            except json.JSONDecodeError: err_detail = e.response.text
        return f"Chat model connection error. (Details: {err_detail})"
    except Exception as e: return f"Unexpected chat error: {str(e)}"

def get_chat_response(plant_care_info_dict, plant_id_result_dict, chat_history_list, current_user_prompt, image_bytes_for_chat=None, image_type_for_chat="image/jpeg"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Chat feature disabled: Gemini API Key not set."

    plant_name = "this plant"
    prompt_parts = ["CONTEXT: Short chatbot response (1-3 sentences).", "TASK: Act *exclusively* as the plant. Stay in character. NO mention of AI/model."]
    rules = ["RESPONSE RULES:", "1. First person (I, me, my).", "2. Embody personality.", "3. Concise (1-3 sentences).", "4. **Never break character or mention AI.**"]

    if image_bytes_for_chat:
        prompt_parts.append("INFO: User may have provided an image with their message. If they refer to 'the image' or 'this', consider the visual context.")

    if plant_care_info_dict and isinstance(plant_care_info_dict, dict):
        p = create_personality_profile(plant_care_info_dict)
        plant_name = plant_care_info_dict.get('Plant Name', 'a plant')
        prompt_parts.extend([f"PERSONALITY: '{p['title']}' (traits: {p['traits']}). Philosophy: {p['prompt']}", "CARE NEEDS (Use ONLY these):",
                             f"- Light: {plant_care_info_dict.get('Light Requirements', 'N/A')}", f"- Water: {plant_care_info_dict.get('Watering', 'N/A')}",
                             f"- Temp: {plant_care_info_dict.get('Temperature Range', 'N/A')}"])
    elif plant_id_result_dict and isinstance(plant_id_result_dict, dict) and 'error' not in plant_id_result_dict:
        plant_name = plant_id_result_dict.get('common_name', plant_id_result_dict.get('scientific_name', 'this plant'))
        if plant_name == 'N/A' or not plant_name.strip(): plant_name = 'this plant'
        prompt_parts.extend([f"Identified as '{plant_name}'. No specific stored profile.", f"Answer generally about '{plant_name}' plants."])
    else:
        return "Sorry, not enough info to chat."
    
    sys_prompt = "\n".join(prompt_parts + rules)
    
    messages_for_api = [{"role": "user", "parts": [{"text": sys_prompt}]}, 
                        {"role": "model", "parts": [{"text": f"Understood. I am {plant_name}. Ask away!"}]}]
    
    for entry in chat_history_list: # Assumes chat_history_list contains {'role': 'user/assistant', 'content': '...'}
        api_role = "model" if entry["role"] in ["assistant", "model"] else "user"
        messages_for_api.append({"role": api_role, "parts": [{"text": str(entry["content"]) if entry["content"] else ""}]})
    
    # Add current user prompt
    user_prompt_parts = [{"text": current_user_prompt}]
    if image_bytes_for_chat: # Gemini expects image with the user turn that provides it
        pass # Handled by send_message_to_gemini by adding to last user message
        
    messages_for_api.append({"role": "user", "parts": user_prompt_parts})

    return send_message_to_gemini(messages_for_api, image_bytes=image_bytes_for_chat, image_type=image_type_for_chat)


# --- MongoDB Sensor Data Helper ---
def get_latest_generic_sensor_stats():
    if sensor_collection is not None:
        try:
            latest_data = sensor_collection.find_one(sort=[('timestamp', -1)])
            if latest_data:
                return {
                    "temperature": latest_data.get("temperature"),
                    "moisture_value": latest_data.get("moisture_value"),
                    "timestamp": latest_data.get("timestamp")
                }
        except Exception as e:
            # st.toast(f"Error fetching generic sensor data: {e}", icon=" M")
            print(f"Error fetching generic sensor data: {e}") # Log instead of toast for less UI noise
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
        db_plant_name = p_entry.get('Plant Name','').lower().strip()
        if s_sci and (s_sci == db_sci_name or s_sci == db_plant_name): return p_entry
        if s_com and (s_com == db_plant_name): return p_entry
        db_common_names_list = p_entry.get('Common Names',[])
        if isinstance(db_common_names_list, list):
            if s_com and s_com in [c.lower().strip() for c in db_common_names_list if isinstance(c, str)]: return p_entry
        elif isinstance(db_common_names_list, str):
             if s_com and s_com == db_common_names_list.lower().strip(): return p_entry

    all_names_map = {}
    for p_obj in care_data_list:
        names_to_check = [p_obj.get('Scientific Name',''), p_obj.get('Plant Name','')]
        c_names = p_obj.get('Common Names',[])
        if isinstance(c_names, list): names_to_check.extend(c_names)
        elif isinstance(c_names, str): names_to_check.append(c_names)
        for name_str in names_to_check:
            if isinstance(name_str, str) and name_str.strip(): all_names_map[name_str.lower().strip()] = p_obj
    
    if not all_names_map: return None
    best_match_plant, high_score = None, 0 
    for search_term in [s_sci, s_com]:
        if search_term: 
            match_result = process.extractOne(search_term, all_names_map.keys())
            if match_result and match_result[1] >= threshold and match_result[1] > high_score: 
                high_score = match_result[1]
                best_match_plant = all_names_map.get(match_result[0])
    return best_match_plant

def display_identification_result_summary(result):
    if not result or 'error' in result: 
        st.error(f"Identification failed: {result.get('error', 'Unknown error') if result else 'No result.'}")
        return
    conf = result.get('confidence', 0); color = "#28a745" if conf > 75 else ("#ffc107" if conf > 50 else "#dc3545")
    st.markdown(f"""
    - **Scientific Name:** `{result.get('scientific_name', 'N/A')}`
    - **Common Name:** `{result.get('common_name', 'N/A')}`
    - **Confidence:** <strong style='color:{color};'>{conf:.1f}%</strong>
    """, unsafe_allow_html=True)

def display_care_instructions_details(care_info):
    if not care_info: st.warning("No detailed care information available."); return
    name = care_info.get('Plant Name', 'This Plant')
    st.subheader(f"🌱 {name} Care Guide")
    
    details_map = {
        "☀️ Light": 'Light Requirements', "💧 Water": 'Watering', "🌡️ Temp": 'Temperature Range',
        "💦 Humidity": 'Humidity Preferences', "🍃 Feeding": 'Feeding Schedule', "흙 Soil": 'Soil Type',
        "🪴 Potting": 'Potting & Repotting', "⚠️ Toxicity": 'Toxicity', "✨ Pro Tips": 'Additional Care'
    }
    
    left_col_keys = ["☀️ Light", "💧 Water", "🌡️ Temp", "💦 Humidity"]
    right_col_keys = ["🍃 Feeding", "흙 Soil", "🪴 Potting", "⚠️ Toxicity"]

    col1, col2 = st.columns(2)

    with col1:
        for label in left_col_keys:
            key = details_map[label]
            value = care_info.get(key)
            if value and str(value).strip() and str(value).lower() != 'n/a':
                st.markdown(f"**{label}**")
                st.caption(str(value))
    
    with col2:
        for label in right_col_keys:
            key = details_map[label]
            value = care_info.get(key)
            if value and str(value).strip() and str(value).lower() != 'n/a':
                st.markdown(f"**{label}**")
                st.caption(str(value))

    additional_care = care_info.get(details_map["✨ Pro Tips"])
    if additional_care and str(additional_care).strip() and str(additional_care).lower() != 'n/a':
        with st.expander("✨ Pro Tips", expanded=False):
            st.markdown(str(additional_care))


def find_similar_plant_matches(id_r, care_data_list, limit=3, score_thresh=60):
    if not id_r or 'error' in id_r or not care_data_list: return []
    all_names_map = {}
    for p_obj in care_data_list:
        names_to_check = [p_obj.get('Scientific Name',''), p_obj.get('Plant Name','')]
        c_names = p_obj.get('Common Names',[])
        if isinstance(c_names, list): names_to_check.extend(c_names)
        elif isinstance(c_names, str): names_to_check.append(c_names)
        for name_str in names_to_check:
            if isinstance(name_str, str) and name_str.strip():
                all_names_map[name_str.lower().strip()] = p_obj 
    if not all_names_map: return []
    search_terms = [term.lower().strip() for term in [id_r.get('scientific_name',''), id_r.get('common_name','')] if term]
    potential_matches = {}
    for term in search_terms:
        if term:
            fuzz_results = process.extract(term, all_names_map.keys(), limit=limit*3)
            for match_name_key, score in fuzz_results:
                if score >= score_thresh:
                    matched_plant_obj = all_names_map[match_name_key]
                    plant_id = id(matched_plant_obj) 
                    if plant_id not in potential_matches or score > potential_matches[plant_id]['score']:
                        potential_matches[plant_id] = {'plant': matched_plant_obj, 'score': score}
    sorted_matches = sorted(potential_matches.values(), key=lambda x: x['score'], reverse=True)
    final_suggestions = []
    seen_plant_names_for_suggestion = set()
    original_id_name = id_r.get('common_name', id_r.get('scientific_name', '')).lower()
    for match_data in sorted_matches:
        p_info = match_data['plant']
        p_sugg_name = p_info.get('Plant Name', p_info.get('Scientific Name',''))
        if p_sugg_name.lower() == original_id_name: continue
        if p_sugg_name not in seen_plant_names_for_suggestion:
            final_suggestions.append(p_info)
            seen_plant_names_for_suggestion.add(p_sugg_name)
            if len(final_suggestions) >= limit: break
    return final_suggestions

def display_suggestion_buttons_for_id_flow(suggestions, care_data):
    if not suggestions: return
    st.info("📋 No exact care guide found. Perhaps one of these is a closer match from our database?")
    cols = st.columns(len(suggestions))
    for i, p_info in enumerate(suggestions):
        p_name = p_info.get('Plant Name', p_info.get('Scientific Name', f'Suggestion {i+1}'))
        tip = f"Select {p_name}" + (f" (Sci: {p_info.get('Scientific Name')})" if p_info.get('Scientific Name','') != p_name else "")
        
        if cols[i].button(p_name, key=f"id_sugg_btn_{i}", help=tip, use_container_width=True):
            new_id_result = {
                'scientific_name': p_info.get('Scientific Name','N/A'), 
                'common_name': p_name, 
                'confidence': 100.0, 
                'raw_data': {"message": "Selected from database suggestion"}
            }
            st.session_state.current_id_result = new_id_result
            st.session_state.current_id_care_info = p_info # Suggestion is the care info
            st.session_state.current_id_suggestions = [] # Clear suggestions
            st.session_state.current_id_chat_history = [] # Reset chat
            st.experimental_rerun()

def display_chat_ui_custom(
    chat_history_list,  # The list of messages [{role: str, content: str, time: str}]
    chatbot_name_str,
    plant_care_info_dict,
    plant_id_result_dict,
    on_new_message_submit, # Callback: takes (user_prompt, image_bytes, image_type) returns bot_response
    allow_image_upload_in_chat=False, # Optional: enable image upload within chat
    chat_input_key_suffix=""
    ):

    st.markdown("""<style>.message-container{padding:1px 5px}.user-message{background:#0b81fe;color:white;border-radius:18px 18px 0 18px;padding:8px 14px;margin:3px 0 3px auto;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.1);animation:fadeIn .3s ease-out}.bot-message{background:#e5e5ea;color:#000;border-radius:18px 18px 18px 0;padding:8px 14px;margin:3px auto 3px 0;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.05);animation:fadeIn .3s ease-out}.message-meta{font-size:.7rem;color:#777;margin-top:3px}.bot-message .message-meta{text-align:left;color:#555}.user-message .message-meta{text-align:right}@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}</style>""", unsafe_allow_html=True)

    chat_container = st.container(height=350)
    with chat_container:
        for msg in chat_history_list:
            role, content, time = msg.get("role"), msg.get("content", ""), msg.get("time", "")
            if role == "user": st.markdown(f'<div class="message-container"><div class="user-message">{content}<div class="message-meta">You • {time}</div></div></div>', unsafe_allow_html=True)
            elif role in ["assistant", "model"]: st.markdown(f'<div class="message-container"><div class="bot-message">🌿 {content}<div class="message-meta">{chatbot_name_str} • {time}</div></div></div>', unsafe_allow_html=True)

    chat_img_bytes, chat_img_type = None, None
    if allow_image_upload_in_chat:
        # Simplified: Use a flag in session state if an image is uploaded specifically for chat
        # This example doesn't implement a separate uploader *in* the chat UI
        # It relies on an external mechanism to set image_bytes_for_chat if needed
        if st.session_state.get(f"send_image_with_next_message_{chat_input_key_suffix}"):
            chat_img_bytes = st.session_state.get(f"active_chat_image_bytes_{chat_input_key_suffix}")
            chat_img_type = st.session_state.get(f"active_chat_image_type_{chat_input_key_suffix}")


    if prompt := st.chat_input(f"Ask {chatbot_name_str}...", key=f"chat_input_{chat_input_key_suffix}"):
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        chat_history_list.append({"role": "user", "content": prompt, "time": timestamp})
        
        # If an image was flagged to be sent with this message
        current_chat_img_bytes = chat_img_bytes if st.session_state.get(f"send_image_with_next_message_{chat_input_key_suffix}") else None
        current_chat_img_type = chat_img_type if st.session_state.get(f"send_image_with_next_message_{chat_input_key_suffix}") else None

        with st.spinner(f"{chatbot_name_str} is thinking..."):
            bot_response_content = get_chat_response(
                plant_care_info_dict,
                plant_id_result_dict,
                chat_history_list[:-1], # History before this new user prompt
                prompt, # Current user prompt
                image_bytes_for_chat=current_chat_img_bytes,
                image_type_for_chat=current_chat_img_type
            )
        
        chat_history_list.append({"role": "assistant", "content": bot_response_content, "time": datetime.now(EASTERN_TZ).strftime("%H:%M")})
        
        # Reset image flag for this chat context
        if st.session_state.get(f"send_image_with_next_message_{chat_input_key_suffix}"):
            st.session_state[f"send_image_with_next_message_{chat_input_key_suffix}"] = False
            st.session_state[f"active_chat_image_bytes_{chat_input_key_suffix}"] = None
            st.session_state[f"active_chat_image_type_{chat_input_key_suffix}"] = None

        # The caller of display_chat_ui_custom is responsible for saving updated chat_history_list
        on_new_message_submit() # This will trigger a rerun in the parent context

# --- Health Score Calculation ---
def calculate_health_score_component(value, ideal_min, ideal_max, lower_is_better=False):
    if value is None or ideal_min is None or ideal_max is None: return 50
    if lower_is_better:
        if value <= ideal_min: return 100
        if value >= ideal_max: return 0
        return 100 - ((value - ideal_min) / (ideal_max - ideal_min) * 100)
    else:
        if ideal_min <= value <= ideal_max: return 100
        if value < ideal_min:
            range_below = ideal_min - (ideal_min * 0.5) 
            if range_below <= 0: range_below = ideal_min / 2 if ideal_min > 0 else 50
            penalty = ((ideal_min - value) / range_below) * 100
            return max(0, 100 - penalty)
        if value > ideal_max:
            range_above = (ideal_max * 1.5) - ideal_max
            if range_above <= 0: range_above = ideal_max / 2 if ideal_max > 0 else 50
            penalty = ((value - ideal_max) / range_above) * 100
            return max(0, 100 - penalty)
    return 50

def calculate_overall_health(p_data_stats):
    if not p_data_stats: return 0, "No data"
    scores = []
    moisture_val = p_data_stats.get("moisture_level", 50)
    moisture_score = calculate_health_score_component(moisture_val, 40, 80) # Ideal: 40-80%
    scores.append(moisture_score)

    temp_val = p_data_stats.get("temperature_value")
    ideal_temp_min, ideal_temp_max = None, None
    if p_data_stats.get("care_info") and p_data_stats["care_info"].get("Temperature Range"):
        ideal_temp_min, ideal_temp_max = parse_temp_range(p_data_stats["care_info"]["Temperature Range"])
    temp_score = calculate_health_score_component(temp_val, ideal_temp_min or 60, ideal_temp_max or 80)
    scores.append(temp_score)

    last_check_ts = p_data_stats.get("last_check_timestamp")
    mins_ago = FRESHNESS_MAX_MINUTES_AGO + 1
    if last_check_ts:
        if isinstance(last_check_ts, datetime):
            last_check_ts_aware = last_check_ts.astimezone(EASTERN_TZ) if last_check_ts.tzinfo else EASTERN_TZ.localize(last_check_ts)
            mins_ago = (datetime.now(EASTERN_TZ) - last_check_ts_aware).total_seconds() / 60
        elif isinstance(last_check_ts, (int, float)):
            last_check_dt = datetime.fromtimestamp(last_check_ts, tz=timezone.utc).astimezone(EASTERN_TZ)
            mins_ago = (datetime.now(EASTERN_TZ) - last_check_dt).total_seconds() / 60
    freshness_score = calculate_health_score_component(mins_ago, 0, FRESHNESS_MAX_MINUTES_AGO, lower_is_better=True)
    scores.append(freshness_score)

    overall_score = sum(scores) / len(scores) if scores else 0
    status_text = "Needs Attention"
    if overall_score >= 80: status_text = "Excellent"
    elif overall_score >= 60: status_text = "Good"
    elif overall_score >= 40: status_text = "Fair"
    return round(overall_score), status_text

def get_health_score_emoji_html(score):
    heart_class = "health-bad"; heart_symbol = "💔"
    if score >= 80: heart_class = "health-good"; heart_symbol = "❤️"
    elif score >= 50: heart_class = "health-medium"; heart_symbol = "💛"
    animation_style = "animation: pulse_green 1.5s infinite;" if score >=80 else ""
    return f'<span class="health-score-heart {heart_class}" style="{animation_style}">{heart_symbol}</span> Overall Health: {score:.0f}%'

# --- Initialize Session State ---
def initialize_session_state_V2():
    defaults = {
        "current_nav_choice": "🏠 Home",
        "saved_photos": {}, # Main store for saved plants
        "viewing_saved_plant_nickname": None, # For "My Plants" detail view

        # States for "Identify New Plant" flow
        "current_id_image_bytes": None,
        "current_id_image_type": None,
        "current_id_result": None,
        "current_id_care_info": None,
        "current_id_suggestions": None,
        "current_id_chat_history": [],
        "current_id_send_image_with_next_message": False, # Flag if main image should be sent with chat
        
        "welcome_response_generated": False,
        "welcome_response": "",

        # For chat in "My Saved Plants" (manages its own image context if ever needed)
        "saved_plant_chat_send_image": False, 
        "saved_plant_chat_image_bytes": None,
        "saved_plant_chat_image_type": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: 
            if isinstance(v, list): st.session_state[k] = list(v) # Ensure mutable defaults are copied
            elif isinstance(v, dict): st.session_state[k] = dict(v)
            else: st.session_state[k] = v
    
    # Load example plants if none are saved and files exist
    if not st.session_state.saved_photos:
        example_plants_loaded = 0
        try:
            example_plant_1_path = os.path.join(os.path.dirname(__file__), "example_plant_1.jpg")
            if os.path.exists(example_plant_1_path):
                with open(example_plant_1_path, "rb") as f1: img1_bytes = f1.read()
                img1_b64 = base64.b64encode(img1_bytes).decode()
                care_info1 = find_care_instructions("Monstera deliciosa", SAMPLE_PLANT_CARE_DATA)
                st.session_state.saved_photos["My Monstera"] = {
                    "nickname": "My Monstera", "image": f"data:image/jpg;base64,{img1_b64}",
                    "id_result": {'scientific_name': 'Monstera deliciosa', 'common_name': 'Monstera', 'confidence': 95.0},
                    "care_info": care_info1 or (SAMPLE_PLANT_CARE_DATA[0] if SAMPLE_PLANT_CARE_DATA else {}),
                    "chat_log": [{"role": "assistant", "content": "Hello! I'm your example Monstera.", "time": ""}],
                    "moisture_level": random.randint(40,70), "temperature_value": random.uniform(68.0,75.0),
                    "last_check_timestamp": datetime.now(EASTERN_TZ) - timedelta(hours=random.randint(1,5)),
                    "health_history": []
                }
                example_plants_loaded +=1
        except Exception as e: print(f"Error loading example plant 1: {e}")
        
        try:
            example_plant_2_path = os.path.join(os.path.dirname(__file__), "example_plant_2.jpg")
            if os.path.exists(example_plant_2_path):
                with open(example_plant_2_path, "rb") as f2: img2_bytes = f2.read()
                img2_b64 = base64.b64encode(img2_bytes).decode()
                care_info2 = find_care_instructions("Snake Plant", SAMPLE_PLANT_CARE_DATA)
                st.session_state.saved_photos["Sneaky Snake"] = {
                    "nickname": "Sneaky Snake", "image": f"data:image/jpg;base64,{img2_b64}",
                    "id_result": {'scientific_name': 'Dracaena trifasciata', 'common_name': 'Snake Plant', 'confidence': 92.0},
                    "care_info": care_info2 or (SAMPLE_PLANT_CARE_DATA[1] if len(SAMPLE_PLANT_CARE_DATA)>1 else {}),
                    "chat_log": [],
                    "moisture_level": random.randint(30,60), "temperature_value": random.uniform(70.0,78.0),
                    "last_check_timestamp": datetime.now(EASTERN_TZ) - timedelta(minutes=random.randint(30,180)),
                    "health_history": []
                }
                example_plants_loaded +=1
        except Exception as e: print(f"Error loading example plant 2: {e}")
        
        if example_plants_loaded == 0 and not st.session_state.saved_photos:
             st.toast("Example plant images not found. Examples skipped.", icon="⚠️")


def clear_current_identification_flow_data():
    st.session_state.current_id_image_bytes = None
    st.session_state.current_id_image_type = None
    st.session_state.current_id_result = None
    st.session_state.current_id_care_info = None
    st.session_state.current_id_suggestions = None
    st.session_state.current_id_chat_history = []
    st.session_state.current_id_send_image_with_next_message = False
    # st.toast("Identification flow cleared.", icon="🧹")


# =======================================================
# ===== PAGE RENDERING FUNCTIONS =====
# =======================================================

def render_home_page(care_data):
    st.markdown("<div class='home-tab-content'>", unsafe_allow_html=True)
    st.header("🌿 Welcome to Plant Buddy!")
    
    default_welcome_msg = "Welcome! Identify plants, get care tips, chat with them, and track their health. Happy gardening!"
    if not st.session_state.welcome_response_generated:
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            with st.spinner("Plant Buddy is waking up..."):
                welcome_payload = [{"role":"user","parts":[{"text":"System: You are Plant Buddy, a friendly app assistant. Generate a short, cheerful welcome (2-3 sentences) for users. Mention identifying plants, care tips, plant chat, and health tracking."}]}, {"role":"model","parts":[{"text":"Okay!"}]}]
                st.session_state.welcome_response = send_message_to_gemini(welcome_payload)
        else:
            st.session_state.welcome_response = default_welcome_msg
        st.session_state.welcome_response_generated = True
    
    current_welcome = st.session_state.welcome_response or default_welcome_msg
    if "Sorry" in current_welcome or "disabled" in current_welcome or "blocked" in current_welcome: current_welcome = default_welcome_msg

    st.markdown(f"""<div style="background-color: #3c5c5c; padding:20px; border-radius:10px; border-left:5px solid #d08b00; margin-bottom:20px;"><h3 style="color:#efefef;">🌱 Hello Plant Lover!</h3><p style="font-size:1.1em; color:#f0f0f0;">{current_welcome}</p></div>""", unsafe_allow_html=True)
    
    st.subheader("🔍 Quick Actions")
    hc1,hc2=st.columns(2)
    if hc1.button("📸 Identify My Plant!",use_container_width=True,type="primary"): 
        st.session_state.current_nav_choice="🆔 Identify New Plant"; st.experimental_rerun()
    if hc2.button("💚 Go to My Plants",use_container_width=True): 
        st.session_state.current_nav_choice="🪴 My Plants"; st.experimental_rerun()

    if mongo_client and sensor_collection:
        st.divider()
        st.subheader("🛰️ Live Sensor Snapshot (MongoDB)")
        latest_mongo_stats = get_latest_generic_sensor_stats()
        if latest_mongo_stats and latest_mongo_stats.get("timestamp"):
            ts_val = latest_mongo_stats["timestamp"]
            ts = datetime.fromtimestamp(ts_val, tz=timezone.utc).astimezone(EASTERN_TZ) if isinstance(ts_val, (int,float)) else EASTERN_TZ.localize(ts_val) if isinstance(ts_val, datetime) and ts_val.tzinfo is None else ts_val
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric("Temperature", f"{latest_mongo_stats.get('temperature','N/A')}°F")
            stat_col2.metric("Moisture", f"{latest_mongo_stats.get('moisture_value','N/A')}")
            if ts: stat_col3.caption(f"Last Update: {ts.strftime('%H:%M:%S %Z')}")
            else: stat_col3.caption("Last Update: Error")

            if latest_mongo_stats.get('moisture_value', 500) < 400 : st.warning("Low moisture detected by general sensor.")
            elif latest_mongo_stats.get('moisture_value', 500) > 800 : st.info("High moisture detected by general sensor.")

        else:
            st.info("No live generic sensor data available at the moment.", icon="📡")


    if st.session_state.saved_photos:
        st.divider(); st.subheader("🪴 Your Recently Added Plants") 
        recent_nicks = list(reversed(list(st.session_state.saved_photos.keys())))[:3]
        if recent_nicks: 
            cols_home = st.columns(len(recent_nicks))
            for i, nick in enumerate(recent_nicks): 
                p_data = st.session_state.saved_photos[nick]
                with cols_home[i], st.container(border=True):
                    display_image_with_max_height(p_data.get("image",""), caption=nick, max_height_px=180, use_container_width=True, fit_contain=True)
                    id_res = p_data.get("id_result", {}); com_n = id_res.get('common_name', 'N/A')
                    if com_n and com_n != 'N/A' and com_n.lower() != nick.lower(): st.caption(f"({com_n})")
                    overall_score, _ = calculate_overall_health(p_data)
                    st.markdown(get_health_score_emoji_html(overall_score), unsafe_allow_html=True)
                    if st.button("View Details", key=f"home_view_{nick.replace(' ','_')}", use_container_width=True):
                        st.session_state.viewing_saved_plant_nickname = nick
                        st.session_state.current_nav_choice = "🪴 My Plants"
                        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_identify_page(care_data):
    st.header("🔎 Identify a New Plant")

    uploader_col, controls_col = st.columns([2,1])
    with uploader_col:
        up_file = st.file_uploader("Upload a clear photo of your plant:", type=["jpg","jpeg","png"], key="id_uploader")
    
    with controls_col:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🗑️ Clear Image & Start Over", key="id_clear_flow", use_container_width=True, type="secondary"):
            clear_current_identification_flow_data()
            st.experimental_rerun()

    if up_file:
        new_file_bytes = up_file.getvalue()
        if st.session_state.current_id_image_bytes != new_file_bytes:
            clear_current_identification_flow_data() # Clear previous ID flow for new image
            st.session_state.current_id_image_bytes = new_file_bytes
            st.session_state.current_id_image_type = up_file.type
            st.session_state.current_id_send_image_with_next_message = True # Flag this image for chat
            st.experimental_rerun()

    if st.session_state.current_id_image_bytes:
        st.divider()
        img_display_col, id_action_col = st.columns([2,1])
        with img_display_col:
            display_image_with_max_height(st.session_state.current_id_image_bytes, "Your Plant", max_height_px=350, use_container_width=True, fit_contain=True)
        
        if not st.session_state.current_id_result: # Only show Identify button if no result yet
            with id_action_col:
                st.write("") # Spacer
                if st.button("💡 Identify This Plant!", key="id_identify_btn", type="primary", use_container_width=True):
                    with st.spinner("Identifying your plant... 🌱"):
                        st.session_state.current_id_result = identify_plant_wrapper(
                            st.session_state.current_id_image_bytes, 
                            "uploaded_plant_image.jpg"
                        )
                        # After identification, try to fetch care info or suggestions
                        if st.session_state.current_id_result and 'error' not in st.session_state.current_id_result:
                            st.session_state.current_id_care_info = find_care_instructions(st.session_state.current_id_result, care_data)
                            if not st.session_state.current_id_care_info:
                                st.session_state.current_id_suggestions = find_similar_plant_matches(st.session_state.current_id_result, care_data)
                            else: # Care info found
                                st.session_state.current_id_suggestions = [] # No need for suggestions
                        else: # Error in identification
                            st.session_state.current_id_care_info = None
                            st.session_state.current_id_suggestions = []
                    st.experimental_rerun()
        
        st.divider()

        if st.session_state.current_id_result:
            tab1, tab2, tab3 = st.tabs(["🔍 Results & Care", "💬 Chat", "💾 Save"])

            with tab1:
                st.subheader("Identification & Care Information")
                display_identification_result_summary(st.session_state.current_id_result)
                
                if st.session_state.current_id_result and 'error' not in st.session_state.current_id_result:
                    if st.session_state.current_id_care_info:
                        display_care_instructions_details(st.session_state.current_id_care_info)
                    elif st.session_state.current_id_suggestions:
                        display_suggestion_buttons_for_id_flow(st.session_state.current_id_suggestions, care_data)
                    else: # No care info and no suggestions (or suggestions processed)
                        st.info("No specific care instructions or close matches found in our database for this identification.")
                else:
                     st.warning("Cannot fetch care information due to identification error.")


            with tab2:
                st.subheader("Chat With This Plant")
                if st.session_state.current_id_result and 'error' not in st.session_state.current_id_result:
                    chatbot_name = st.session_state.current_id_result.get('common_name') or st.session_state.current_id_result.get('scientific_name', 'this plant')
                    if st.session_state.current_id_care_info:
                         chatbot_name = st.session_state.current_id_care_info.get('Plant Name', chatbot_name)
                    
                    # This callback will be used by display_chat_ui_custom to trigger a rerun
                    def id_chat_rerun_callback():
                        st.experimental_rerun()

                    # Logic to pass the main uploaded image to the first chat message if desired
                    # The `current_id_send_image_with_next_message` flag controls this.
                    # It's set when a new image is uploaded.
                    # `get_chat_response` will check for image_bytes_for_chat.
                    # The display_chat_ui_custom itself won't handle the image bytes directly for *this initial image*,
                    # but rather it's passed to get_chat_response via its parameters.
                    # We rely on `current_id_send_image_with_next_message` to be managed carefully.
                    
                    # Note: `display_chat_ui_custom` modifies `st.session_state.current_id_chat_history` in place.
                    display_chat_ui_custom(
                        chat_history_list=st.session_state.current_id_chat_history,
                        chatbot_name_str=chatbot_name,
                        plant_care_info_dict=st.session_state.current_id_care_info,
                        plant_id_result_dict=st.session_state.current_id_result,
                        on_new_message_submit=id_chat_rerun_callback,
                        chat_input_key_suffix="identify_flow"
                        # allow_image_upload_in_chat = True # If we add separate uploader in chat
                    )
                    # After the first message that might use the image, clear the flag
                    if st.session_state.current_id_send_image_with_next_message and st.session_state.current_id_chat_history:
                        st.session_state.current_id_send_image_with_next_message = False

                else:
                    st.warning("Identify the plant first to enable chat.")

            with tab3:
                st.subheader("Save Plant Profile")
                if st.session_state.current_id_result and 'error' not in st.session_state.current_id_result:
                    default_nick = st.session_state.current_id_result.get('common_name') or \
                                   st.session_state.current_id_result.get('scientific_name', 'My Plant')
                    
                    with st.form("save_id_form"):
                        plant_nickname = st.text_input("Plant Nickname:", value=default_nick)
                        submitted_save = st.form_submit_button("✅ Confirm & Save to My Plants")

                        if submitted_save:
                            if not plant_nickname.strip():
                                st.warning("Nickname is required.")
                            elif plant_nickname in st.session_state.saved_photos:
                                st.warning(f"Nickname '{plant_nickname}' already exists.")
                            else:
                                img_b64 = base64.b64encode(st.session_state.current_id_image_bytes).decode()
                                st.session_state.saved_photos[plant_nickname] = {
                                    "nickname": plant_nickname,
                                    "image": f"data:{st.session_state.current_id_image_type};base64,{img_b64}",
                                    "id_result": st.session_state.current_id_result,
                                    "care_info": st.session_state.current_id_care_info,
                                    "chat_log": st.session_state.current_id_chat_history,
                                    "moisture_level": random.randint(30,90),
                                    "temperature_value": random.uniform(65.0, 78.0),
                                    "last_check_timestamp": datetime.now(EASTERN_TZ) - timedelta(hours=random.randint(1,3)),
                                    "health_history": [(datetime.now(EASTERN_TZ).isoformat(), random.randint(60,90))]
                                }
                                st.success(f"'{plant_nickname}' saved! View in 'My Plants'."); st.balloons()
                                prev_nick = plant_nickname # Store for navigation
                                clear_current_identification_flow_data()
                                st.session_state.current_nav_choice = "🪴 My Plants"
                                st.session_state.viewing_saved_plant_nickname = prev_nick
                                st.experimental_rerun()
                else:
                    st.warning("Identify the plant first to save its profile.")
    else:
        st.info("Upload an image above to begin plant identification.")


def render_my_plants_page(care_data):
    st.header("🪴 My Saved Plant Profiles")
    nick_to_view = st.session_state.get("viewing_saved_plant_nickname")

    if not st.session_state.saved_photos:
        st.info("You haven't saved any plants yet. Go to 'Identify New Plant' to start your collection!")
        return

    if nick_to_view and nick_to_view in st.session_state.saved_photos:
        # --- Detail View ---
        plant_data = st.session_state.saved_photos[nick_to_view]
        
        if st.button("← Back to All Saved Plants", key="myplants_back_to_gallery"):
            st.session_state.viewing_saved_plant_nickname = None
            st.experimental_rerun()

        st.subheader(f"'{nick_to_view}'")
        if plant_data.get("image"):
            display_image_with_max_height(plant_data["image"], nick_to_view, max_height_px=300, use_container_width=True, fit_contain=True)
        st.divider()

        tab_overview, tab_chat, tab_stats, tab_manage = st.tabs(["📋 Overview & Care", "💬 Chat Log", "📊 Health Stats", "⚙️ Manage"])

        with tab_overview:
            if plant_data.get("id_result"):
                display_identification_result_summary(plant_data["id_result"])
            st.divider()
            display_care_instructions_details(plant_data.get("care_info"))

        with tab_chat:
            # Ensure chat history is a list
            if not isinstance(plant_data.get('chat_log'), list):
                plant_data['chat_log'] = []

            def saved_chat_rerun_callback():
                # The chat history is directly modified in plant_data['chat_log'] by display_chat_ui_custom
                st.session_state.saved_photos[nick_to_view] = plant_data # Save back to main store
                st.experimental_rerun()
            
            display_chat_ui_custom(
                chat_history_list=plant_data['chat_log'], # Pass the mutable list
                chatbot_name_str=nick_to_view,
                plant_care_info_dict=plant_data.get("care_info"),
                plant_id_result_dict=plant_data.get("id_result"),
                on_new_message_submit=saved_chat_rerun_callback,
                chat_input_key_suffix=f"saved_{nick_to_view.replace(' ','_')}"
            )


        with tab_stats:
            render_plant_health_stats_tab(plant_data, nick_to_view)
        
        with tab_manage:
            st.subheader("Manage Profile")
            confirm_key = f"confirm_del_saved_{nick_to_view.replace(' ','_')}"
            if confirm_key not in st.session_state: st.session_state[confirm_key] = False
            
            if st.button(f"🗑️ Delete '{nick_to_view}' Profile", key=f"del_btn_detail_{nick_to_view.replace(' ','_')}", type="secondary", use_container_width=True):
                st.session_state[confirm_key] = True
                st.experimental_rerun()

            if st.session_state[confirm_key]:
                st.error(f"Are you sure you want to permanently delete '{nick_to_view}'?")
                c1d,c2d, _ = st.columns([1,1,2]) 
                if c1d.button("Yes, Delete",key=f"yes_del_final_detail_{nick_to_view.replace(' ','_')}",type="primary", use_container_width=True):
                    del st.session_state.saved_photos[nick_to_view]
                    st.session_state.viewing_saved_plant_nickname = None
                    st.session_state[confirm_key]=False
                    st.success(f"Deleted '{nick_to_view}'.")
                    st.experimental_rerun()
                if c2d.button("No, Cancel",key=f"no_del_final_detail_{nick_to_view.replace(' ','_')}", use_container_width=True):
                    st.session_state[confirm_key]=False
                    st.experimental_rerun()
    else:
        # --- Gallery View ---
        st.info("Select a plant to view its details, or add a new one via 'Identify New Plant'.")
        num_g_cols=3
        sorted_plant_nicks = sorted(list(st.session_state.saved_photos.keys()))
        
        for i in range(0, len(sorted_plant_nicks), num_g_cols):
            cols = st.columns(num_g_cols)
            for j in range(num_g_cols):
                if i + j < len(sorted_plant_nicks):
                    nick = sorted_plant_nicks[i+j]
                    data = st.session_state.saved_photos[nick]
                    with cols[j], st.container(border=True, height=380): # Adjusted height
                        if data.get("image"): 
                            display_image_with_max_height(data["image"], caption=nick, max_height_px=200, use_container_width=True, fit_contain=True)
                        else: st.markdown(f"**{nick}**")
                        
                        id_res_g = data.get("id_result",{}); com_n_g=id_res_g.get('common_name','N/A')
                        if com_n_g and com_n_g!='N/A' and com_n_g.lower()!=nick.lower(): 
                            st.caption(f"({com_n_g})")
                        
                        overall_score, _ = calculate_overall_health(data)
                        st.markdown(get_health_score_emoji_html(overall_score), unsafe_allow_html=True)
                        
                        if st.button("View Details",key=f"gallery_detail_{nick.replace(' ','_')}",use_container_width=True): 
                            st.session_state.viewing_saved_plant_nickname=nick
                            st.experimental_rerun()

def render_plant_health_stats_tab(p_data_stats, p_nick_stats):
    st.subheader("Current Health Assessment")
    current_score, health_status_text = calculate_overall_health(p_data_stats)
    MAX_HISTORY = 30
    now_iso = datetime.now(EASTERN_TZ).isoformat()
    if 'health_history' not in p_data_stats: p_data_stats['health_history'] = []
    
    add_new_score_to_history = True
    if p_data_stats['health_history']:
        last_ts_str, last_score = p_data_stats['health_history'][-1]
        try: last_dt = datetime.fromisoformat(last_ts_str)
        except ValueError: last_dt = datetime.now(EASTERN_TZ) - timedelta(days=1) # force update if parse fails
        
        if last_score == current_score and (datetime.now(EASTERN_TZ) - last_dt).total_seconds() < 3600 : 
            add_new_score_to_history = False
            
    if add_new_score_to_history:
         p_data_stats['health_history'].append((now_iso, current_score))
         p_data_stats['health_history'] = p_data_stats['health_history'][-MAX_HISTORY:]
         st.session_state.saved_photos[p_nick_stats] = p_data_stats 

    st.markdown(get_health_score_emoji_html(current_score), unsafe_allow_html=True)
    st.markdown(f"**Status:** {health_status_text}")
    st.divider()
    
    moisture_level_plant = p_data_stats.get("moisture_level", random.randint(30, 70))
    last_check_ts_for_ring_raw = p_data_stats.get("last_check_timestamp", datetime.now(EASTERN_TZ))
    
    # Ensure last_check_ts_for_ring is a datetime object and timezone-aware
    if isinstance(last_check_ts_for_ring_raw, str):
        try: last_check_ts_for_ring = datetime.fromisoformat(last_check_ts_for_ring_raw)
        except ValueError: last_check_ts_for_ring = datetime.now(EASTERN_TZ) # Fallback
    elif isinstance(last_check_ts_for_ring_raw, (int, float)): # Unix timestamp
        last_check_ts_for_ring = datetime.fromtimestamp(last_check_ts_for_ring_raw, tz=timezone.utc)
    else: # Already datetime
        last_check_ts_for_ring = last_check_ts_for_ring_raw

    if last_check_ts_for_ring.tzinfo is None: 
        last_check_ts_for_ring = EASTERN_TZ.localize(last_check_ts_for_ring)
    else:
        last_check_ts_for_ring = last_check_ts_for_ring.astimezone(EASTERN_TZ)

    sim_time_moisture = last_check_ts_for_ring.strftime('%H:%M')

    ring1_moisture = generate_ring_html("Moisture", str(moisture_level_plant), f"OF {MOISTURE_MAX_PERCENT}%", 
                                        moisture_level_plant, MOISTURE_COLOR, MOISTURE_TRACK_COLOR, 
                                        sim_time_moisture, f"Soil moisture: {moisture_level_plant}%. Ideal varies.", 0)
    current_temp_plant = p_data_stats.get("temperature_value", random.uniform(68.0, 78.0))
    care_s = p_data_stats.get("care_info", {})
    temp_rng_str = care_s.get("Temperature Range", "65-85°F")
    temp_prog_display = ((current_temp_plant - TEMP_DISPLAY_MIN_F) / (TEMP_DISPLAY_MAX_F - TEMP_DISPLAY_MIN_F)) * 100
    ring2_temp = generate_ring_html("Temperature", str(int(current_temp_plant)), "°F NOW", temp_prog_display, 
                                    TEMPERATURE_COLOR, TEMPERATURE_TRACK_COLOR, sim_time_moisture, 
                                    f"Ambient: {current_temp_plant:.1f}°F. Ideal: {temp_rng_str or 'N/A'}.", 1)
    mins_ago_plant = int((datetime.now(EASTERN_TZ) - last_check_ts_for_ring).total_seconds() / 60)
    fresh_prog_plant = max(0, (1 - (min(mins_ago_plant, FRESHNESS_MAX_MINUTES_AGO) / FRESHNESS_MAX_MINUTES_AGO))) * 100
    ring3_fresh = generate_ring_html("Last Update", str(mins_ago_plant), "MINS AGO", fresh_prog_plant, 
                                     FRESHNESS_COLOR, FRESHNESS_TRACK_COLOR, sim_time_moisture, 
                                     f"Data updated {mins_ago_plant} mins ago.", 2)
    st.markdown(f'<div class="watch-face-grid">{ring1_moisture}{ring2_temp}{ring3_fresh}</div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("📈 Health Score Over Time")
    health_hist_data = p_data_stats.get("health_history", [])
    if health_hist_data and len(health_hist_data) > 1 :
        df_hist = pd.DataFrame(health_hist_data, columns=['Timestamp', 'Health Score'])
        try:
            df_hist['Timestamp'] = pd.to_datetime(df_hist['Timestamp'])
            df_hist = df_hist.set_index('Timestamp')
            st.line_chart(df_hist['Health Score'])
        except Exception as e:
            st.warning(f"Could not display health chart: {e}")
            # st.dataframe(df_hist) # For debugging
    else:
        st.info("Not enough health history recorded yet to display a chart for this plant.")


# --- Main App Logic ---
def main():
    st.markdown(f'<link rel="manifest" href="manifest.json">', unsafe_allow_html=True)
    initialize_session_state_V2()
    st.markdown(get_ring_html_css(), unsafe_allow_html=True)

    st.sidebar.title("📚 Plant Buddy")
    # Sidebar API status
    if PLANTNET_API_KEY and PLANTNET_API_KEY != "your_plantnet_api_key_here":
        st.sidebar.success("PlantNet API Ready", icon="🌿")
    else:
        st.sidebar.warning("PlantNet API Key Missing. ID uses demo data.",icon="🔑")
    
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        st.sidebar.success("Gemini API Ready", icon="💬")
    else:
        st.sidebar.warning("Gemini API Key Missing. Chat disabled/limited.",icon="🔑")

    if mongo_client and sensor_collection:
        st.sidebar.success("MongoDB Connected", icon="🍃")
    else:
        st.sidebar.info("MongoDB Not Connected. Live sensor data unavailable.", icon="ℹ️")
    
    st.sidebar.divider()

    nav_options = ["🏠 Home", "🆔 Identify New Plant", "🪴 My Plants"]
    try: nav_idx = nav_options.index(st.session_state.current_nav_choice)
    except ValueError: nav_idx = 0; st.session_state.current_nav_choice = "🏠 Home"

    nav_choice = st.sidebar.radio("Navigation", nav_options, key="main_nav_V2", index=nav_idx, label_visibility="collapsed")

    if nav_choice != st.session_state.current_nav_choice:
        st.session_state.current_nav_choice = nav_choice
        # Reset detail view when navigating away from "My Plants" unless navigating to it
        if nav_choice != "🪴 My Plants": st.session_state.viewing_saved_plant_nickname = None
        # Optionally clear ID flow if navigating away from it and no image is loaded
        if nav_choice != "🆔 Identify New Plant" and not st.session_state.current_id_image_bytes:
            clear_current_identification_flow_data()
        st.experimental_rerun()
    
    st.sidebar.divider()
    st.sidebar.markdown("--- \n Made with 💚 for plants!")

    care_data = load_plant_care_data()

    if st.session_state.current_nav_choice == "🏠 Home":
        render_home_page(care_data)
    elif st.session_state.current_nav_choice == "🆔 Identify New Plant":
        render_identify_page(care_data)
    elif st.session_state.current_nav_choice == "🪴 My Plants":
        render_my_plants_page(care_data)

# --- Run the App ---
if __name__ == "__main__":
    main()