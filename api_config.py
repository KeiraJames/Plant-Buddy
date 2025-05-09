import os
from dotenv import load_dotenv

load_dotenv()

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY")
GEMINI_API_KEY = "AIzaSyCd-6N83gfhMx_-D4WCAc-8iOFSb6hDJ_Q"
MONGO_URI = os.getenv("MONGO_URI")
