import os
from dotenv import load_dotenv

load_dotenv()

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
