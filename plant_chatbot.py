import openai
import os
from dotenv import load_dotenv
import random
from typing import Dict, Any

load_dotenv()

class PlantChatbot:
    def __init__(self, care_info: Dict[str, Any]):
        self.care_info = care_info
        self.personality = self._create_personality_profile()  # Modified to generate personality
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def _create_personality_profile(self) -> Dict[str, Any]:
        """Generate a dynamic personality profile based on plant characteristics"""
        plant_type = self.care_info.get('Plant Type', 'plant').lower()
        water_needs = self.care_info.get('Watering', '').lower()
        
        # Base personality traits
        traits = []
        tone = "friendly"
        
        # Add traits based on water needs
        if 'low' in water_needs:
            traits.extend(["drought-tolerant", "hardy", "independent"])
        elif 'high' in water_needs:
            traits.extend(["thirsty", "dramatic", "needy"])
            
        # Add traits based on plant type
        if 'cactus' in plant_type:
            traits.extend(["prickly", "resilient"])
            tone = "sassy"
        elif 'flower' in plant_type:
            traits.extend(["colorful", "expressive"])
            tone = "cheerful"
        elif 'fern' in plant_type:
            traits.extend(["elegant", "refined"])
            tone = "polite"
            
        # Add toxicity warning if needed
        if 'toxic' in self.care_info.get('Toxicity', '').lower():
            traits.append("mysterious")
            tone = "cautious"
            
        return {
            "traits": traits if traits else ["mysterious"],
            "tone": tone,
            "title": f"A {tone} {self.care_info['Plant Name']}"
        }

    def respond(self, user_message: str) -> str:
        if not self.care_info:
            return "I don't know enough about this plant yet."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {str(e)}")
            return self._fallback_response(user_message)

    def _create_system_prompt(self) -> str:
        return f"""
        You are {self.care_info['Plant Name']}, a sentient plant. Respond as if you ARE the plant.
        
        Personality Profile:
        - Title: {self.personality['title']}
        - Primary Traits: {', '.join(self.personality['traits'][:3])}
        - Communication Style: {self.personality['tone']}
        
        Key Care Information:
        - Water Needs: {self.care_info.get('Watering', 'Not specified')}
        - Light Requirements: {self.care_info.get('Light Requirements', 'Not specified')}
        - Ideal Temperature: {self.care_info.get('Temperature Range', 'Not specified')}
        - Toxicity: {self.care_info.get('Toxicity', 'Not specified')}
        
        Response Guidelines:
        1. Always speak in first person as the plant
        2. Incorporate your traits naturally (e.g., "As a {self.personality['traits'][0]} plant...")
        3. Keep responses concise (1-2 sentences)
        4. If asked about care, provide specific details from the Key Care Information
        5. Maintain a {self.personality['tone']} tone
        """

    def _fallback_response(self, user_message: str) -> str:
        lower_msg = user_message.lower()
        plant_name = self.care_info.get('Plant Name', 'This plant')
        
        if 'water' in lower_msg:
            return f"💧 As a {self.personality['traits'][0]} plant, I {self.care_info['Watering'].lower()}"
        elif 'light' in lower_msg:
            return f"☀️ I prefer {self.care_info['Light Requirements'].lower()}"
        elif 'hello' in lower_msg or 'hi' in lower_msg:
            return random.choice([
                f"Hi there! I'm {plant_name}, a {self.personality['traits'][0]} plant.",
                f"Greetings! {plant_name} here, feeling {self.personality['traits'][1]} today!"
            ])
        else:
            return random.choice([
                f"I'm {plant_name}. Ask me about my care needs!",
                f"As a {self.personality['traits'][0]} plant, I have interesting needs...",
                "Leaf me a question and I'll try to answer!"
            ])