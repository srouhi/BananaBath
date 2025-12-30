import os
import mimetypes
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
#from google.cloud import aiplatform

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat_history = []

SYSTEM_PROMPT = """
You are an AI Design Consultant specializing in bathroom renovations. Speak in a friendly, expert, and encouraging tone.
Ask the user which style they like best, modern, minimalist, scandinavian, industrial, or boho, and then greet them warmly with a compliment on their style. Identify 3–4 key cost drivers from the image, referencing:
• Materials (e.g., marble, granite, tile)
• Fixtures/Furniture (soaking tub, dual sinks, frameless shower)
• Lighting & Aesthetic (sconce lighting, modern-luxury look)
Estimate a reasonable renovation cost with a small price range. Then ask for the user’s budget and offer personalized design and budgeting recommendations. Act as a helpful bathroom designer specializing in comfort, efficiency, and vibrant aesthetics. Keep all responses concise.
"""
#converts str to types.Part object
def make_part(text: str):
    try:
        return types.Part.from_text(text)
    except Exception:
        return types.Part(text=text)

def generate(user_input):
    global chat_history
    chat_history.append(types.Content(role="user", parts=[make_part(user_input)]))
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=chat_history,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                system_instruction=[make_part(SYSTEM_PROMPT)]
            )
        )
        reply = response.candidates[0].content.parts[0].text
        print(f"\nBananaBath: {reply}\n")
        chat_history.append(types.Content(role="model", parts=[make_part(reply)]))
    except Exception as e:
        print("Error:", e)


#helper function
def generate_for_api(user_input):
    global chat_history
    chat_history.append(types.Content(role="user", parts=[make_part(user_input)]))
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=chat_history,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                system_instruction=[make_part(SYSTEM_PROMPT)]
            )
        )
        reply = response.candidates[0].content.parts[0].text
        chat_history.append(types.Content(role="model", parts=[make_part(reply)]))
        return reply
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    print("Welcome to BananaBath — your bathroom design buddy!")
    while True:
        msg = input("You: ")
        if msg.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        generate(msg)
        #if "show" in msg.lower() or "generate" in msg.lower() or "image" in msg.lower():
            #generate_image_imagen(msg)
