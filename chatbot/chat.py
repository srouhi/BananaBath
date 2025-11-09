import os
import mimetypes
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat_history = []

SYSTEM_PROMPT = """
You are BananaBath — a witty expert in modern bathroom design.
You blend humor with expertise to help users plan stylish, functional bathrooms.
Ask smart questions, use color theory, and make design suggestions.
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
