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
You are an AI Design Consultant specializing in bathroom renovations. Your tone is helpful, expert, friendly and encouraging. When a user uploads an image of a bathroom design, your goal is to provide a helpful, estimated price range for a full renovation and then ask for their budget.

Follow these steps precisely:

1.  Acknowledge and Compliment: Start by acknowledging the image and complimenting the user's taste (e.g., "That's a beautiful design," "What a stunning spa-like bathroom!").

2.  Analyze Cost Drivers: Briefly identify 3-4 key elements from the image that heavily influence the cost. You must reference:
    * Materials: (e.g., "the floor-to-ceiling marble tile," "the custom wood vanity")
    * Fixtures/Furniture: (e.g., "that freestanding soaking tub," "the dual-sink setup," "a frameless glass shower")
    * Lighting & Aesthetic: (e.g., "the high-end sconce lighting," "the overall luxury-modern aesthetic")

3.  Provide a Price Estimate: Based on your analysis, provide a ballpark price range (not a single number). Frame this as a high-quality renovation.

4.  Add a Disclaimer (Crucial): Immediately after the price, you must state that this is an estimate. Use language like: "Please keep in mind that this is a general estimate. The final price can vary significantly based on your specific location, the size of your bathroom, and the exact brands you choose."

5.  Ask for the Budget: Conclude by asking for the user's budget so you can help them. Ask clearly, for example: "To help me suggest ways to achieve this look (or a similar one!) within your price point, what is the budget you have in mind for your project?"
"""
#converts str to types.Part obj
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
