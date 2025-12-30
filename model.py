import os
import replicate
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware # Added for frontend

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Chatbot Image Generator",
    description="A backend API to generate images using Stable Diffusion and ControlNet via Replicate.",
)

# --- 2. Add CORS Middleware ---
# This is CRUCIAL for your friend's website to be able
# to call your backend from a different domain (e.g., localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For testing. Change to your frontend's domain later
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- 3. Define the Replicate ControlNet Models ---
# Store model IDs in a dictionary for easy access
CONTROLNET_MODELS = {
    "canny": "jagilley/controlnet-canny:505e50523d16bde5b948512e0d02f5e71420d750c8e03f5d75f2d011116c28f3",
    "openpose": "jagilley/controlnet-pose:2e97b1ab8f851b965f336683510006e88CED1E3350439A867946B70C00a85ac5",
    "depth": "jagilley/controlnet-depth:3b1a62013913f011130d2105d21c0e309531e21b7755866f5d52042f01f40e04",
    # Add other models here (e.g., 'scribble', 'hed')
}

# --- 4. Create the API Endpoint ---
@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    controlnet_type: str = Form("canny"), # 'canny', 'openpose', 'depth'
    file: UploadFile = File(...)
):
    """
    Receives a prompt and an IMAGE UPLOAD, then generates a new image
    using a ControlNet model on Replicate.
    """
    
    # Check if the API token is set
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HTTPException(
            status_code=500, 
            detail="REPLICATE_API_TOKEN environment variable not set."
        )

    # --- 5. Read and Encode the Uploaded File ---
    try:
        # Check if the uploaded file is an image
        mime_type = file.content_type
        if not mime_type or not mime_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (png, jpg, etc.).")

        print(f"Received upload: Prompt='{prompt}', Filename='{file.filename}', Type='{controlnet_type}'")

        # Read the file's binary data
        image_data = await file.read()
        
        # Encode the binary data to a Base64 string
        # and format it as a data URI for the Replicate API
        image_base64 = f"data:{mime_type};base64,{base64.b64encode(image_data).decode('utf-8')}"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or encode file: {e}")
    finally:
        await file.close() # Close the file

    # --- 6. Prepare and Call Replicate ---
    
    # Get the correct Replicate model ID
    model_id = CONTROLNET_MODELS.get(controlnet_type)
    if not model_id:
        raise HTTPException(status_code=400, detail=f"Invalid controlnet_type. Valid types are: {list(CONTROLNET_MODELS.keys())}")

    # This is the input payload for the Replicate API
    api_input = {
        "image": image_base64,       # The Base64 data URI
        "prompt": prompt,            # The text prompt
        "image_resolution": "512",
        # "detect_resolution": 512,  # Specific to Canny/Depth
        # "num_samples": 1,
        # "guidance_scale": 9,
        # "steps": 20,
        # "negative_prompt": "blurry, low quality"
    }

    # Clean up input for different model types
    if controlnet_type == "openpose":
       api_input.pop("detect_resolution", None)

    try:
        print(f"Calling Replicate with model: {model_id}")
        
        output = replicate.run(model_id, input=api_input)
        
        print(f"Replicate output: {output}")

        if isinstance(output, list) and len(output) > 0:
            # ControlNet models often return [generated_image, control_map]
            # We want the first one, which is the final image.
            generated_image_url = output[0]
            return {"new_image_url": generated_image_url}
        else:
            raise HTTPException(status_code=500, detail="Invalid response from Replicate.")

    except replicate.exceptions.ReplicateError as e:
        print(f"Error from Replicate: {e}")
        raise HTTPException(status_code=500, detail=f"Replicate API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- 7. Add a simple root endpoint for testing ---
@app.get("/")
def read_root():
    return {"message": "Image Generation Chatbot Backend is running."}