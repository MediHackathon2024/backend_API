import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Groq client with the API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Add CORS middleware to allow requests from React frontend
origins = [
    "http://localhost:5173",  # React local app's URL
    "your frontend deployed url", # Frontend deployed app's URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from React
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define Pydantic model to handle the user input
class EmergencyQuery(BaseModel):
    dispatcher_input: str  # The query input from dispatcher
    scenario: str  # The scenario, e.g., "Cardiac Arrest"
    medical_recommendations: bool = False  # Flag to indicate if medical recommendations are requested

@app.post("/emergency-assistance/")
async def emergency_assistance(request: EmergencyQuery):
    # Construct the input for the model based on the dispatcher input and scenario
    prompt = f"""
    Scenario: {request.scenario}
    Dispatcher Input: {request.dispatcher_input}
    What should the dispatcher do?
    """

    # Add medical recommendation request if the flag is True
    if request.medical_recommendations:
        prompt += """
        Additionally, provide medical recommendations that the dispatcher should inform the caller of.
        These should include specific actions or steps the dispatcher can guide the caller through.
        """

    # Use Groq API to get model completion based on the constructed prompt
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",  # The model name, replace if necessary
        )

        # Get the response from the model
        response = chat_completion.choices[0].message.content

        # Clean up the response by removing unwanted characters (like newlines or numbered lists)
        cleaned_response = response.replace("\n\n", " ").replace("\n", " ").strip()

        # Return the cleaned response
        return {"response": cleaned_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

# For health check or testing
@app.get("/")
def read_root():
    return {"message": "QuickTriage Conversation AI is running!"}
