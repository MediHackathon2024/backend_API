# FastAPI Project: Emergency Dispatch Conversation/Copilot AI

## Setup Instructions

### 1. Install Requirements

First, ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory of the project with the necessary environment variables. Example:

```env
BASE_URL=http://localhost:8000
GROQ_API_KEY=your_api_key
```

### 3. Run the Application

To start the FastAPI server, use the following command:

```bash
uvicorn main:app --reload
```

This will run the application locally and make it accessible at `http://127.0.0.1:8000`.

---

This simplified README covers the basic setup your team needs to get started quickly.
