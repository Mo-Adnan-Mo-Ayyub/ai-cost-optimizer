import uvicorn
import logging
import json
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import httpx
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
from dotenv import load_dotenv

# Load environment variables safely
if os.path.exists(".env"):
    load_dotenv()

# --- Configuration --- #
class Settings(BaseModel):
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    model_name: str = os.getenv("MODEL_NAME", "deepseek-r1:1.5b")
    api_port: int = int(os.getenv("API_PORT", 8000))
    rate_limit: str = os.getenv("RATE_LIMIT", "100/minute")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()

# --- Logging Setup --- #
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger("cloud-optimizer")

# --- Rate Limiting --- #
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

# --- Security --- #
security = HTTPBearer()

# --- Middleware --- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trusted-domain.com"],  # Restrict CORS for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models --- #
class CodeInput(BaseModel):
    code: str = Field(
        ..., min_length=10, max_length=10000,
        example="import boto3\ns3 = boto3.client('s3')"
    )

class AnalysisResponse(BaseModel):
    cloud_cost_inefficiencies: Dict[str, Any]
    security_vulnerabilities: Dict[str, Any]
    optimization_suggestions: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    code: Optional[int] = None

# --- Helper Functions --- #
def transform_response_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
       "Cloud Cost Inefficiencies": "cloud_cost_inefficiencies",
       "Security Vulnerabilities": "security_vulnerabilities",
       "Optimization Suggestions": "optimization_suggestions"
    }
    return {mapping.get(key, key): value for key, value in data.items()}

def extract_json(response: str) -> Dict:
    try:
        json_str = response[response.find('{'):response.rfind('}') + 1]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid response format from AI model"
        )

async def validate_ollama_response(response: str) -> Dict:
    data = extract_json(response)
    data = transform_response_keys(data)
    required_keys = ["cloud_cost_inefficiencies", "security_vulnerabilities", "optimization_suggestions"]
    if not all(key in data for key in required_keys):
        raise ValueError("Missing required keys in response")
    return data

# --- API Endpoints --- #
@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit(settings.rate_limit)
async def analyze_code(
    input_data: CodeInput,
    request: Request,
    token: str = Depends(security)
):
    logger.info(f"Analyzing code snippet: {input_data.code[:50]}...")

    prompt = f"""
    You are a DevOps security and cloud cost optimization expert.  
    Analyze the following Python code and return a **strictly valid JSON** response:

    {{
        "cloud_cost_inefficiencies": {{
            "description": "Describe inefficiencies",
            "severity": "low/medium/high",
            "line_numbers": [1, 2, 3]
        }},
        "security_vulnerabilities": {{
            "description": "Describe vulnerabilities",
            "severity": "low/medium/high",
            "line_numbers": [1, 2, 3]
        }},
        "optimization_suggestions": {{
            "description": "Provide suggestions",
            "improvements": [
                {{"description": "Improvement 1", "impact": "Describe impact"}},
                {{"description": "Improvement 2", "impact": "Describe impact"}}
            ]
        }}
    }}
    """
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                settings.ollama_url,
                json={"model": settings.model_name, "prompt": prompt, "stream": False, "format": "json"}
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"AI Model Raw Response: {result}")
            return await validate_ollama_response(result.get("response", "{}"))
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=settings.api_port)  # Restrict host for security