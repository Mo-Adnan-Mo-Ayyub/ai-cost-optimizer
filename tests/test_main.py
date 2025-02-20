# tests/test_main.py
import pytest
import json
from fastapi.testclient import TestClient
from main import app, transform_response_keys, validate_ollama_response

client = TestClient(app)

def test_transform_response_keys():
    input_data = {
       "Cloud Cost Inefficiencies": {"description": "Test", "severity": "low", "line_numbers": [1]},
       "Security Vulnerabilities": {"description": "Test", "severity": "medium", "line_numbers": [2]},
       "Optimization Suggestions": {"description": "Test", "improvements": []}
    }
    output = transform_response_keys(input_data)
    assert "cloud_cost_inefficiencies" in output
    assert "security_vulnerabilities" in output
    assert "optimization_suggestions" in output

@pytest.mark.asyncio
async def test_validate_ollama_response_valid():
    # Create a valid JSON string that matches the required keys.
    valid_json = json.dumps({
        "Cloud Cost Inefficiencies": {"description": "Test", "severity": "low", "line_numbers": [1]},
        "Security Vulnerabilities": {"description": "Test", "severity": "medium", "line_numbers": [2]},
        "Optimization Suggestions": {"description": "Test", "improvements": []}
    })
    result = await validate_ollama_response(valid_json)
    assert result["cloud_cost_inefficiencies"]["description"] == "Test"

def test_analyze_code_endpoint():
    response = client.post(
        "/analyze",
        headers={"Authorization": "Bearer YOUR_TOKEN", "Content-Type": "application/json"},
        json={"code": "import boto3\ns3 = boto3.client('s3')"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "cloud_cost_inefficiencies" in data
    assert "security_vulnerabilities" in data
    assert "optimization_suggestions" in data
