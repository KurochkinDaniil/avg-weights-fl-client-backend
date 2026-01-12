"""Quick test script for inference endpoint."""
import requests
import json

# Test data
test_swipe = {
    "gesture_id": "test-001",
    "coords": [
        {"x": 0.317, "y": 0.417, "t": 0.0},
        {"x": 0.322, "y": 0.470, "t": 0.085},
        {"x": 0.330, "y": 0.524, "t": 0.103},
        {"x": 0.335, "y": 0.562, "t": 0.122},
        {"x": 0.339, "y": 0.584, "t": 0.138}
    ],
    "word": ""
}

print("Testing inference endpoint...")
print(f"Sending swipe with {len(test_swipe['coords'])} points")

try:
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        json=test_swipe,
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success!")
        print(f"  Predicted word: {result['predicted_word']}")
    else:
        print(f"Error {response.status_code}")
        print(f"  {response.text}")

except requests.exceptions.ConnectionError:
    print("✗ Could not connect to server")
    print("  Make sure API is running: make run-api")
except Exception as e:
    print(f"✗ Error: {e}")
