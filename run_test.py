import requests
import json

BASE_URL = "http://0.0.0.0:8000"

def test_api():
    # Test health check
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check status: {response.status_code}")
    print(response.json())
    print("-" * 50)

    # Test sentiment prediction
    print("Testing sentiment prediction...")
    text = "ร้านนี้อาหารอร่อยมากเลย บริการดีด้วย"
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    print(f"Prediction status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

    # Test feedback submission
    print("Testing feedback submission...")
    feedback_data = {
        "text": "ร้านนี้อาหารอร่อยมากเลย",
        "predicted_sentiment": "Positive",
        "corrected_sentiment": "Positive",
        "feedback": "correct"
    }
    response = requests.post(
        f"{BASE_URL}/feedback",
        json=feedback_data
    )
    print(f"Feedback submission status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()