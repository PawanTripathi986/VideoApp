import requests

BASE_URL = "http://localhost:5000/api/cartoon"

def test_cartoon_script(service="default", script="Hello world"):
    url = f"{BASE_URL}/script"
    payload = {
        "script": script,
        "service": service
    }
    response = requests.post(url, json=payload)
    print(f"Test /script with service={service}: Status {response.status_code}")
    print(response.json())

def test_cartoon_url():
    url = f"{BASE_URL}/url"
    payload = {
        "url": "https://example.com/sample_video.mp4"
    }
    response = requests.post(url, json=payload)
    print(f"Test /url: Status {response.status_code}")
    print(response.json())

def test_cartoon_live():
    url = f"{BASE_URL}/live"
    payload = {
        "stream_url": "https://example.com/live_stream"
    }
    response = requests.post(url, json=payload)
    print(f"Test /live: Status {response.status_code}")
    print(response.json())

if __name__ == "__main__":
    # Test default service
    test_cartoon_script()
    # Test RunwayML service
    test_cartoon_script(service="runwayml")
    # Test Synthesia service
    test_cartoon_script(service="synthesia")
    # Test OpenAI service
    test_cartoon_script(service="openai")
    # Test free/open-source services
    test_cartoon_script(service="first_order_motion")
    test_cartoon_script(service="huggingface_spaces")

    test_cartoon_url()
    test_cartoon_live()
