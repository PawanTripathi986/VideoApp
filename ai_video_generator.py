import os
import requests

class AIVideoGenerator:
    def __init__(self):
        # Load Synthesia API key and endpoint from environment variables or config
        self.synthesia_api_key = os.getenv("SYNTHESIA_API_KEY")
        self.synthesia_endpoint = "https://api.synthesia.io/v1/videos"

    def generate_video(self, script):
        return self._generate_with_synthesia(script)

    def _generate_with_synthesia(self, script):
        if not self.synthesia_api_key:
            raise ValueError("Synthesia API key not configured")
        headers = {
            "Authorization": f"Bearer {self.synthesia_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "script": script,
            "voice": "en-US-Wavenet-D",  # example voice
            "avatar": "default"  # example avatar
        }
        response = requests.post(self.synthesia_endpoint, json=data, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Synthesia API error: {response.text}")
        result = response.json()
        video_url = result.get("video_url")
        import uuid
        unique_filename = f"synthesia_output_{uuid.uuid4()}.mp4"
        video_path = self._download_video(video_url, unique_filename)
        return video_path

    def _download_video(self, url, filename):
        if not url:
            raise ValueError("No video URL provided")
        local_path = os.path.join("output", filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_path
