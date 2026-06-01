import sys
import os
import json
import base64
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure the backend directory is in sys.path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main
from main import app

# Set dummy API keys so main.py triggers the client code paths
os.environ["OPENAI_API_KEY"] = "mock_openai_key"
os.environ["GEMINI_API_KEY"] = "mock_gemini_key"
main.OPENAI_API_KEY = "mock_openai_key"

class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

class MockAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def post(self, url, **kwargs):
        json_payload = kwargs.get("json", {})
        messages = json_payload.get("messages", [])
        
        # Check if the prompt is list-based (audio transcription request)
        if messages and isinstance(messages[0].get("content"), list):
            return MockResponse(200, {
                "choices": [{
                    "message": {
                        "content": '{"transcript": "Can you explain... well... [pause] how to use decorators in Python?", "duration": 4.4}'
                    }
                }]
            })
            
        prompt = messages[-1]["content"] if messages else ""
        
        if "Well, I would like to check" in str(prompt):
            content = '[{"status": "success", "start_of_speech": "12:00:00.000", "end_of_speech": "12:00:02.142", "confidence": 0.96, "reasoning": "Sentence ends with terminal punctuation.", "eos_detected": true}, {"status": "pending", "start_of_speech": "12:00:02.142", "end_of_speech": "12:00:02.500", "confidence": 0.95, "reasoning": "Trailing filler word.", "eos_detected": false}]'
        elif "Please show me the documentation" in str(prompt):
            content = '{"status": "success", "start_of_speech": "12:00:00.000", "end_of_speech": "12:00:04.100", "confidence": 0.96, "reasoning": "Sentence ends with terminal punctuation.", "eos_detected": true}'
        elif "Hello assistant" in str(prompt):
            content = '[{"status": "pending", "start_of_speech": "12:00:00.000", "end_of_speech": "12:00:01.000", "confidence": 0.8, "reasoning": "Short phrase.", "eos_detected": false}, {"status": "success", "start_of_speech": "12:00:01.000", "end_of_speech": "12:00:04.500", "confidence": 0.96, "reasoning": "Sentence ends with terminal punctuation.", "eos_detected": true}, {"status": "pending", "start_of_speech": "12:00:04.500", "end_of_speech": "12:00:06.000", "confidence": 0.8, "reasoning": "Short phrase.", "eos_detected": false}]'
        elif "explain polymorphism" in str(prompt):
            content = 'Polymorphism allows objects of different classes to respond to the same method call...'
        elif "Analyze this full batch transcript" in str(prompt):
            content = '{"status": "success", "start_of_speech": "00:00:00.000", "end_of_speech": "00:04:04.0", "confidence": 0.95, "reasoning": "The utterance concludes with a clear statement of intent.", "eos_detected": true}'
        else:
            content = '{"status": "pending", "start_of_speech": "00:00:00.000", "end_of_speech": "00:00:00.0", "confidence": 0.5, "reasoning": "Default mock response.", "eos_detected": false}'

        return MockResponse(200, {
            "choices": [{
                "message": {
                    "content": content
                }
            }]
        })

# Google GenAI SDK mock classes
class MockGenaiFiles:
    def upload(self, **kwargs):
        mock_file = MagicMock()
        mock_file.name = "mock_uploaded_file"
        return mock_file

    def delete(self, **kwargs):
        pass

class MockGenaiModels:
    def generate_content(self, **kwargs):
        mock_resp = MagicMock()
        mock_resp.text = '{"transcript": "Can you explain... well... [pause] how to use decorators in Python?", "duration": 4.4}'
        return mock_resp

class MockGenaiClient:
    def __init__(self, *args, **kwargs):
        self.files = MockGenaiFiles()
        self.models = MockGenaiModels()

# Set up the mocks in main module
main.genai.Client = MockGenaiClient
main.HAS_GENAI = True

@patch("main.httpx.AsyncClient", MockAsyncClient)
def run_tests():
    print("=== STARTING VOICE AI BACKEND VERIFICATION (WITH MOCKS) ===\n")
    client = TestClient(app)

    try:
        # Test 1: Pending state with fillers
        print("[+] Testing Semantic EOS endpoint (Pending state with fillers)...")
        res = client.post("/api/evaluate-eos", json={
            "transcript": "Well, I would like to check... uh...",
            "duration_seconds": 2.5,
            "session_start_time": "12:00:00"
        })
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("    Response:", data)
        t1_eos = data[-1]["eos_detected"] if isinstance(data, list) else data["eos_detected"]
        t1_status = data[-1]["status"] if isinstance(data, list) else data["status"]
        t1_sos = data[-1]["start_of_speech"] if isinstance(data, list) else data["start_of_speech"]
        assert t1_eos is False, "Expected eos_detected: False for filler word 'uh'"
        assert t1_status == "pending", "Expected status: pending for filler word"
        assert t1_sos is not None, "Expected start_of_speech to be present"
        print("    => Pass")

        # Test 2: Complete sentence with terminal punctuation
        print("\n[+] Testing Semantic EOS endpoint (Completed state with punctuation)...")
        res = client.post("/api/evaluate-eos", json={
            "transcript": "Please show me the documentation.",
            "duration_seconds": 4.1,
            "session_start_time": "12:00:00"
        })
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("    Response:", data)
        t2_eos = data[-1]["eos_detected"] if isinstance(data, list) else data["eos_detected"]
        t2_status = data[-1]["status"] if isinstance(data, list) else data["status"]
        t2_sos = data[-1]["start_of_speech"] if isinstance(data, list) else data["start_of_speech"]
        assert t2_eos is True, "Expected eos_detected: True for terminal punctuation"
        assert t2_status == "success", "Expected status: success"
        assert t2_sos == "12:00:00.000", f"Expected start_of_speech to be '12:00:00.000', got {t2_sos}"
        print("    => Pass")

        # Test 2b: Multiple sentences returning a list of documents
        print("\n[+] Testing Semantic EOS endpoint (Multiple End-Of-Speeches)...")
        res = client.post("/api/evaluate-eos", json={
            "transcript": "Hello assistant. Can you tell me what polymorphism is? Also explain decorators.",
            "duration_seconds": 6.0,
            "session_start_time": "12:00:00"
        })
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("    Response:", data)
        assert isinstance(data, list), "Expected response to be an array of documents for multiple End-Of-Speeches"
        assert len(data) >= 2, "Expected at least 2 EOS segments evaluated"
        assert all("start_of_speech" in item for item in data), "Expected start_of_speech in each segment"
        print("    => Pass")

        # Test 3: Response Generation
        print("\n[+] Testing Response Generation (Polymorphism)...")
        res = client.post("/api/generate-response", json={
            "prompt": "explain polymorphism"
        })
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("    Response:", data)
        assert "polymorphism" in data["response"].lower(), "Expected polymorphism definition"
        print("    => Pass")

        # Test 4: File Upload Limits & Type Checks (Negative Test)
        print("\n[+] Testing File Upload (Invalid Type - Text File)...")
        files = {'file': ('test.txt', b'some text content', 'text/plain')}
        res = client.post("/api/upload-audio", files=files)
        print(f"    Response status: {res.status_code}")
        assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
        print("    Response:", res.json())
        print("    => Pass")

        # Test 5: File Upload (Valid Type - WAV file)
        print("\n[+] Testing File Upload (Valid Type - WAV file)...")
        files = {'file': ('audio.wav', b'RIFFxxxxWAVEfmt xxxxdataxxxx', 'audio/wav')}
        res = client.post("/api/upload-audio", files=files)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("    Response:", data)
        is_list = isinstance(data, list)
        first_item = data[0] if is_list else data
        assert first_item.get("status") == "success", "Expected success status"
        assert "evaluation" in first_item, "Expected evaluation in response"
        first_eval = first_item["evaluation"][0] if isinstance(first_item["evaluation"], list) else first_item["evaluation"]
        assert "eos_detected" in first_eval, "Expected eos_detected in evaluation"
        assert "start_of_speech" in first_eval, "Expected start_of_speech in evaluation"
        print("    => Pass")



        print("\n===========================================")
        print(" ALL BACKEND VERIFICATIONS COMPLETED: PASS ")
        print("===========================================")
        return True

    except Exception as e:
        print(f"\n[!] Verification FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
