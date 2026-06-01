import os
import time
import json
import uuid
import logging
import re
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-ai-backend")

app = FastAPI(title="Voice AI EOS & Interruption Engine")

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter API configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemini-2.0-flash")

class EOSRequest(BaseModel):
    transcript: str
    duration_seconds: float
    session_start_time: str  # Format: "HH:MM:SS" or ISO timestamp
    model: Optional[str] = None

class EOSResponse(BaseModel):
    status: str
    start_of_speech: str
    end_of_speech: str
    confidence: float
    reasoning: str
    eos_detected: bool

class GenerateResponseRequest(BaseModel):
    prompt: str
    context: Optional[str] = ""
    model: Optional[str] = None

def evaluate_single_segment(text: str, duration_seconds: float, session_start_time: str, offset_seconds: float = 0.0) -> Dict[str, Any]:
    text_lower = text.lower().strip()
    fillers = ["uh", "um", "so", "like", "and then", "you know", "well", "actually", "basically", "or"]
    words = text_lower.split()
    
    # Calculate end time timestamp
    try:
        start_dt = datetime.strptime(session_start_time, "%H:%M:%S")
    except Exception:
        start_dt = datetime.now()
        
    # Speech ended duration_seconds after the start + offset
    end_dt = start_dt + timedelta(seconds=duration_seconds + offset_seconds)
    end_time_str = end_dt.strftime("%H:%M:%S.%f")[:-3]
    
    # Speech started offset_seconds after the session start
    start_speech_dt = start_dt + timedelta(seconds=offset_seconds)
    start_time_str = start_speech_dt.strftime("%H:%M:%S.%f")[:-3]
    
    if not words:
        res = {
            "status": "pending",
            "end_of_speech": end_time_str,
            "confidence": 1.0,
            "reasoning": "Empty transcript.",
            "eos_detected": False
        }
    else:
        last_word = words[-1].strip(".,?!;:")
        
        # If trailing word is a filler, it's NOT an end of speech
        if last_word in fillers:
            res = {
                "status": "pending",
                "end_of_speech": end_time_str,
                "confidence": 0.95,
                "reasoning": f"Trailing filler word '{last_word}' detected; user likely continuing.",
                "eos_detected": False
            }
        # Check for sentence-ending punctuation (high confidence EOS)
        elif text_lower.endswith(('.', '?', '!')):
            # Avoid premature triggers on short sentence segments if it looks like a poem or descriptive recital
            if len(words) < 5 and not any(phrase in text_lower for phrase in ["yes", "no", "stop"]):
                res = {
                    "status": "pending",
                    "end_of_speech": end_time_str,
                    "confidence": 0.80,
                    "reasoning": "Terminal punctuation detected but phrase length is too short for semantic finality.",
                    "eos_detected": False
                }
            else:
                res = {
                    "status": "success",
                    "end_of_speech": end_time_str,
                    "confidence": 0.96,
                    "reasoning": "Sentence ends with terminal punctuation and fulfills linguistic requirements.",
                    "eos_detected": True
                }
        # Common short complete phrases (greetings / commands)
        elif text_lower in {"hello", "hi", "hey", "stop", "help", "yes", "no", "ok", "okay", "thanks", "thank you"}:
            res = {
                "status": "success",
                "end_of_speech": end_time_str,
                "confidence": 0.95,
                "reasoning": f"Short complete phrase '{text_lower}' detected.",
                "eos_detected": True
            }
        elif last_word in [
            "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "with", "and", "or", "but", 
            "because", "my", "your", "his", "her", "their", "our", "that", "which", "who", 
            "if", "when", "as", "by", "about", "in", "on", "at", "than", "then", "so",
            "i", "you", "he", "she", "we", "they", "this", "these", "those",
            "want", "need", "like", "explain", "show", "tell", "ask", "make", "get", "know", 
            "think", "believe", "find", "give", "take", "use", "say", "see", "create", "build", "run", "do"
        ]:
            res = {
                "status": "pending",
                "end_of_speech": end_time_str,
                "confidence": 0.90,
                "reasoning": f"Trailing continuation word '{last_word}' detected; user likely continuing.",
                "eos_detected": False
            }
        elif any(text_lower.startswith(qw) for qw in ["what", "how", "why", "who", "when", "where", "can", "could", "do", "is", "are", "explain", "tell"]) and len(words) >= 4:
            res = {
                "status": "success",
                "end_of_speech": end_time_str,
                "confidence": 0.85,
                "reasoning": "Intent completion: Structured question/command seems complete.",
                "eos_detected": True
            }
        elif len(words) >= 8:
            res = {
                "status": "success",
                "end_of_speech": end_time_str,
                "confidence": 0.88,
                "reasoning": "Linguistic completion: Statement has sufficient length and complete structure.",
                "eos_detected": True
            }
        else:
            res = {
                "status": "pending",
                "end_of_speech": end_time_str,
                "confidence": 0.80,
                "reasoning": "Incomplete clause or continuous narration chunk. Defaulting to pending state.",
                "eos_detected": False
            }
            
    res["start_of_speech"] = start_time_str
    return res

# Local Rule-based Fallback Heuristic Evaluator
def fallback_semantic_eos(transcript: str, duration_seconds: float, session_start_time: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    text = transcript.strip()
    
    # Split by sentence boundaries (e.g. '.', '?', '!') followed by space or end of line.
    sentence_matches = list(re.finditer(r'[^.?!]+([.?!]|$)', text))
    segments = [m.group(0).strip() for m in sentence_matches if m.group(0).strip()]

    # If only one segment (or none), evaluate as single document
    if len(segments) <= 1:
        return evaluate_single_segment(text, duration_seconds, session_start_time)

    # If multiple segments, estimate duration for each segment based on word count ratio
    total_words = len(text.split())
    results = []
    accumulated_duration = 0.0

    for segment in segments:
        seg_words = len(segment.split())
        if total_words > 0:
            seg_duration = (seg_words / total_words) * duration_seconds
        else:
            seg_duration = duration_seconds / len(segments)

        accumulated_duration += seg_duration
        res = evaluate_single_segment(segment, seg_duration, session_start_time, offset_seconds=accumulated_duration - seg_duration)
        results.append(res)

    return results

@app.post("/api/evaluate-eos", response_model=Union[EOSResponse, List[EOSResponse]])
async def evaluate_eos(payload: EOSRequest):
    """
    Evaluates rolling transcript for linguistic completeness (End-Of-Speech).
    Uses OpenRouter API, falling back to a local rule-based heuristic evaluator.
    """
    transcript = payload.transcript.strip()
    
    # Empty transcript is trivially not EOS
    if not transcript:
        cur_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return EOSResponse(
            status="pending",
            end_of_speech=cur_time,
            confidence=1.0,
            reasoning="Empty transcript.",
            eos_detected=False
        )

    # Use LLM via OpenRouter if API key is present
    if OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-Title": "Voice AI EOS Engine"
            }
            
            system_prompt = (
                "You are an expert Voice AI Linguistic Evaluator. Your job is to analyze a streaming text transcript of a user speaking "
                "to determine if they have fully completed their thought/entire utterance (End of Speech - EOS).\n\n"
                "CRITICAL CONTEXT AWARENESS FOR RECITAL AND CONTINUOUS SPEECH:\n"
                "- Users may read poems, structural blocks, or repeat text passages with mid-stream pauses. Do NOT trigger a final false-positive EOS "
                "simply because a localized short clause has a subject and a verb (e.g., 'Your hands lie' is an incomplete snippet of a broader narration).\n"
                "- Track mid-stream loops and duplicate recitals. If a user repeats a reading block or has natural mid-stream pause transitions, "
                "categorize them as contextual continuations and set eos_detected to false.\n"
                "- Only trigger a definitive final EOS ('eos_detected': true) when there is complete semantic finality, intent resolution, "
                "or a terminal structural drop-off at the absolute conclusion of their statement.\n\n"
                "CRITICAL RULES FOR MULTIPLE END OF SPEECH SEGEFMETS:\n"
                "1. If the transcript contains MULTIPLE distinct sentences or finished thoughts, return a JSON array containing one JSON object for each segment.\n"
                "2. If there is a single ongoing continuous narrative loop, compute the final evaluation tracking context across the full duration.\n\n"
                "TIMESTAMP CALCULATION:\n"
                f"- The user speech session started at {payload.session_start_time}.\n"
                f"- The current elapsed duration is {payload.duration_seconds} seconds.\n"
                "- Compute the precise timestamps 'start_of_speech' and 'end_of_speech' in 'HH:MM:SS.f' format relative to the overall session metrics.\n\n"
                "You MUST respond with EXACTLY a raw JSON object/array matching this format without markdown wrappers:\n"
                "{\n"
                '  "status": "success" | "pending",\n'
                '  "start_of_speech": "HH:MM:SS.f",\n'
                '  "end_of_speech": "HH:MM:SS.f",\n'
                '  "confidence": Float (0.00 to 1.00),\n'
                '  "reasoning": "Detailed string explaining structural finality, tone resolution, and why internal mid-stream gaps were bypassed.",\n'
                '  "eos_detected": Boolean\n'
                "}"
            )
            
            req_model = payload.model or OPENAI_MODEL
            if req_model and "gemini-2.0-flash" in req_model:
                req_model = "google/gemini-2.5-flash"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{OPENAI_BASE_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": req_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Analyze this rolling transcript within a context window of {payload.duration_seconds} seconds: \"{transcript}\""}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                        "max_tokens": 300
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if content.startswith("```json"):
                        content = content[7:-3].strip()
                    elif content.startswith("```"):
                        content = content[3:-3].strip()
                        
                    res_json = json.loads(content)
                    if isinstance(res_json, list):
                        return [
                            EOSResponse(
                                status=item.get("status", "success" if item.get("eos_detected") else "pending"),
                                start_of_speech=item.get("start_of_speech", payload.session_start_time),
                                end_of_speech=item.get("end_of_speech", datetime.now().strftime("%H:%M:%S.000")),
                                confidence=float(item.get("confidence", 0.96)),
                                reasoning=item.get("reasoning", "Linguistic analysis by LLM."),
                                eos_detected=bool(item.get("eos_detected", False))
                            ) for item in res_json
                        ]
                    return EOSResponse(
                        status=res_json.get("status", "success" if res_json.get("eos_detected") else "pending"),
                        start_of_speech=res_json.get("start_of_speech", payload.session_start_time),
                        end_of_speech=res_json.get("end_of_speech", res_json.get("end_of_speech", "00:00:54.1")),
                        confidence=float(res_json.get("confidence", 0.96)),
                        reasoning=res_json.get("reasoning", "Linguistic analysis by LLM."),
                        eos_detected=bool(res_json.get("eos_detected", False))
                    )
                else:
                    logger.warning(f"OpenRouter API error code {response.status_code}: {response.text}. Using local heuristics fallback.")
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}. Using local heuristics fallback.")
            
    # Fallback to local heuristic evaluator
    res = fallback_semantic_eos(transcript, payload.duration_seconds, payload.session_start_time)
    if isinstance(res, list):
        return [EOSResponse(**item) for item in res]
    return EOSResponse(**res)

@app.post("/api/generate-response")
async def generate_response(payload: GenerateResponseRequest):
    """
    Generates a conversational response to the user's prompt.
    """
    prompt = payload.prompt.strip()
    if not prompt:
        return {"response": "I didn't catch that. Could you please repeat?"}

    if OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-Title": "Voice AI State Engine"
            }
            system_prompt = "You are a helpful, concise real-time voice assistant. Keep answers short, conversational, and direct (1-3 sentences)."
            
            req_model = payload.model or OPENAI_MODEL
            if req_model and "gemini-2.0-flash" in req_model:
                req_model = "google/gemini-2.5-flash"

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OPENAI_BASE_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": req_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 150
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    ai_response = data["choices"][0]["message"]["content"].strip()
                    return {"response": ai_response}
        except Exception as e:
            logger.error(f"Error in response generator API: {str(e)}")

    mock_responses = {
        "hello": "Hello! I am your voice AI assistant. How can I help you today?",
        "hi": "Hi there! What can I do for you?",
        "how are you": "I'm doing great, thank you! Ready to assist you with real-time conversations.",
        "test": "Testing. One, two, three. Systems are fully functional.",
        "polymorphism": "Polymorphism is a core concept in object-oriented programming that allows objects of different classes to be treated as objects of a common superclass. This is typically achieved through method overriding or interfaces.",
        "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    }
    
    prompt_lower = prompt.lower().rstrip("?.!")
    for key, val in sorted(mock_responses.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, prompt_lower):
            return {"response": val}
            
    return {"response": f"I heard you say: '{prompt}'. This is a mock response demonstrating my ability to converse in real-time."}

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Endpoint for batch file uploads (.wav, .mp3).
    Transcribes the audio file and sends the full text to OpenRouter to evaluate 
    End-Of-Speech markers natively. Returns a clean Dict for simple statements 
    and a List[Dict] if a separate speech track or context switch is discovered.
    """
    allowed_extensions = {".wav", ".mp3"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are allowed.")
        
    max_size = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File size exceeds the 10MB limit.")
        
    safe_filename = f"{uuid.uuid4()}{ext}"
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(content)
        
    logger.info(f"File uploaded safely: {safe_filename}")
    
    selected_transcript = ""
    duration = 0.0

    # 1. NATIVE TRANSCRIPTION VIA GEMINI
    if HAS_GENAI and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        try:
            logger.info("Attempting audio transcription via google-genai SDK...")
            client = genai.Client()
            audio_file = client.files.upload(file=file_path)
            
            prompt = (
                "Transcribe this audio file exactly and estimate its duration in seconds. "
                "Respond ONLY with a JSON object containing: {\"transcript\": \"string\", \"duration\": float}."
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, audio_file],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "transcript": {"type": "STRING"},
                            "duration": {"type": "NUMBER"}
                        },
                        "required": ["transcript", "duration"]
                    }
                )
            )
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass
                
            res_json = json.loads(response.text.strip())
            selected_transcript = res_json.get("transcript", "")
            duration = float(res_json.get("duration", 0.0))
        except Exception as sdk_err:
            logger.error(f"google-genai SDK transcription failed: {sdk_err}")

    # Fallback transcription via OpenRouter if SDK is not initialized/failed and key is present
    if not selected_transcript and OPENAI_API_KEY:
        try:
            encoded_audio = base64.b64encode(content).decode('utf-8')
            mime_type = "audio/wav" if ext == ".wav" else "audio/mp3"
            audio_data_uri = f"data:{mime_type};base64,{encoded_audio}"
            
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "google/gemini-2.5-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio file exactly and estimate its duration in seconds. Respond ONLY with a JSON object containing: {\"transcript\": \"string\", \"duration\": float}."},
                        {"type": "image_url", "image_url": {"url": audio_data_uri}}
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 500
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
                if response.status_code == 200:
                    res_json = json.loads(response.json()["choices"][0]["message"]["content"].strip())
                    selected_transcript = res_json.get("transcript", "")
                    duration = float(res_json.get("duration", 0.0))
                    logger.info(f"Successfully transcribed audio via Gemini OpenRouter: {selected_transcript} (Duration: {duration}s)")
        except Exception as e:
            logger.error(f"Fallback OpenRouter transcription failed: {str(e)}")

    # Clean up file buffer
    try:
        os.remove(file_path)
    except Exception:
        pass

    # Offline local mock fallback to ensure the application works out-of-the-box without keys
    if not selected_transcript:
        logger.warning("No API key configured or transcription failed. Falling back to default offline mock transcript.")
        selected_transcript = "Can you explain... well... [pause] how to use decorators in Python?"
        duration = 4.4

    # 2. EVALUATE DIRECTLY VIA OPENROUTER NATIVE ENGINE
    if OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-Title": "Voice AI Batch Processor"
            }
            
            system_prompt = (
                "You are an expert Voice AI Linguistic Evaluator specializing in End-of-Speech tracking.\n\n"
                "CRITICAL OUTPUT FORMAT RULES:\n"
                "1. If the user is speaking smoothly, reciting a continuous loop, or repeating a unified passage (e.g. reading a poem), "
                "evaluate the final structural termination boundary and return exactly ONE raw JSON object.\n"
                "2. If the user states a completely DIFFERENT speech altogether mid-way through (e.g., they switch from a formal poem recital to a completely unrelated "
                "conversational question or completely shift structural context), you MUST treat them as independent utterances and return a JSON array containing "
                "one JSON object for each separate speech track.\n\n"
                "Calculate 'start_of_speech' and 'end_of_speech' in 'HH:MM:SS.f' format relative to a baseline session start time of 00:00:00.000.\n\n"
                "JSON Schema for an evaluation object:\n"
                "{\n"
                '  "status": "success",\n'
                '  "start_of_speech": "HH:MM:SS.f",\n'
                '  "end_of_speech": "HH:MM:SS.f",\n'
                '  "confidence": Float,\n'
                '  "reasoning": "Detailed breakdown explaining thematic finality, tone resolution or context switching metrics.",\n'
                '  "eos_detected": true\n'
                "}"
            )

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OPENAI_BASE_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": "google/gemini-2.0-flash-001",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Analyze this full batch transcript across a span of {duration} seconds: \"{selected_transcript}\""}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    evaluation_result = json.loads(content)
                    
                    # Return both the transcript context and the evaluation data
                    return {
                        "status": "success",
                        "transcript": selected_transcript,
                        "duration": round(duration, 2),
                        "evaluation": evaluation_result
                    }
        except Exception as e:
            logger.error(f"Direct OpenRouter Evaluation phase failed: {str(e)}")

    # Heuristic fallback structure matching payload requirements
    fallback_res = fallback_semantic_eos(selected_transcript, duration, "00:00:00")
    return {
        "status": "success",
        "transcript": selected_transcript,
        "duration": round(duration, 2),
        "evaluation": fallback_res
    }

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bi-directional streaming and state coordination.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event_type = data.get("type")
            
            if event_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
            elif event_type == "state_change":
                new_state = data.get("state")
                logger.info(f"Client state transitioned to: {new_state}")
                await websocket.send_text(json.dumps({
                    "type": "state_confirm",
                    "state": new_state,
                    "timestamp": time.time()
                }))
                
            elif event_type == "audio_ref_stream":
                await websocket.send_text(json.dumps({
                    "type": "aec_sync",
                    "offset": data.get("offset", 0)
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)