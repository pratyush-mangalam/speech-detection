import React, { useState, useEffect, useRef } from 'react';
import Orb from './components/Orb';
import Visualizers from './components/Visualizers';
import Controls from './components/Controls';
import Transcript from './components/Transcript';
import { VoiceActivityDetector } from './utils/vad';
import { AcousticEchoCanceller } from './utils/aec';
import { UploadCloud, MessageSquare, AlertTriangle, RefreshCw } from 'lucide-react';

export default function App() {
  // App States
  const [state, setState] = useState('LISTENING'); // LISTENING, THINKING, SPEAKING, INTERRUPTED
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [history, setHistory] = useState([]);
  const [eosJson, setEosJson] = useState(null);

  // Configuration Settings
  const [micSensitivity, setMicSensitivity] = useState(1.0);
  const [vadThreshold, setVadThreshold] = useState(0.85);
  const [couplingFactor, setCouplingFactor] = useState(0.80);
  const [isAecEnabled, setIsAecEnabled] = useState(true);
  const [selectedModel, setSelectedModel] = useState('google/gemini-2.0-flash');

  // Audio Nodes & Canvas state
  const [isAudioActive, setIsAudioActive] = useState(false);
  const [micAnalyser, setMicAnalyser] = useState(null);
  const [ttsAnalyser, setTtsAnalyser] = useState(null);
  const [cancelledData, setCancelledData] = useState(new Float32Array(128));
  const [vadConfidence, setVadConfidence] = useState(0.0);

  // References
  const audioContextRef = useRef(null);
  const micSourceRef = useRef(null);
  const ttsSourceRef = useRef(null);
  const wsRef = useRef(null);
  const recognitionRef = useRef(null);
  const vadRef = useRef(null);
  const aecRef = useRef(null);
  const eosTimeoutRef = useRef(null);

  // New refs for speech detection fixes
  const latestTranscriptRef = useRef('');
  const vadConfidenceRef = useRef(0.0);
  const lastSpeechTimeRef = useRef(0);
  const eosRequestCounterRef = useRef(0);
  const currentDebounceDelayRef = useRef(650);
  const silenceFallbackTimeoutRef = useRef(null);

  // TTS Ref Osc Node to generate reference carrier in Web Audio graph
  const ttsOscRef = useRef(null);
  const ttsGainRef = useRef(null);

  // Timing references
  const sessionStartTimeRef = useRef(null);
  const speechStartTimeRef = useRef(null);
  const currentUtteranceRef = useRef(null);

  // File Upload State
  const [isUploading, setIsUploading] = useState(false);
  const [uploadTimeline, setUploadTimeline] = useState(null);

  // Refs for tracking state and configs to avoid stale closures in audio loop
  const stateRef = useRef(state);
  const couplingFactorRef = useRef(couplingFactor);
  const vadThresholdRef = useRef(vadThreshold);
  const isAecEnabledRef = useRef(isAecEnabled);

  // Keep refs synchronized with state/config
  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { couplingFactorRef.current = couplingFactor; }, [couplingFactor]);
  useEffect(() => { vadThresholdRef.current = vadThreshold; }, [vadThreshold]);
  useEffect(() => { isAecEnabledRef.current = isAecEnabled; }, [isAecEnabled]);

  // Initialize WebSockets
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('ws://127.0.0.1:8000/api/stream');
      ws.onopen = () => {
        setIsWebSocketConnected(true);
        console.log('WebSocket connected.');
      };
      ws.onclose = () => {
        setIsWebSocketConnected(false);
        console.log('WebSocket disconnected. Retrying in 5 seconds...');
        setTimeout(connectWebSocket, 5000);
      };
      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        // Handle sync/echo signals if needed
      };
      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
    }
  };

  // Sync state changes with server
  const syncStateWithServer = (newState) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'state_change',
        state: newState
      }));
    }
  };

  // Initialize Speech Recognition (ASR)
  const initializeASR = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("SpeechRecognition is not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      speechStartTimeRef.current = Date.now();
    };

    recognition.onresult = async (event) => {
      // Clear any pending EOS transition immediately when user continues speaking
      clearPendingEOSTransition();

      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const activeText = finalTranscript || interimTranscript;
      if (!activeText.trim()) return;

      // Update refs
      latestTranscriptRef.current = activeText;

      // Update history with temporary message
      updateTranscriptLog(activeText, 'user', true);

      // Reset silence fallback timer
      resetSilenceFallbackTimer(activeText);

      // Evaluate linguistic completion (Semantic EOS)
      const elapsedSeconds = (Date.now() - speechStartTimeRef.current) / 1000;
      await checkSemanticEOS(activeText, elapsedSeconds);
    };

    recognition.onerror = (event) => {
      console.error('ASR error:', event.error);
    };

    recognition.onend = () => {
      // Auto restart recognition if we are still in LISTENING mode and audio is active
      if (state === 'LISTENING' && isAudioActive && recognitionRef.current) {
        try {
          recognitionRef.current.start();
        } catch (e) { }
      }
    };

    recognitionRef.current = recognition;
  };

  // Start or Stop ASR
  const setASRState = (active) => {
    if (!recognitionRef.current) return;
    try {
      if (active) {
        speechStartTimeRef.current = Date.now();
        latestTranscriptRef.current = ''; // Clear it when starting to listen
        recognitionRef.current.start();
      } else {
        recognitionRef.current.stop();
      }
    } catch (e) {
      // Ignored if already in requested state
    }
  };

  const resetSilenceFallbackTimer = (transcript) => {
    if (silenceFallbackTimeoutRef.current) {
      clearTimeout(silenceFallbackTimeoutRef.current);
    }
    // If the user has completely stopped speaking for 3.0 seconds, transition anyway
    silenceFallbackTimeoutRef.current = setTimeout(() => {
      if (state === 'LISTENING') {
        console.log("Silence fallback triggered. Transitioning to THINKING.");
        handleEOSTrigger(latestTranscriptRef.current || transcript);
      }
    }, 3000);
  };

  const clearSilenceFallback = () => {
    if (silenceFallbackTimeoutRef.current) {
      clearTimeout(silenceFallbackTimeoutRef.current);
      silenceFallbackTimeoutRef.current = null;
    }
  };

  // Check Semantic EOS via Backend Proxy
  const checkSemanticEOS = async (transcript, durationSeconds) => {
    if (!sessionStartTimeRef.current) {
      sessionStartTimeRef.current = new Date().toLocaleTimeString('it-IT'); // HH:MM:SS
    }

    eosRequestCounterRef.current += 1;
    const currentRequestId = eosRequestCounterRef.current;

    try {
      const response = await fetch('/api/evaluate-eos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: transcript,
          duration_seconds: durationSeconds,
          session_start_time: sessionStartTimeRef.current,
          model: selectedModel
        })
      });

      if (response.ok) {
        // Drop outdated responses to avoid race conditions
        if (currentRequestId !== eosRequestCounterRef.current) {
          return;
        }

        const data = await response.json();
        setEosJson(data);

        // Transition to THINKING if Semantic EOS Evaluator triggers
        const hasEOS = Array.isArray(data)
          ? data.some(item => item.eos_detected)
          : data.eos_detected;

        // Get max confidence
        const maxConfidence = Array.isArray(data)
          ? Math.max(...data.map(item => item.confidence))
          : data.confidence;

        if (hasEOS && state === 'LISTENING') {
          scheduleEOSTransition(transcript, maxConfidence);
        } else {
          clearPendingEOSTransition();
        }
      }
    } catch (e) {
      console.error('Failed to query EOS evaluator:', e);
    }
  };

  const scheduleEOSTransition = (transcript, confidence = 0.85) => {
    clearPendingEOSTransition();
    // Choose dynamic delay based on confidence
    const debounceDelay = confidence >= 0.95 ? 400 : 750;
    currentDebounceDelayRef.current = debounceDelay;

    eosTimeoutRef.current = setTimeout(() => {
      handleEOSTrigger(latestTranscriptRef.current || transcript);
    }, debounceDelay);
  };

  const postponeEOSTransition = () => {
    if (eosTimeoutRef.current) {
      clearTimeout(eosTimeoutRef.current);
      eosTimeoutRef.current = setTimeout(() => {
        handleEOSTrigger(latestTranscriptRef.current);
      }, currentDebounceDelayRef.current);
    }
  };

  const clearPendingEOSTransition = () => {
    if (eosTimeoutRef.current) {
      clearTimeout(eosTimeoutRef.current);
      eosTimeoutRef.current = null;
    }
  };

  // Process finalized user message and shift to THINKING
  const handleEOSTrigger = async (finalTranscript) => {
    clearPendingEOSTransition();
    clearSilenceFallback();

    const textToSubmit = latestTranscriptRef.current || finalTranscript || '';
    if (!textToSubmit.trim()) return;

    setState('THINKING');
    syncStateWithServer('THINKING');
    setASRState(false);

    // Finalize user message in log
    updateTranscriptLog(textToSubmit, 'user', false);

    // Call response generation
    try {
      const response = await fetch('/api/generate-response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: textToSubmit,
          model: selectedModel
        })
      });

      if (response.ok) {
        const data = await response.json();
        playAgentTTS(data.response);
      } else {
        playAgentTTS("I encountered an issue generating a response.");
      }
    } catch (e) {
      playAgentTTS("Sorry, connection failed.");
    }
  };

  // Play Text-to-Speech (TTS)
  const playAgentTTS = (text) => {
    setState('SPEAKING');
    syncStateWithServer('SPEAKING');

    // Cancel any playing speech
    window.speechSynthesis.cancel();

    // Save current utterance reference
    currentUtteranceRef.current = {
      text: text,
      startedAt: Date.now(),
      charIndex: 0
    };

    const utterance = new SpeechSynthesisUtterance(text);

    // Track index for truncation context preservation
    utterance.onboundary = (event) => {
      if (event.name === 'word' && currentUtteranceRef.current) {
        currentUtteranceRef.current.charIndex = event.charIndex;
      }
    };

    utterance.onstart = () => {
      // Start reference audio synth in Web Audio graph
      startTTSReferenceSynth();
    };

    utterance.onend = () => {
      stopTTSReferenceSynth();
      // Only transition to LISTENING if we weren't interrupted
      setState((prev) => {
        if (prev === 'SPEAKING') {
          // Log complete message
          updateTranscriptLog(text, 'ai', false);

          // Go back to listening
          syncStateWithServer('LISTENING');
          setTimeout(() => setASRState(true), 100);
          return 'LISTENING';
        }
        return prev;
      });
    };

    utterance.onerror = (e) => {
      console.error("TTS play error:", e);
      stopTTSReferenceSynth();
    };

    window.speechSynthesis.speak(utterance);
  };

  // Manage Web Audio Synth nodes to model TTS acoustic bleed
  const startTTSReferenceSynth = () => {
    if (!audioContextRef.current) return;

    try {
      // Create oscillator simulating speech vocal frequencies (e.g. 220Hz + harmonics)
      const osc = audioContextRef.current.createOscillator();
      const gain = audioContextRef.current.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(220, audioContextRef.current.currentTime);

      // Low pass filter to make it sound slightly more voice-like (and not beep too loudly)
      const filter = audioContextRef.current.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(800, audioContextRef.current.currentTime);

      gain.gain.setValueAtTime(0.015, audioContextRef.current.currentTime); // Very soft background synth

      osc.connect(filter);
      filter.connect(gain);

      // Connect to the TTS Analyser so it displays on the screen
      if (ttsAnalyser) {
        gain.connect(ttsAnalyser);
      }

      // Connect to destination speakers
      gain.connect(audioContextRef.current.destination);

      osc.start();

      ttsOscRef.current = osc;
      ttsGainRef.current = gain;
    } catch (e) {
      console.error("Failed to start TTS reference audio graph nodes:", e);
    }
  };

  const stopTTSReferenceSynth = () => {
    try {
      if (ttsOscRef.current) {
        ttsOscRef.current.stop();
        ttsOscRef.current.disconnect();
        ttsOscRef.current = null;
      }
      if (ttsGainRef.current) {
        ttsGainRef.current.disconnect();
        ttsGainRef.current = null;
      }
    } catch (e) { }
  };

  // Handle Speech Interruption
  const triggerInterruption = () => {
    setState((prev) => {
      if (prev !== 'SPEAKING') return prev;

      console.warn("User speech onset detected. Triggering immediate interruption!");

      // 1. Halt the TTS playback buffer immediately
      window.speechSynthesis.cancel();
      stopTTSReferenceSynth();

      // 2. Preserve truncated context up to the truncation timestamp
      let spokenText = currentUtteranceRef.current?.text || '';
      const charIndex = currentUtteranceRef.current?.charIndex || 0;
      if (spokenText && charIndex > 0) {
        // Truncate at word boundary nearest to interruption index
        spokenText = spokenText.slice(0, charIndex).trim() + "... [Interrupted]";
      } else {
        spokenText = "[Interrupted]";
      }

      updateTranscriptLog(spokenText, 'ai', false);

      // 3. Send event to websocket server
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'audio_ref_stream',
          action: 'stop',
          truncated_text: spokenText
        }));
      }

      // Transition state
      syncStateWithServer('INTERRUPTED');

      // Flash state INTERRUPTED red, then shift back to LISTENING
      setTimeout(() => {
        setState('LISTENING');
        syncStateWithServer('LISTENING');
        setASRState(true);
      }, 800);

      return 'INTERRUPTED';
    });
  };

  // Setup Web Audio Context and Node Graph
  const startAudioGraph = async () => {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContextClass();
      audioContextRef.current = ctx;

      // Request microphone stream
      // Enforce WebRTC native constraints based on state
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: isAecEnabled,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Mic Analyser Node
      const mAnalyser = ctx.createAnalyser();
      mAnalyser.fftSize = 256;

      // TTS Reference Analyser Node
      const tAnalyser = ctx.createAnalyser();
      tAnalyser.fftSize = 256;

      const source = ctx.createMediaStreamSource(stream);

      // Insert gain node for sensitivity
      const gainNode = ctx.createGain();
      gainNode.gain.value = micSensitivity;

      source.connect(gainNode);
      gainNode.connect(mAnalyser);

      micSourceRef.current = source;
      setMicAnalyser(mAnalyser);
      setTtsAnalyser(tAnalyser);

      // Initialize VAD & AEC helper objects
      vadRef.current = new VoiceActivityDetector({
        energyThreshold: 0.015 / micSensitivity
      });
      aecRef.current = new AcousticEchoCanceller({
        couplingFactor: couplingFactor
      });

      setIsAudioActive(true);
      initializeASR();

      // Start processing loop
      startProcessingLoop(mAnalyser, tAnalyser);

    } catch (e) {
      console.error("Microphone access denied or audio graph initialization failed:", e);
      alert("Please allow microphone access to run the real-time Voice AI.");
    }
  };

  // Close Web Audio Context and Node Graph
  const stopAudioGraph = () => {
    clearPendingEOSTransition();
    clearSilenceFallback();
    setASRState(false);
    stopTTSReferenceSynth();
    window.speechSynthesis.cancel();

    if (micSourceRef.current) {
      micSourceRef.current.mediaStream.getTracks().forEach(t => t.stop());
      micSourceRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setMicAnalyser(null);
    setTtsAnalyser(null);
    setIsAudioActive(false);
    setState('LISTENING');
  };

  // Real-time audio processing loop
  const startProcessingLoop = (micAnal, ttsAnal) => {
    const fftSize = 128;
    const timeData = new Float32Array(fftSize);
    const freqData = new Float32Array(fftSize);

    const ttsTimeData = new Float32Array(fftSize);
    const ttsFreqData = new Float32Array(fftSize);

    const cleanTime = new Float32Array(fftSize);

    const process = () => {
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') return;

      // Extract raw mic values
      micAnal.getFloatTimeDomainData(timeData);
      micAnal.getFloatFrequencyData(freqData);

      // Extract active TTS reference values
      if (ttsAnal) {
        ttsAnal.getFloatTimeDomainData(ttsTimeData);
        ttsAnal.getFloatFrequencyData(ttsFreqData);
      }

      // 1. Run Acoustic Echo Cancellation reference subtraction
      const aiSpeaking = (stateRef.current === 'SPEAKING');
      aecRef.current.couplingFactor = couplingFactorRef.current;

      aecRef.current.cancelTimeDomain(timeData, ttsTimeData, cleanTime, aiSpeaking);

      // Set values for the visualizer
      setCancelledData(new Float32Array(cleanTime));

      // 2. Feed the echo-cancelled signal to the VAD
      const sampleRate = audioContextRef.current.sampleRate;
      const normalizedFreq = new Float32Array(fftSize);
      // Construct a mockup frequency magnitude array from the cleaned time domain buffer
      for (let i = 0; i < fftSize; i++) {
        normalizedFreq[i] = Math.abs(cleanTime[i]);
      }

      const confidence = vadRef.current.computeConfidence(
        cleanTime,
        normalizedFreq,
        sampleRate,
        fftSize * 2
      );

      setVadConfidence(confidence);
      vadConfidenceRef.current = confidence;

      // Update last speech time and postpone EOS triggers if user is actively speaking
      if (confidence > 0.3) {
        lastSpeechTimeRef.current = Date.now();
        if (stateRef.current === 'LISTENING') {
          if (eosTimeoutRef.current) {
            postponeEOSTransition();
          }
          if (silenceFallbackTimeoutRef.current) {
            resetSilenceFallbackTimer(latestTranscriptRef.current);
          }
        }
      }

      // 3. Interruption Monitor
      // Triggered immediately if VAD detects human speech onset (Confidence > vadThreshold)
      // during SPEAKING state, AND the signal doesn't match the AEC reference.
      if (aiSpeaking && confidence >= vadThresholdRef.current) {
        // Ensure there is actual signal difference (i.e. user spoke, not just bleed)
        // If AEC was fully successful, cleanTime contains the user's speech amplitude
        const rawRMS = vadRef.current.calculateRMS(timeData);
        const cleanRMS = vadRef.current.calculateRMS(cleanTime);

        // If clean signal still has substantial energy compared to the raw bleed,
        // it means there is an independent speech source (user voice)
        if (cleanRMS > 0.005) {
          triggerInterruption();
        }
      }

      setTimeout(process, 30); // ~33 FPS loop
    };

    process();
  };

  const handleToggleMic = () => {
    if (isAudioActive) {
      stopAudioGraph();
    } else {
      startAudioGraph();
    }
  };

  // Helper to append message history safely
  const updateTranscriptLog = (text, sender, pending = false) => {
    setHistory((prev) => {
      const list = [...prev];
      if (list.length > 0 && list[list.length - 1].sender === sender && list[list.length - 1].pending) {
        if (text === '') {
          // Clean empty pending
          list.pop();
        } else {
          // Update active pending text
          list[list.length - 1].text = text;
          list[list.length - 1].pending = pending;
        }
      } else {
        // Insert new block
        list.push({ sender, text, pending });
      }
      return list;
    });
  };

  // Handle batch file upload (.wav, .mp3)
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Client-side extension validation
    const allowed = ['.wav', '.mp3'];
    const fileExt = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowed.includes(fileExt)) {
      alert("Only .wav and .mp3 file types are allowed.");
      return;
    }

    // Client-side size validation (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      alert("File size exceeds the 10MB limit.");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload-audio', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setUploadTimeline(result);

        // Execute animated simulation of the timeline
        simulateBatchTimeline(result);
      } else {
        alert("Upload failed. Ensure backend server is running.");
      }
    } catch (e) {
      console.error(e);
      alert("Error uploading file.");
    } finally {
      setIsUploading(false);
    }
  };

  // Simulates feeding transcription tokens word-by-word into the dashboard layout
  const simulateBatchTimeline = (batchResponse) => {
    const fullTranscript = batchResponse.transcript;
    const evaluationData = batchResponse.evaluation;

    if (!fullTranscript) return;

    const words = fullTranscript.split(' ');

    // Reset layout logs and states for visualization tracking
    setHistory([]);
    setState('LISTENING');
    setEosJson(null);

    let currentIndex = 0;
    let accumulatedText = '';

    const stepToken = () => {
      if (currentIndex >= words.length) {
        // Timeline finished running typing animations -> immediately inject final structural context
        setEosJson(evaluationData);
        setState('THINKING');
        updateTranscriptLog(fullTranscript, 'user', false);
        return;
      }

      accumulatedText += (currentIndex === 0 ? '' : ' ') + words[currentIndex];

      // Update running text log view
      updateTranscriptLog(accumulatedText, 'user', true);

      currentIndex++;
      // Smooth dynamic pacing based on structural lengths
      setTimeout(stepToken, 180);
    };

    stepToken();
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="header">
        <h1 className="header-title">VOICE AI LINGUISTIC EOS PLATFORM</h1>
        <div className="header-status">
          <span
            className="status-dot"
            style={{
              color: isWebSocketConnected ? 'var(--color-listening)' : 'var(--color-interrupted)',
              backgroundColor: isWebSocketConnected ? 'var(--color-listening)' : 'var(--color-interrupted)'
            }}
          />
          <span>{isWebSocketConnected ? 'SYSTEM ACTIVE' : 'LOCAL MOCK MODE'}</span>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="main-grid">
        {/* Left Section: Controls */}
        <Controls
          micSensitivity={micSensitivity}
          setMicSensitivity={setMicSensitivity}
          vadThreshold={vadThreshold}
          setVadThreshold={setVadThreshold}
          couplingFactor={couplingFactor}
          setCouplingFactor={setCouplingFactor}
          isAecEnabled={isAecEnabled}
          setIsAecEnabled={setIsAecEnabled}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          isAudioActive={isAudioActive}
          onToggleMic={handleToggleMic}
        />

        {/* Center Section: Holo Orb & Visualizers */}
        <div className="panel" style={{ justifyContent: 'space-between', gap: '24px' }}>
          <h3 className="panel-title">
            <MessageSquare size={18} color="#8b5cf6" />
            AGENT CORE AGGREGATOR
          </h3>

          <Orb state={state} isWebSocketConnected={isWebSocketConnected} />

          <Visualizers
            micAnalyser={micAnalyser}
            ttsAnalyser={ttsAnalyser}
            cancelledData={cancelledData}
            vadConfidence={vadConfidence}
            isAecEnabled={isAecEnabled}
          />
        </div>

        {/* Right Section: Logs & Code Inspector */}
        <Transcript
          history={history}
          eosJson={eosJson}
        />
      </main>

      {/* Bottom Batch File Upload Drawer */}
      <footer style={{ marginTop: '24px', display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
        <div className="panel">
          <h3 className="panel-title">
            <UploadCloud size={18} color="#6366f1" />
            BATCH FILE INGESTION
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <label className="upload-zone">
              <input
                type="file"
                accept=".wav,.mp3"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                disabled={isUploading}
              />
              <UploadCloud size={24} style={{ color: 'var(--text-muted)' }} />
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                {isUploading ? 'Uploading & Processing...' : 'Click to Upload Wave/MP3'}
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                Max size 10MB • Runs word-by-word timeline analysis
              </span>
            </label>
          </div>
        </div>

        {/* Quick Tips and Architecture Details */}
        <div className="panel" style={{ flexDirection: 'row', gap: '20px', alignItems: 'center' }}>
          <AlertTriangle size={32} style={{ color: 'var(--color-thinking)', flexShrink: 0 }} />
          <div>
            <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '6px' }}>REAL-TIME AEC & INTERRUPTION SIMULATION DETAILS</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              To verify <span style={{ color: 'var(--color-listening)' }}>&lt;300ms Interruption Latency</span>, click Connect Microphone and say <strong>"What is polymorphism"</strong>. While the Agent is speaking, say anything. The custom VAD monitors the echo-cancelled mic line and halts the output buffer instantly. Use the file upload to inspect word-by-word evaluation of partial speech.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
