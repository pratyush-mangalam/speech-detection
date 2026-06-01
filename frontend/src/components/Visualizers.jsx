import React, { useEffect, useRef } from 'react';

export const Visualizers = ({ 
  micAnalyser, 
  ttsAnalyser, 
  cancelledData, // Float32Array containing cleaned time-domain data
  vadConfidence = 0.0,
  isAecEnabled = true
}) => {
  const micCanvasRef = useRef(null);
  const ttsCanvasRef = useRef(null);
  const aecCanvasRef = useRef(null);
  
  const animationRef = useRef(null);

  useEffect(() => {
    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      
      // 1. Draw Raw Mic Canvas
      drawWave(micCanvasRef.current, micAnalyser, '#00f2fe', 'Raw Microphone');
      
      // 2. Draw TTS Reference Canvas
      drawWave(ttsCanvasRef.current, ttsAnalyser, '#8b5cf6', 'AI TTS Reference');

      // 3. Draw Echo-Cancelled Canvas
      drawCancelledWave(
        aecCanvasRef.current, 
        cancelledData, 
        isAecEnabled ? '#10b981' : '#ef4444', 
        isAecEnabled ? 'Echo-Cancelled Signal (AEC)' : 'Unprocessed Signal (AEC OFF)'
      );
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [micAnalyser, ttsAnalyser, cancelledData, isAecEnabled]);

  // Helper to draw standard analyser node time-domain data
  const drawWave = (canvas, analyser, color, label) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    if (!analyser) {
      // Draw flat baseline
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);
    
    // Read float time domain data
    analyser.getFloatTimeDomainData(dataArray);

    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      // dataArray values are -1.0 to 1.0. Scale and offset to fit height.
      const v = dataArray[i] * 1.5; // Amplify for visualization
      const y = (v + 1) * (height / 2);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();
  };

  // Helper to draw the custom calculated cancelled array
  const drawCancelledWave = (canvas, timeData, color, label) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    if (!timeData || timeData.length === 0) {
      // Draw flat baseline
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const bufferLength = timeData.length;
    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = timeData[i] * 1.5; // Amplify for visualization
      const y = (v + 1) * (height / 2);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();
  };

  return (
    <div className="visualizers-container">
      {/* 1. Raw Mic */}
      <div className="visualizer-card">
        <div className="visualizer-label">
          <span>🎤 RAW MICROPHONE INPUT</span>
          <span style={{ color: micAnalyser ? '#00f2fe' : 'var(--text-muted)' }}>
            {micAnalyser ? 'ACTIVE' : 'IDLE'}
          </span>
        </div>
        <canvas 
          ref={micCanvasRef} 
          width={400} 
          height={60} 
          className="visualizer-canvas" 
        />
      </div>

      {/* 2. TTS Reference */}
      <div className="visualizer-card">
        <div className="visualizer-label">
          <span>🔊 AI TTS REFERENCE FEED</span>
          <span style={{ color: ttsAnalyser ? '#8b5cf6' : 'var(--text-muted)' }}>
            {ttsAnalyser ? 'PLAYING' : 'SILENT'}
          </span>
        </div>
        <canvas 
          ref={ttsCanvasRef} 
          width={400} 
          height={60} 
          className="visualizer-canvas" 
        />
      </div>

      {/* 3. Echo-Cancelled & VAD */}
      <div className="visualizer-card" style={{ position: 'relative' }}>
        <div className="visualizer-label">
          <span>🛡️ ECHO-CANCELLED SIGNAL (VAD INGESTION)</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {/* VAD confidence meter */}
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              VAD: {(vadConfidence * 100).toFixed(0)}%
            </span>
            <span 
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: vadConfidence > 0.85 ? 'var(--color-listening)' : 'rgba(255,255,255,0.1)',
                boxShadow: vadConfidence > 0.85 ? '0 0 10px var(--color-listening)' : 'none',
                transition: 'background-color 0.1s'
              }}
            />
          </span>
        </div>
        <canvas 
          ref={aecCanvasRef} 
          width={400} 
          height={60} 
          className="visualizer-canvas" 
        />
      </div>
    </div>
  );
};

export default Visualizers;
