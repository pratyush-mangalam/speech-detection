import React from 'react';
import { Sliders, Shield, Volume2, Mic, Settings } from 'lucide-react';

export const Controls = ({
  micSensitivity,
  setMicSensitivity,
  vadThreshold,
  setVadThreshold,
  couplingFactor,
  setCouplingFactor,
  isAecEnabled,
  setIsAecEnabled,
  selectedModel,
  setSelectedModel,
  isAudioActive,
  onToggleMic
}) => {
  return (
    <div className="panel">
      <h3 className="panel-title">
        <Settings size={18} color="#a5b4fc" />
        SYSTEM CONTROL DASHBOARD
      </h3>

      <div style={{ marginBottom: '24px' }}>
        <button 
          className={`action-btn ${isAudioActive ? 'listening' : ''}`}
          onClick={onToggleMic}
          style={{ width: '100%' }}
        >
          {isAudioActive ? (
            <>
              <Mic size={18} />
              DISCONNECT MICROPHONE
            </>
          ) : (
            <>
              <Mic size={18} />
              CONNECT MICROPHONE
            </>
          )}
        </button>
      </div>

      <div className="control-group">
        {/* 1. Mic Sensitivity */}
        <div className="control-item">
          <div className="control-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Volume2 size={14} />
              Mic Input Gain
            </span>
            <span>{micSensitivity.toFixed(1)}x</span>
          </div>
          <input 
            type="range" 
            min="0.1" 
            max="3.0" 
            step="0.1"
            value={micSensitivity} 
            onChange={(e) => setMicSensitivity(parseFloat(e.target.value))}
            className="slider-input" 
          />
        </div>

        {/* 2. VAD Interruption Confidence Threshold */}
        <div className="control-item">
          <div className="control-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Shield size={14} />
              VAD Onset Threshold
            </span>
            <span>{(vadThreshold * 100).toFixed(0)}%</span>
          </div>
          <input 
            type="range" 
            min="0.50" 
            max="0.95" 
            step="0.05"
            value={vadThreshold} 
            onChange={(e) => setVadThreshold(parseFloat(e.target.value))}
            className="slider-input" 
          />
        </div>

        {/* 3. Echo Cancellation Coupling Gain */}
        <div className="control-item">
          <div className="control-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sliders size={14} />
              AEC Coupling Factor
            </span>
            <span>{couplingFactor.toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            min="0.00" 
            max="1.50" 
            step="0.05"
            value={couplingFactor} 
            onChange={(e) => setCouplingFactor(parseFloat(e.target.value))}
            className="slider-input" 
          />
        </div>
      </div>

      {/* Toggles */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '24px' }}>
        <div className="toggle-container">
          <div>
            <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>WebRTC AEC</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Use browser echo cancellation</div>
          </div>
          <label className="toggle-switch">
            <input 
              type="checkbox" 
              checked={isAecEnabled}
              onChange={(e) => setIsAecEnabled(e.target.checked)}
            />
            <span className="slider-toggle"></span>
          </label>
        </div>
      </div>

      {/* Model Selection */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
          SEMANTIC EOS EVALUATOR MODEL
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{
            background: 'rgba(0, 0, 0, 0.3)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-glow)',
            padding: '10px 14px',
            borderRadius: '8px',
            outline: 'none',
            fontSize: '0.85rem',
            cursor: 'pointer'
          }}
        >
          <option value="google/gemini-2.0-flash">Gemini 2.0 Flash (Recommended)</option>
          <option value="meta-llama/llama-3-8b-instruct">Llama 3 8B Instruct</option>
          <option value="qwen/qwen-2.5-coder-32b-instruct">Qwen 2.5 Coder (32B)</option>
        </select>
      </div>
    </div>
  );
};

export default Controls;
