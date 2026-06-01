import React, { useEffect, useRef, useState } from 'react';
import { Terminal, Code2, Maximize2, Minimize2 } from 'lucide-react';

export const Transcript = ({ 
  history, // Array of { sender: 'user'|'ai', text: string, pending: boolean }
  eosJson // Last JSON response from the EOS Evaluator
}) => {
  const terminalEndRef = useRef(null);
  const [isExpanded, setIsExpanded] = useState(false);

  // Auto-scroll transcript terminal to bottom
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history]);

  // Stylized JSON parser for the inspector
  const renderFormattedJson = (jsonObj) => {
    if (!jsonObj) {
      return <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Waiting for speech activity to trigger semantic evaluation...</div>;
    }

    const renderObj = (obj) => {
      if (!obj) return null;
      const keys = Object.keys(obj);
      return (
        <div>
          <span>{'{'}</span>
          <div style={{ paddingLeft: '16px' }}>
            {keys.map((key, index) => {
              const val = obj[key];
              let valElement;
              
              if (typeof val === 'string') {
                valElement = <span className="json-string">"{val}"</span>;
              } else if (typeof val === 'number') {
                valElement = <span className="json-number">{val}</span>;
              } else if (typeof val === 'boolean') {
                valElement = <span className="json-boolean">{val ? 'true' : 'false'}</span>;
              } else {
                valElement = <span>{JSON.stringify(val)}</span>;
              }

              return (
                <div key={key}>
                  <span className="json-key">"{key}"</span>: {valElement}
                  {index < keys.length - 1 ? ',' : ''}
                </div>
              );
            })}
          </div>
          <span>{'}'}</span>
        </div>
      );
    };

    try {
      if (Array.isArray(jsonObj)) {
        return (
          <div>
            <span>{'['}</span>
            <div style={{ paddingLeft: '16px' }}>
              {jsonObj.map((item, index) => (
                <div key={index} style={{ marginBottom: '8px' }}>
                  {renderObj(item)}
                  {index < jsonObj.length - 1 ? ',' : ''}
                </div>
              ))}
            </div>
            <span>{']'}</span>
          </div>
        );
      }
      return renderObj(jsonObj);
    } catch (e) {
      return <pre>{JSON.stringify(jsonObj, null, 2)}</pre>;
    }
  };

  return (
    <div className="panel" style={{ gap: '16px' }}>
      {/* Transcript Board */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <h3 className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Terminal size={18} color="#00f2fe" />
            SPEECH TRANSCRIPTION LOG
          </span>
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: 'var(--text-primary)',
              borderRadius: '6px',
              padding: '4px 10px',
              fontSize: '0.75rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'background 0.2s',
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 600
            }}
          >
            {isExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
            {isExpanded ? 'COLLAPSE' : 'EXPAND'}
          </button>
        </h3>
        
        <div 
          className="transcript-area" 
          style={{ 
            maxHeight: isExpanded ? '480px' : '250px', 
            transition: 'max-height 0.3s ease' 
          }}
        >
          {history.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', margin: 'auto', fontSize: '0.85rem' }}>
              Microphone output is clean. Start speaking to stream.
            </div>
          ) : (
            history.map((msg, index) => (
              <div 
                key={index} 
                className={`transcript-bubble ${msg.sender === 'user' ? 'transcript-user' : 'transcript-ai'} ${msg.pending ? 'transcript-pending' : ''}`}
              >
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', fontWeight: 600, opacity: 0.6, marginBottom: '2px' }}>
                  {msg.sender === 'user' ? 'User (ASR)' : 'Agent (TTS)'}
                </div>
                <div>{msg.text}</div>
              </div>
            ))
          )}
          <div ref={terminalEndRef} />
        </div>
      </div>

      {/* JSON Inspector */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <h3 className="panel-title" style={{ marginBottom: '12px' }}>
          <Code2 size={18} color="#10b981" />
          SEMANTIC EOS EVALUATOR SCHEMAS
        </h3>
        <div 
          className="json-container" 
          style={{ 
            height: isExpanded ? '90px' : '220px', 
            transition: 'height 0.3s ease' 
          }}
        >
          {renderFormattedJson(eosJson)}
        </div>
      </div>
    </div>
  );
};

export default Transcript;
