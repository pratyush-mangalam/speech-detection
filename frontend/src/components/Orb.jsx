import React from 'react';

export const Orb = ({ state, isWebSocketConnected, onOrbClick }) => {
  // Map states to colors and shadows
  const stateConfig = {
    LISTENING: {
      label: 'Listening',
      color: 'var(--color-listening)',
      shadow: 'var(--shadow-listening)',
      className: 'listening-mode'
    },
    THINKING: {
      label: 'Thinking',
      color: 'var(--color-thinking)',
      shadow: 'var(--shadow-thinking)',
      className: 'thinking-mode'
    },
    SPEAKING: {
      label: 'Speaking',
      color: 'var(--color-speaking)',
      shadow: 'var(--shadow-speaking)',
      className: 'speaking-mode'
    },
    INTERRUPTED: {
      label: 'Interrupted',
      color: 'var(--color-interrupted)',
      shadow: 'var(--shadow-interrupted)',
      className: 'interrupted-mode'
    }
  };

  const current = stateConfig[state] || stateConfig.LISTENING;

  return (
    <div 
      className={`center-container ${state === 'SPEAKING' ? 'clickable-speaking' : ''}`}
      onClick={state === 'SPEAKING' ? onOrbClick : undefined}
      style={{
        cursor: state === 'SPEAKING' ? 'pointer' : 'default'
      }}
      title={state === 'SPEAKING' ? 'Click to interrupt AI' : undefined}
    >
      <div className="orb-wrapper">
        {/* Layer 1 - Outer Glow */}
        <div 
          className="orb-element" 
          style={{
            background: `radial-gradient(circle, ${current.color} 0%, transparent 70%)`,
            boxShadow: current.shadow,
            transform: 'scale(1.2)',
            opacity: 0.4,
            animation: 'pulse-slow 4s infinite alternate'
          }}
        />
        
        {/* Layer 2 - Morphing Base */}
        <div 
          className="orb-element" 
          style={{
            background: `linear-gradient(135deg, ${current.color} 0%, rgba(0,0,0,0) 100%)`,
            borderRadius: '40% 60% 70% 30% / 40% 50% 60% 50%',
            animation: 'morph 6s infinite ease-in-out alternate, spin 12s infinite linear',
            opacity: 0.7
          }}
        />

        {/* Layer 3 - Core Energy */}
        <div 
          className="orb-element" 
          style={{
            background: '#ffffff',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            filter: 'blur(12px)',
            opacity: 0.9,
            transform: 'scale(0.8)',
            animation: 'pulse-fast 1.5s infinite alternate'
          }}
        />

        {/* Connection status overlay ring */}
        <div 
          style={{
            position: 'absolute',
            width: '180px',
            height: '180px',
            borderRadius: '50%',
            border: `2px dashed ${isWebSocketConnected ? 'rgba(255,255,255,0.15)' : 'rgba(239, 68, 68, 0.4)'}`,
            animation: 'spin 30s infinite linear'
          }}
        />
      </div>

      <div style={{ textAlign: 'center', zIndex: 10 }}>
        <div 
          style={{ 
            fontFamily: 'Outfit', 
            fontWeight: 800, 
            fontSize: '1.5rem', 
            textTransform: 'uppercase', 
            letterSpacing: '4px',
            color: current.color,
            textShadow: `0 0 10px ${current.color}40`,
            transition: 'color 0.3s ease'
          }}
        >
          {current.label}
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px' }}>
          {isWebSocketConnected ? 'AEC Feed Enabled • Latency < 300ms' : 'WS Server Offline • Simulated Local'}
        </div>
      </div>

      {/* Embedded CSS Animations */}
      <style>{`
        @keyframes morph {
          0% {
            border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
          }
          50% {
            border-radius: 60% 40% 50% 70% / 50% 60% 40% 60%;
          }
          100% {
            border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
          }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse-slow {
          0% { transform: scale(1.0); opacity: 0.3; }
          100% { transform: scale(1.3); opacity: 0.5; }
        }
        @keyframes pulse-fast {
          0% { transform: scale(0.7); opacity: 0.8; }
          100% { transform: scale(0.95); opacity: 1.0; }
        }
        .clickable-speaking {
          transition: transform 0.2s ease, filter 0.2s ease;
        }
        .clickable-speaking:hover {
          transform: scale(1.05);
          filter: brightness(1.2);
        }
        .clickable-speaking:active {
          transform: scale(0.98);
        }
      `}</style>
    </div>
  );
};
export default Orb;
