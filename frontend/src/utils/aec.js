/**
 * Acoustic Echo Cancellation (AEC) Simulator / Processor
 * Performs real-time spectral subtraction to model echo cancellation.
 * It references the AI's active Text-to-Speech (TTS) output channel
 * and filters it out of the incoming Microphone signal.
 */
export class AcousticEchoCanceller {
  constructor(options = {}) {
    // Scaling factor for echo subtraction (adaptive coupling gain)
    this.couplingFactor = options.couplingFactor !== undefined ? options.couplingFactor : 0.8;
    this.noiseFloor = options.noiseFloor !== undefined ? options.noiseFloor : 0.01;
  }

  /**
   * Performs Spectral Subtraction between the raw Mic signal and the TTS Reference signal.
   * Both arrays are frequency magnitude spectra (e.g., from AnalyserNode.getByteFrequencyData or getFloatFrequencyData)
   * 
   * @param {Float32Array} micFreq - Raw microphone frequency magnitudes (0.0 to 1.0)
   * @param {Float32Array} ttsFreq - Active TTS reference frequency magnitudes (0.0 to 1.0)
   * @param {Float32Array} outputFreq - Destination array for echo-cancelled magnitudes
   */
  cancelSpectral(micFreq, ttsFreq, outputFreq) {
    if (!micFreq || !outputFreq) return;

    for (let i = 0; i < micFreq.length; i++) {
      const micVal = micFreq[i];
      // If TTS is active, subtract its spectral footprint scaled by the coupling factor
      const ttsVal = ttsFreq ? ttsFreq[i] : 0.0;
      
      // Spectral subtraction: E[k] = max(noiseFloor, M[k] - beta * T[k])
      const cleanVal = micVal - (this.couplingFactor * ttsVal);
      outputFreq[i] = Math.max(this.noiseFloor, cleanVal);
    }
  }

  /**
   * Cleans the raw time-domain microphone buffer by applying a dynamic gain reduction
   * in frequencies where the reference TTS is dominant, or applying time-domain cancellation.
   * 
   * @param {Float32Array} micTime - Raw time-domain microphone data (-1.0 to 1.0)
   * @param {Float32Array} ttsTime - Active TTS reference time-domain data (-1.0 to 1.0)
   * @param {Float32Array} outputTime - Destination array for cleaned time-domain data
   * @param {boolean} isSpeaking - Whether the AI is currently playing TTS
   */
  cancelTimeDomain(micTime, ttsTime, outputTime, isSpeaking) {
    if (!micTime || !outputTime) return;

    if (!isSpeaking || !ttsTime || ttsTime.length === 0) {
      // If AI is not speaking, pass through the mic signal directly
      outputTime.set(micTime);
      return;
    }

    // Time-domain adaptive reference subtraction:
    // We align the signals and subtract. Since physical delay exists,
    // we also apply a soft gate/attenuation based on the coupling factor.
    for (let i = 0; i < micTime.length; i++) {
      const micVal = micTime[i];
      const ttsVal = ttsTime[i] || 0.0;
      
      // Subtract matching phase components
      let cleanVal = micVal - (this.couplingFactor * ttsVal * 0.5);
      
      // Apply slight threshold attenuation if TTS is highly active to simulate room bleed suppression
      if (Math.abs(ttsVal) > 0.05) {
        cleanVal *= (1.0 - (this.couplingFactor * 0.4));
      }
      
      outputTime[i] = Math.max(-1.0, Math.min(1.0, cleanVal));
    }
  }
}
