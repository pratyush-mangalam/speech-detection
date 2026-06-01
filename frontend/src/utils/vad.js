/**
 * Voice Activity Detector (VAD)
 * Analyzes audio buffers to compute speech confidence based on:
 * 1. RMS Energy (volume level)
 * 2. Zero Crossing Rate (ZCR) - distinguishes voiced speech from noise/sibilance
 * 3. Spectral Centroid - estimates the spectral center of mass (speech formant range)
 */
export class VoiceActivityDetector {
  constructor(options = {}) {
    // Configurable thresholds
    this.energyThreshold = options.energyThreshold !== undefined ? options.energyThreshold : 0.015; // RMS
    this.zcrMin = options.zcrMin !== undefined ? options.zcrMin : 0.02;
    this.zcrMax = options.zcrMax !== undefined ? options.zcrMax : 0.25;
    this.centroidMin = options.centroidMin !== undefined ? options.centroidMin : 300; // Hz
    this.centroidMax = options.centroidMax !== undefined ? options.centroidMax : 3500; // Hz
    
    // Smoothing parameters
    this.smoothingFactor = options.smoothingFactor || 0.3; // Exponential moving average
    this.lastConfidence = 0.0;
    
    // State
    this.speechHistory = [];
    this.historyLength = 10; // Number of frames to check for onset
  }

  /**
   * Calculates Root Mean Square (RMS) energy of the time domain buffer
   */
  calculateRMS(timeData) {
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) {
      sum += timeData[i] * timeData[i];
    }
    return Math.sqrt(sum / timeData.length);
  }

  /**
   * Calculates Zero Crossing Rate (ZCR) of the time domain buffer
   */
  calculateZCR(timeData) {
    let crossings = 0;
    for (let i = 1; i < timeData.length; i++) {
      // Check if sign crossed zero
      if ((timeData[i] >= 0 && timeData[i - 1] < 0) || (timeData[i] < 0 && timeData[i - 1] >= 0)) {
        crossings++;
      }
    }
    return crossings / timeData.length;
  }

  /**
   * Calculates Spectral Centroid from frequency magnitude buffer
   */
  calculateSpectralCentroid(freqData, sampleRate, fftSize) {
    let numerator = 0;
    let denominator = 0;
    const binWidth = sampleRate / fftSize;

    for (let i = 0; i < freqData.length; i++) {
      const frequency = i * binWidth;
      // Convert dB representation or normalized magnitude back
      const magnitude = freqData[i];
      numerator += frequency * magnitude;
      denominator += magnitude;
    }

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Evaluates speech confidence (0.0 to 1.0) for a given audio frame
   * @param {Float32Array} timeData - raw audio time domain data (-1.0 to 1.0)
   * @param {Float32Array} freqData - audio frequency magnitude data (normalized 0.0 to 1.0 or raw amplitude)
   * @param {number} sampleRate - audio context sample rate (e.g., 44100)
   * @param {number} fftSize - FFT size used for analyzer
   */
  computeConfidence(timeData, freqData, sampleRate, fftSize) {
    if (!timeData || timeData.length === 0) return 0.0;

    const rms = this.calculateRMS(timeData);
    const zcr = this.calculateZCR(timeData);
    const centroid = this.calculateSpectralCentroid(freqData, sampleRate, fftSize);

    // 1. Energy confidence
    // High sensitivity to energy. If silent, probability is near 0.
    const energyConf = Math.min(1.0, rms / this.energyThreshold);

    // 2. ZCR confidence
    // Voiced speech typically lies in a specific band. High values are sibilants/noise, low values are silence.
    let zcrConf = 0.0;
    if (zcr >= this.zcrMin && zcr <= this.zcrMax) {
      zcrConf = 1.0;
    } else if (zcr > this.zcrMax && zcr < this.zcrMax * 2) {
      // Degrade gracefully
      zcrConf = 1.0 - ((zcr - this.zcrMax) / this.zcrMax);
    }

    // 3. Spectral Centroid confidence
    // Standard human vocal speech ranges from 300Hz to 3400Hz (telephone band)
    let centroidConf = 0.0;
    if (centroid >= this.centroidMin && centroid <= this.centroidMax) {
      centroidConf = 1.0;
    } else if (centroid < this.centroidMin && centroid > this.centroidMin / 2) {
      centroidConf = (centroid - (this.centroidMin / 2)) / (this.centroidMin / 2);
    } else if (centroid > this.centroidMax && centroid < this.centroidMax * 1.5) {
      centroidConf = 1.0 - ((centroid - this.centroidMax) / (this.centroidMax * 0.5));
    }

    // Weight combination
    // Energy is critical. If energy is 0, confidence is 0.
    let confidence = 0.0;
    if (rms > this.energyThreshold * 0.2) {
      // Combined score: 50% energy, 30% spectral centroid, 20% ZCR
      confidence = (energyConf * 0.5) + (centroidConf * 0.3) + (zcrConf * 0.2);
    }
    
    // Enforce strict limits
    confidence = Math.max(0.0, Math.min(1.0, confidence));

    // Smooth output using exponential moving average
    const smoothedConfidence = (confidence * (1.0 - this.smoothingFactor)) + (this.lastConfidence * this.smoothingFactor);
    this.lastConfidence = smoothedConfidence;

    // Save history
    this.speechHistory.push(smoothedConfidence);
    if (this.speechHistory.length > this.historyLength) {
      this.speechHistory.shift();
    }

    return smoothedConfidence;
  }

  /**
   * Helper to check if user speech onset has been detected.
   * Onset: Average confidence of the last few frames is above a threshold.
   */
  isSpeechOnset(threshold = 0.85) {
    if (this.speechHistory.length < 3) return false;
    // Check average of last 3 frames to avoid transient spikes
    const lastFrames = this.speechHistory.slice(-3);
    const avg = lastFrames.reduce((a, b) => a + b, 0) / lastFrames.length;
    return avg >= threshold;
  }
}
