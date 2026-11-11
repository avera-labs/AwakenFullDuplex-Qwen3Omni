// Mic capture on the audio thread: posts int16 chunks of `chunk` samples (20 ms @ 24 kHz) with the
// AudioContext time at which the chunk's LAST sample was captured (same clock as the playback worklet).
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunk = (options.processorOptions && options.processorOptions.chunk) || 480;
    this.buf = new Float32Array(this.chunk);
    this.n = 0;
    this.on = true;
    this.port.onmessage = (e) => { if (e.data && e.data.type === "enable") this.on = !!e.data.on; };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || !this.on) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.chunk) {
        const out = new Int16Array(this.chunk);
        for (let k = 0; k < this.chunk; k++) {
          const s = Math.max(-1, Math.min(1, this.buf[k]));
          out[k] = s < 0 ? s * 32768 : s * 32767;
        }
        // currentTime = start of this render quantum; sample i is i/sampleRate later
        this.port.postMessage({ pcm: out.buffer, t: currentTime + (i + 1) / sampleRate }, [out.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
