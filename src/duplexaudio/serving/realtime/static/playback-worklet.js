// Agent-audio playback on the audio thread: ring buffer + adaptive jitter target.
//   underrun                          -> play silence, rebuffer, target += 20 ms (cap 300 ms)
//   2 s window minimum > target+20 ms -> drop the excess, ONLY inside a frame the talker marked silent
//   10 s without underrun             -> target -= 10 ms (floor 30 ms)
//   ring overflow (client stalled)    -> drop the oldest audio, count it (never overwrite unread samples silently)
// Reports buffer state and, for latency measurement, the context time each agent frame STARTS playing.
const RING = 24000 * 20;
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(RING);
    this.w = 0; this.r = 0;                       // absolute sample counters
    this.frames = [];                             // {start, end, j, silent}
    this.target = 0.06 * sampleRate; this.playing = false;
    this.underruns = 0; this.drops = 0; this.dropped = 0; this.overflows = 0;
    this.winMin = Infinity; this.winStart = currentTime; this.lastUnderrun = currentTime;
    this.lastReport = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "frame") this.push(m.pcm, m.j, m.silent);
      else if (m.type === "reset") { this.w = this.r = 0; this.frames = []; this.playing = false; }
    };
  }
  push(pcm, j, silent) {
    const x = new Float32Array(pcm);
    if (this.w + x.length - this.r > RING) {      // overflow guard: drop the oldest unread audio
      const k = this.w + x.length - this.r - RING;
      this.r += k; this.overflows++; this.dropped += k;
      while (this.frames.length && this.frames[0].end <= this.r) this.frames.shift();
    }
    for (let i = 0; i < x.length; i++) this.ring[(this.w + i) % RING] = x[i];
    this.frames.push({ start: this.w, end: this.w + x.length, j, silent, reported: false });
    this.w += x.length;
  }
  process(_, outputs) {
    const out = outputs[0][0];
    const n = out.length;
    let avail = this.w - this.r;
    if (!this.playing && avail >= this.target) this.playing = true;
    if (this.playing && avail < n) {               // underrun
      this.underruns++; this.playing = false; this.lastUnderrun = currentTime;
      this.target = Math.min(this.target + 0.02 * sampleRate, 0.3 * sampleRate);
    }
    if (this.playing) {
      for (let i = 0; i < n; i++) out[i] = this.ring[(this.r + i) % RING];
      for (const f of this.frames) {
        if (!f.reported && f.start < this.r + n && f.end > this.r) {
          f.reported = true;
          this.port.postMessage({ type: "playing", j: f.j, t: currentTime + Math.max(0, f.start - this.r) / sampleRate });
        }
      }
      this.r += n;
    } else {
      out.fill(0);
    }
    avail = this.w - this.r;
    while (this.frames.length && this.frames[0].end <= this.r) this.frames.shift();
    this.winMin = Math.min(this.winMin, avail);
    if (currentTime - this.winStart >= 2) {      // drift control on a 2 s window
      const excess = this.winMin - this.target - 0.02 * sampleRate;
      const cur = this.frames[0];
      if (this.playing && excess > 0 && cur && cur.silent) {
        const k = Math.min(excess, cur.end - this.r);
        this.r += k; this.drops++; this.dropped += k;
      }
      this.winMin = Infinity; this.winStart = currentTime;
    }
    if (currentTime - this.lastUnderrun > 10 && this.target > 0.03 * sampleRate) {
      this.target = Math.max(this.target - 0.01 * sampleRate, 0.03 * sampleRate);
      this.lastUnderrun = currentTime;
    }
    if (currentTime - this.lastReport > 0.2) {
      this.lastReport = currentTime;
      this.port.postMessage({ type: "stats", buffered_ms: 1000 * avail / sampleRate, target_ms: 1000 * this.target / sampleRate,
                              underruns: this.underruns, drops: this.drops, overflows: this.overflows,
                              dropped_ms: 1000 * this.dropped / sampleRate, playing: this.playing });
    }
    return true;
  }
}
registerProcessor("playback-processor", PlaybackProcessor);
