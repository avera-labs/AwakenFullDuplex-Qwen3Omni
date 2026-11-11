// Realtime duplex demo client: mic (or a wav file) -> 20 ms int16 @ 24 kHz -> WS; agent frames -> playback worklet.
// One clock for every latency number: the AudioContext's (capture + playback worklets).
// All URLs are relative, so the page also works behind a reverse proxy that serves it under a sub-path.
"use strict";
const $ = (s) => document.querySelector(s);
const FRAME = 1920;                                   // samples per 80 ms frame @ 24 kHz
const HDR = 36;                                       // LCA2 header bytes (protocol.py DOWN_HDR.size)
let ws = null, ctx = null, capNode = null, playNode = null, sinkGain = null, srcNode = null, micStream = null;
let state = "loading", running = false, capFrame = new Map(), samplesSent = 0, pipeLat = [], rtts = [];
let modelLagMs = 0, curTurn = null, capFrames = 0;

function log(msg, cls) {
  const d = document.createElement("div"); if (cls) d.className = cls;
  d.textContent = new Date().toLocaleTimeString() + "  " + msg; $("#log").prepend(d);
}
function median(a) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; }
function setV(id, text, cls) { const el = $(id); el.textContent = text; el.className = "v" + (cls ? " " + cls : ""); }
function setPill(st, msg) {
  state = st;
  const map = { ready: ["pill-ready", "ready"], live: ["pill-live", "live"], loading: ["pill-busy", "loading"],
                arming: ["pill-busy", "preparing"], error: ["pill-err", "error"], closed: ["pill-err", "disconnected"] };
  const [cls, txt] = map[running ? "live" : st] || ["pill-wait", st];
  $("#pill").className = "pill " + cls; $("#pill").textContent = txt;
  $("#statusMsg").textContent = msg || "";
  $("#startBtn").disabled = !(st === "ready" && !running) || !!insecure;
  $("#stopBtn").disabled = !running;
}

// ── websocket ────────────────────────────────────────────────────────────────────────────────────────
function connect() {
  const u = new URL("ws", location.href);
  u.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(u.href);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => log("connected");
  ws.onclose = () => { running = false; teardownInput(); setPill("closed", "disconnected (another client is connected, or the server stopped)"); };
  ws.onmessage = (e) => (typeof e.data === "string" ? onJson(JSON.parse(e.data)) : onFrame(e.data));
}
setInterval(() => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "ping", t: performance.now() })); }, 1000);

function onJson(m) {
  switch (m.type) {
    case "status":
      $("#profile").textContent = `profile ${m.profile} · thinker ${m.thinker} · talker ${m.talker}`;
      setPill(m.state, m.msg); break;
    case "ready": onReady(m); break;
    case "text": onText(m); break;
    case "event": if (m.kind === "se") curTurn = null; break;
    case "metrics": onMetrics(m); break;
    case "pong": rtts.push(performance.now() - m.t); if (rtts.length > 30) rtts.shift();
                 setV("#m_rtt", `${median(rtts).toFixed(1)} ms`); break;
    case "warn": log(m.msg, "w"); if (m.kind) showInputWarn(m.msg); break;
    case "error": log(m.msg, "e"); break;
    case "ended": onEnded(m); break;
  }
}

// ── agent audio frames ───────────────────────────────────────────────────────────────────────────────
function onFrame(buf) {
  const dv = new DataView(buf);
  const j = dv.getInt32(8, true), flags = dv.getUint32(12, true), n = dv.getUint16(22, true);
  if (!n || !playNode || j < 0) return;               // frame -1 (voice-prompt spacer) is never played
  const i16 = new Int16Array(buf, HDR, n), f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = i16[i] / 32768;
  playNode.port.postMessage({ type: "frame", j, pcm: f32.buffer, silent: !!(flags & 2) }, [f32.buffer]);
}

// ── conversation lifecycle ───────────────────────────────────────────────────────────────────────────
async function ensureAudio() {
  if (ctx) { await ctx.resume(); return; }
  ctx = new AudioContext({ sampleRate: 24000, latencyHint: "interactive" });
  if (ctx.sampleRate !== 24000) {
    const sr = ctx.sampleRate; await ctx.close(); ctx = null;
    throw new Error(`this browser cannot run audio at 24 kHz (got ${sr} Hz); please use Chrome or Edge`);
  }
  await ctx.audioWorklet.addModule(new URL("static/capture-worklet.js", document.baseURI).href);
  await ctx.audioWorklet.addModule(new URL("static/playback-worklet.js", document.baseURI).href);
  playNode = new AudioWorkletNode(ctx, "playback-processor", { outputChannelCount: [1] });
  playNode.connect(ctx.destination);
  playNode.port.onmessage = (e) => {
    const m = e.data;
    if (m.type === "playing" && capFrame.has(m.j)) {
      pipeLat.push((m.t - capFrame.get(m.j)) * 1000); capFrame.delete(m.j); if (pipeLat.length > 60) pipeLat.shift();
    } else if (m.type === "stats") {
      setV("#m_buf", `${m.buffered_ms.toFixed(0)} ms`, m.underruns ? "warn" : "");
      $("#m_bufd").textContent = `target ${m.target_ms.toFixed(0)} ms · underruns ${m.underruns} · drops ${m.drops} · overflows ${m.overflows}`;
    }
  };
  sinkGain = ctx.createGain(); sinkGain.gain.value = 0; sinkGain.connect(ctx.destination);   // keeps capture pulled
}
async function start() {
  if (insecure) { log(insecure, "e"); return; }
  try {
    await ensureAudio();
    // Default: the browser's noise suppression + echo cancellation ON (a raw laptop mic can idle ~30 dB above the
    // training floor, and speakers echo back), automatic gain control OFF. "raw" turns both off.
    const proc = document.querySelector("input[name=out]:checked").value !== "raw";
    const requested = { echoCancellation: proc, noiseSuppression: proc, autoGainControl: false, channelCount: 1 };
    let audioInfo = { source: "file" };
    const useFile = document.querySelector("input[name=in]:checked").value === "file";
    if (useFile) {
      const f = $("#wavFile").files[0];
      if (!f) { $("#wavFile").click(); return; }
      const buf = await ctx.decodeAudioData(await f.arrayBuffer());
      srcNode = ctx.createBufferSource(); srcNode.buffer = buf;
      srcNode.onended = () => setTimeout(() => running && stop(), 4000);    // let the agent finish its reply
    } else {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: requested });
      const tr = micStream.getAudioTracks()[0];
      audioInfo = { source: "mic", requested, applied: tr && tr.getSettings ? tr.getSettings() : null };
      try { srcNode = ctx.createMediaStreamSource(micStream); }
      catch (err) { throw new Error(`this browser cannot connect the mic to 24 kHz audio (${err.name}); please use Chrome or Edge`); }
    }
    capNode = new AudioWorkletNode(ctx, "capture-processor", { processorOptions: { chunk: 480 } });
    capNode.port.postMessage({ type: "enable", on: false });
    capNode.port.onmessage = (e) => {
      if (!running || !ws || ws.readyState !== 1) return;
      ws.send(e.data.pcm);
      samplesSent += e.data.pcm.byteLength / 2;
      if (samplesSent % FRAME === 0) capFrame.set(samplesSent / FRAME - 1, e.data.t);   // input frame k completes here
    };
    srcNode.connect(capNode); capNode.connect(sinkGain);
    playNode.port.postMessage({ type: "reset" });
    capFrame = new Map(); samplesSent = 0; pipeLat = [];
    $("#transcript").innerHTML = ""; $("#downloads").innerHTML = ""; curTurn = null;
    $("#inputWarn").innerHTML = ""; $("#inputWarn").hidden = true;
    ws.send(JSON.stringify({ type: "start", audio: audioInfo }));          // recorded in the session meta
    $("#startBtn").disabled = true;
  } catch (err) { log("cannot start: " + err.message, "e"); teardownInput(); }
}
function onReady(m) {
  running = true; capFrames = m.cap_frames; modelLagMs = m.model_lag_ms || 0;
  $("#sessionInfo").textContent = `${m.sid} · floor ${m.floor.level_db ?? "off"} dBFS · cap ${(m.cap_frames * 0.08 / 60).toFixed(1)} min`;
  capNode.port.postMessage({ type: "enable", on: true });
  if (srcNode && srcNode.start && srcNode.buffer) srcNode.start();
  setPill("live", "live: just talk; you can interrupt it at any time");
}
function stop() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "stop" }));
  teardownInput();
}
function teardownInput() {
  if (capNode) { capNode.port.postMessage({ type: "enable", on: false }); capNode.disconnect(); capNode = null; }
  if (srcNode) { try { srcNode.stop && srcNode.buffer && srcNode.stop(); } catch (_) {} srcNode.disconnect(); srcNode = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
}
function onEnded(m) {
  running = false; teardownInput();
  const meta = m.meta || {};
  log(`conversation ended (${m.reason}): ${meta.seconds} s, per-frame p90 ${meta.tick_ms?.p90} ms, ` +
      `grammar violations ${meta.validate_violations}, talker input gate ${meta.talker_input_gate?.verdict}`,
      meta.validate_violations ? "e" : "");
  const dl = $("#downloads"); dl.innerHTML = "";
  for (const u of m.downloads || []) { const a = document.createElement("a"); a.href = u; a.textContent = u.split("/").pop(); dl.append(a); }
  setPill(state === "live" ? "ready" : state, "");
}

// ── transcript + metrics ─────────────────────────────────────────────────────────────────────────────
function onText(m) {
  if (!curTurn && (m.delta || "").trim()) {
    curTurn = document.createElement("div"); curTurn.className = "turn";
    const tk = document.createElement("span"); tk.className = "tk"; tk.textContent = `${(m.k * 0.08).toFixed(1)}s`;
    curTurn.append(tk); $("#transcript").append(curTurn);
  }
  if (m.delta && curTurn) curTurn.append(document.createTextNode(m.delta));
  if (m.turn_end) curTurn = null;
  $("#transcript").scrollTop = $("#transcript").scrollHeight;
}
function onMetrics(m) {
  const t = m.tick_ms || {}, s = m.in_to_out_ms || {};
  if (t.n) {
    const rtf = t.p90 / 80;
    setV("#m_tick", `${t.p50.toFixed(0)} / ${t.p90.toFixed(0)} ms`, rtf < 0.8 ? "ok" : rtf < 1 ? "warn" : "bad");
    $("#m_rtf").textContent = `RTF ${rtf.toFixed(2)} (must be < 1) · p99 ${t.p99.toFixed(0)} ms`;
  }
  if (s.n) setV("#m_srv", `${s.p50.toFixed(0)} / ${s.p90.toFixed(0)} ms`);
  $("#m_backlog").textContent = `backlog ${m.backlog_frames} frames`;
  setV("#m_len", `${(m.frames * 0.08).toFixed(0)} s / ${(m.cap_frames * 0.08).toFixed(0)} s`,
       m.frames > 0.9 * m.cap_frames ? "warn" : "");
  const pipe = median(pipeLat);
  $("#m_lag").textContent = `plus the model text lead ≈ ${modelLagMs} ms`;
  if (pipe != null) setV("#m_pipe", `${pipe.toFixed(0)} ms`, pipe < 400 ? "ok" : pipe < 700 ? "warn" : "bad");
}

// ── live mic warnings from the server (echo / mic background): shown on top until the next conversation ──
function showInputWarn(msg) {
  const box = $("#inputWarn"), d = document.createElement("div");
  d.textContent = msg; box.append(d); box.hidden = false;
}

// ── secure context: the browser only gives the mic (getUserMedia) and audio worklets to https or http://localhost ──
function insecureReason() {
  if (window.isSecureContext && navigator.mediaDevices && typeof AudioWorkletNode !== "undefined") return null;
  const port = location.port || (location.protocol === "https:" ? "443" : "80");
  return `This address (${location.origin}) is not a secure context, so the browser blocks the microphone and audio ` +
         `processing. Forward the port to your machine with an SSH tunnel: ssh -L ${port}:localhost:${port} <gpu-box>, ` +
         `then open http://localhost:${port}${location.pathname} in Chrome or Edge.`;
}
const insecure = insecureReason();
if (insecure) { $("#secureWarn").textContent = insecure; $("#secureWarn").hidden = false; }

// ── wiring ───────────────────────────────────────────────────────────────────────────────────────────
$("#startBtn").addEventListener("click", start);
$("#stopBtn").addEventListener("click", stop);
for (const r of document.querySelectorAll("input[name=out]"))
  r.addEventListener("change", () => { $("#aecWarn").hidden = !(r.value === "raw" && r.checked); });
for (const r of document.querySelectorAll("input[name=in]"))
  r.addEventListener("change", () => { if (r.value === "file" && r.checked) $("#wavFile").click(); });
connect();
