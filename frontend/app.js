// AirMouse UI. A "source" feeds hello/state/config messages (the backend's WebSocket
// protocol, see airmouse/server.py). In the app window the page is opened with ?token=...
// and uses LiveSource; opened on its own it falls back to DemoSource, which replays a real
// recorded hand.

"use strict";

const STATES = {
  NO_HAND: ["Show your hand", "Hold it up, palm facing the screen."],
  POINT: ["Pointing", "Your hand moves the cursor."],
  TOUCH: ["Touching", "Open your fingers to click."],
  PINCH: ["Pinching", "Double-clicked. Open your fingers to continue."],
  LEFT_PRESS: ["Tapping", "Pull back to click."],
  RIGHT_PRESS: ["Thumb folded", "Bring it back out to right-click."],
  GRAB: ["Holding", "Open your hand to drop it."],
  SCROLL: ["Scrolling", "Hold above the line to scroll up, below it to scroll down."],
  PAUSED: ["Paused, camera off", "Hand control and the camera are off. Turn them back on "
           + "with the switch above or Ctrl Alt M."],
  OFFLINE: ["Reconnecting", "Lost contact with AirMouse. Trying again…"],
  FITTING: ["Fitting your area", "Move your hand slowly to every edge you can comfortably reach."],
  FITTED: ["Your area is ready", "The solid box fits your reach. Use it or keep the current one."],
  FIT_FAILED: ["Couldn't fit the area", ""],
};

const MODE_TO_GESTURE = {
  POINT: "move", TOUCH: "click", PINCH: "double", LEFT_PRESS: "tap", RIGHT_PRESS: "right",
  GRAB: "drag", SCROLL: "scroll",
};

const EVENT_TOAST = {
  LEFT_CLICK: "Clicked", DOUBLE_CLICK: "Double-clicked", RIGHT_CLICK: "Right-clicked", LEFT_DOWN: "Picked up", LEFT_UP: "Dropped",
};

// Landmarks each gesture's icon rings: the fingertip(s) or part of the hand that acts.
const GLYPH_FOCUS = {
  move: [8], click: [4, 8, 12, 16, 20], double: [4, 8], tap: [8], right: [4], drag: [5, 9, 13, 17],
  scroll: [8, 12],
};

const ACTIVE_MODES = new Set(["TOUCH", "PINCH", "LEFT_PRESS", "RIGHT_PRESS", "GRAB"]);

// Settings shown up front; the rest go under "More settings". Labels/ranges come from the
// backend's schema (airmouse/settings.py).
const MAIN_SETTINGS = ["cursor_speed", "push_threshold", "double_click_window"];

const DEMO_SCHEMA = [
  { name: "cursor_speed", label: "Cursor speed", min: 1, max: 5, step: 0.1,
    help: "Higher = less hand movement to cross the screen" },
  { name: "push_threshold", label: "Click sensitivity", min: 0.1, max: 0.8, step: 0.01,
    help: "How far a finger must push forward to click; lower = easier" },
  { name: "double_click_window", label: "Double-click time", min: 0.5, max: 2.5, step: 0.1,
    help: "Two clicks on the same spot within this many seconds become a double click" },
  { name: "smoothing_alpha", label: "Smoothing", min: 0.05, max: 1, step: 0.01,
    help: "Lower = steadier but laggier cursor" },
  { name: "area_height", label: "Control area height", min: 0.25, max: 0.8, step: 0.01,
    help: "Smaller = less up-and-down hand movement to reach the top and bottom" },
  { name: "area_center_x", label: "Control area side", min: 0.25, max: 0.75, step: 0.01,
    help: "Where your hand works left to right in the camera view" },
  { name: "area_center_y", label: "Control area position", min: 0.25, max: 0.75, step: 0.01,
    help: "Where your hand works in the camera view" },
  { name: "scroll_speed", label: "Scroll speed", min: 200, max: 5000, step: 50, integer: true,
    help: "How far a hand movement scrolls" },
  { name: "fingertip_follow", label: "Fingertip follow", min: 0, max: 1, step: 0.05,
    help: "0 keeps the cursor still while fingers gesture; 1 follows the fingertip exactly" },
];

const DEMO_CONFIG = {
  cursor_speed: 2.2, push_threshold: 0.35, double_click_window: 1.5, smoothing_alpha: 0.25,
  area_height: 0.44, area_center_x: 0.5, area_center_y: 0.36, scroll_speed: 1500, fingertip_follow: 0,
};

// ---- Sources -------------------------------------------------------------------------

class DemoSource {
  constructor(onMessage) {
    this.onMessage = onMessage;
    this.enabled = true;
  }

  async start() {
    const clip = await fetch("demo-hand.json").then((r) => r.json());
    this.clip = clip;
    this.onMessage({
      type: "hello", connections: clip.connections, schema: DEMO_SCHEMA,
      config: { ...DEMO_CONFIG }, hotkeys: { toggle: "Ctrl+Alt+M" },
    });
    this.t0 = performance.now();
    this.index = 0;
    requestAnimationFrame((now) => this.tick(now));
  }

  tick(now) {
    const frames = this.clip.frames;
    const duration = frames[frames.length - 1].t;
    const t = ((now - this.t0) / 1000) % duration;
    if (t < frames[this.index].t) this.index = 0;              // looped
    while (this.index + 1 < frames.length && frames[this.index + 1].t <= t) {
      this.index += 1;
      this.emit(frames[this.index]);
    }
    requestAnimationFrame((n) => this.tick(n));
  }

  emit(f) {
    const [fw, fh] = this.clip.frame_size;
    this.onMessage({
      type: "state", fps: 30, enabled: this.enabled, error: null, notice: null,
      frame_size: [fw, fh], active_region: [0.27, 0.14, 0.73, 0.58],
      taskbar_zone: { left: 0, top: 0, right: 0, bottom: 0.15 },
      hand: f.lm ? { handedness: "Right", score: 0.97, landmarks: f.lm } : null,
      mode: this.enabled ? f.mode : "NO_HAND", events: this.enabled ? f.events : [],
    });
  }

  send(msg) {
    if (msg.type === "set_enabled") this.enabled = msg.enabled;
  }
}

class LiveSource {
  constructor(onMessage, onFrame, token) {
    this.onMessage = onMessage;
    this.onFrame = onFrame;
    this.url = `ws://${location.host}/?token=${encodeURIComponent(token)}`;
    this.retry = 500;
  }

  start() {
    const ws = new WebSocket(this.url);
    ws.binaryType = "blob";
    ws.onopen = () => { this.retry = 500; };
    ws.onmessage = (e) => {
      if (typeof e.data === "string") this.onMessage(JSON.parse(e.data));
      else this.onFrame(e.data);
    };
    ws.onclose = () => {
      this.onMessage({ type: "offline" });
      setTimeout(() => this.start(), this.retry);
      this.retry = Math.min(this.retry * 2, 5000);
    };
    this.ws = ws;
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }
}

// ---- Plate rendering -------------------------------------------------------------------

const plate = document.querySelector(".plate");
const canvas = document.getElementById("plate");
const ctx = canvas.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let connections = [];
let latest = null;
let ripples = [];
let video = null;          // latest camera frame (ImageBitmap), mirrored by the backend
let trail = [];            // aim points seen during "fit area to my reach"

function resizeCanvas() {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * devicePixelRatio);
  canvas.height = Math.round(r.height * devicePixelRatio);
}

// The camera frame is fitted to cover the plate (like object-fit: cover).
function frameToPlate(x, y, frameSize) {
  const [fw, fh] = frameSize;
  const scale = Math.max(canvas.width / fw, canvas.height / fh);
  const ox = (canvas.width - fw * scale) / 2;
  const oy = (canvas.height - fh * scale) / 2;
  return [ox + x * fw * scale, oy + y * fh * scale];
}

// The camera as a cyanotype: greyscale, toned blue, shadows lifted toward Prussian ink,
// a little darker overall so the drawn hand reads as the brightest thing on the plate.
function drawVideo(s) {
  const [fw, fh] = s.frame_size;
  const scale = Math.max(canvas.width / fw, canvas.height / fh);
  const w = fw * scale;
  const h = fh * scale;
  const x = (canvas.width - w) / 2;
  const y = (canvas.height - h) / 2;
  ctx.save();
  ctx.filter = "grayscale(1) contrast(1.15) brightness(0.95)";
  ctx.drawImage(video, x, y, w, h);
  ctx.filter = "none";
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = "#3b78c4";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "screen";
  ctx.fillStyle = "rgba(19, 49, 92, 0.55)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = "rgba(19, 49, 92, 0.25)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const s = latest;
  if (s && s.frame_size) {
    const dpr = devicePixelRatio;
    const active = ACTIVE_MODES.has(s.mode);
    if (video) drawVideo(s);

    // Control area: where hand motion maps onto the screen.
    const [ax0, ay0] = frameToPlate(s.active_region[0], s.active_region[1], s.frame_size);
    const [ax1, ay1] = frameToPlate(s.active_region[2], s.active_region[3], s.frame_size);
    ctx.setLineDash([6 * dpr, 6 * dpr]);
    ctx.lineWidth = 1.25 * dpr;
    ctx.strokeStyle = "rgba(234, 242, 248, 0.35)";
    ctx.strokeRect(ax0, ay0, ax1 - ax0, ay1 - ay0);
    ctx.setLineDash([]);

    // Scroll line: where the hand was when scrolling began. Above = up, below = down.
    if (s.scroll_anchor && s.mode === "SCROLL") {
      const [, ly] = frameToPlate(0, s.scroll_anchor[0], s.frame_size);
      const [, dy0] = frameToPlate(0, s.scroll_anchor[0] - s.scroll_anchor[1], s.frame_size);
      ctx.fillStyle = "rgba(234, 242, 248, 0.10)";
      ctx.fillRect(ax0, dy0, ax1 - ax0, 2 * (ly - dy0));
      ctx.lineWidth = 2 * dpr;
      ctx.strokeStyle = "#eaf2f8";
      ctx.beginPath();
      ctx.moveTo(ax0, ly);
      ctx.lineTo(ax1, ly);
      ctx.stroke();
      ctx.fillStyle = "rgba(234, 242, 248, 0.75)";
      ctx.font = `500 ${12 * dpr}px "Schibsted Grotesk", "Segoe UI", sans-serif`;
      ctx.textBaseline = "bottom";
      ctx.fillText("Scroll line", ax0 + 8 * dpr, dy0 - 4 * dpr);
    }

    // Taskbar zone: the slice of the hand area that maps onto the taskbar strip.
    const zones = s.taskbar_zone || {};
    const zw = ax1 - ax0;
    const zh = ay1 - ay0;
    const bands = {
      bottom: [ax0, ay1 - (zones.bottom || 0) * zh, zw, (zones.bottom || 0) * zh],
      top: [ax0, ay0, zw, (zones.top || 0) * zh],
      left: [ax0, ay0, (zones.left || 0) * zw, zh],
      right: [ax1 - (zones.right || 0) * zw, ay0, (zones.right || 0) * zw, zh],
    };
    for (const [edge, [bx, by, bw, bh]] of Object.entries(bands)) {
      if (!(zones[edge] > 0)) continue;
      ctx.fillStyle = "rgba(234, 242, 248, 0.09)";
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = "rgba(234, 242, 248, 0.6)";
      ctx.font = `500 ${12 * dpr}px "Schibsted Grotesk", "Segoe UI", sans-serif`;
      ctx.textBaseline = "middle";
      ctx.fillText("Taskbar", bx + 8 * dpr, by + Math.min(bh, 24 * dpr) / 2);
    }

    const cal = s.calibration;
    if (cal && cal.active) {
      ctx.fillStyle = "rgba(234, 242, 248, 0.55)";
      for (const [x, y] of trail) {
        ctx.beginPath();
        ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (cal && cal.result) {
      const [px0, py0, px1, py1] = proposedArea(cal.result);
      const [bx0, by0] = frameToPlate(px0, py0, s.frame_size);
      const [bx1, by1] = frameToPlate(px1, py1, s.frame_size);
      ctx.lineWidth = 2.5 * dpr;
      ctx.strokeStyle = "#eaf2f8";
      ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);
    }

    if (s.hand) {
      const pts = s.hand.landmarks.map(([x, y]) => frameToPlate(x, y, s.frame_size));
      // A soft halo first, then the crisp line: light exposed onto blue paper.
      for (const [width, color] of [[9, "rgba(234, 242, 248, 0.10)"], [2.25, "#eaf2f8"]]) {
        ctx.lineWidth = width * dpr;
        ctx.lineCap = "round";
        ctx.strokeStyle = color;
        ctx.beginPath();
        for (const [a, b] of connections) {
          ctx.moveTo(...pts[a]);
          ctx.lineTo(...pts[b]);
        }
        ctx.stroke();
      }
      for (let i = 0; i < pts.length; i++) {
        const tip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
        ctx.fillStyle = "#eaf2f8";
        ctx.beginPath();
        ctx.arc(...pts[i], (tip ? 4 : 2.5) * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
      // Aim point: where the cursor is anchored (index knuckle, lifted toward the tip).
      const [kx, ky] = pts[5];
      const palm = Math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]);
      ctx.lineWidth = 2 * dpr;
      ctx.strokeStyle = active ? "#f5b83d" : "rgba(234, 242, 248, 0.9)";
      ctx.beginPath();
      ctx.arc(kx, ky - 0.78 * palm, 11 * dpr, 0, Math.PI * 2);
      ctx.stroke();
      s.aim = [kx, ky - 0.78 * palm];
      if (cal && cal.active) trail.push(s.aim);
    }

    // Click ripples: the one moment of motion, answering an action.
    const now = performance.now();
    ripples = ripples.filter((r) => now - r.t < 600);
    for (const r of ripples) {
      const k = (now - r.t) / 600;
      ctx.lineWidth = 3 * dpr * (1 - k);
      ctx.strokeStyle = `rgba(245, 184, 61, ${1 - k})`;
      ctx.beginPath();
      ctx.arc(r.x, r.y, (12 + 36 * k) * dpr, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  requestAnimationFrame(draw);
}

// The proposed control area, normalized to the frame (mirrors airmouse/cursor.py).
function proposedArea(r) {
  const w = 1 / r.cursor_speed;
  const h = Math.min(r.area_height, 1);
  const cx = Math.min(Math.max(r.area_center_x, w / 2), 1 - w / 2);
  const cy = Math.min(Math.max(r.area_center_y, h / 2), 1 - h / 2);
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

// ---- Panel -----------------------------------------------------------------------------

const stateEl = document.getElementById("state");
const hintEl = document.getElementById("state-hint");
const toastEl = document.getElementById("toast");
const fpsEl = document.getElementById("fps");
const handEl = document.getElementById("hand");
const enabledEl = document.getElementById("enabled");
let source = null;
let config = {};

function drawGlyphs(glyphs) {
  for (const li of document.querySelectorAll(".gestures li")) {
    const pts = glyphs[li.dataset.gesture];
    const svg = li.querySelector("svg");
    if (!pts) continue;
    const ns = "http://www.w3.org/2000/svg";
    for (const [a, b] of connections) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", pts[a][0]); line.setAttribute("y1", pts[a][1]);
      line.setAttribute("x2", pts[b][0]); line.setAttribute("y2", pts[b][1]);
      svg.appendChild(line);
    }
    for (const i of [4, 8, 12, 16, 20]) {
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", pts[i][0]); dot.setAttribute("cy", pts[i][1]);
      dot.setAttribute("r", 0.05);
      svg.appendChild(dot);
    }
    // Ring the part of the hand that does the work, so similar poses read apart.
    const focus = GLYPH_FOCUS[li.dataset.gesture];
    if (focus) {
      const [cx, cy] = focus.map((i) => pts[i]).reduce(
        ([sx, sy], [x, y]) => [sx + x / focus.length, sy + y / focus.length], [0, 0]);
      const ring = document.createElementNS(ns, "circle");
      ring.setAttribute("class", "focus");
      ring.setAttribute("cx", cx); ring.setAttribute("cy", cy);
      ring.setAttribute("r", focus.length > 2 ? 0.2 : 0.11);
      svg.appendChild(ring);
    }
  }
}

function formatValue(p, v) {
  if (p.name === "double_click_window") return `${v.toFixed(1)} s`;
  if (p.integer) return String(Math.round(v));
  return p.step >= 0.1 ? v.toFixed(1) : v.toFixed(2);
}

function buildSliders(schema) {
  const main = document.getElementById("sliders-main");
  const more = document.getElementById("sliders-more");
  main.replaceChildren();
  more.replaceChildren();
  const byName = Object.fromEntries(schema.map((p) => [p.name, p]));
  const ordered = [...MAIN_SETTINGS.map((n) => byName[n]).filter(Boolean),
                   ...schema.filter((p) => !MAIN_SETTINGS.includes(p.name))];
  for (const p of ordered) {
    const id = `s-${p.name}`;
    const row = document.createElement("div");
    row.className = "slider";
    row.innerHTML = `
      <label for="${id}"></label><output for="${id}"></output>
      <input type="range" id="${id}" min="${p.min}" max="${p.max}" step="${p.step}">
      <p></p>`;
    row.querySelector("label").textContent = p.label;
    row.querySelector("p").textContent = p.help || "";
    const input = row.querySelector("input");
    const out = row.querySelector("output");
    input.value = config[p.name];
    out.textContent = formatValue(p, Number(input.value));
    input.addEventListener("sync", () => { out.textContent = formatValue(p, Number(input.value)); });
    input.addEventListener("input", () => {
      out.textContent = formatValue(p, Number(input.value));
      source.send({ type: "set_config", values: { [p.name]: Number(input.value) } });
    });
    (MAIN_SETTINGS.includes(p.name) ? main : more).appendChild(row);
  }
}

function showToast(text) {
  toastEl.textContent = text;
  toastEl.classList.remove("show");
  void toastEl.offsetWidth;            // restart the animation
  toastEl.classList.add("show");
}

const fitStart = document.getElementById("fit-start");
const fitText = document.getElementById("fit-text");
const fitActions = document.getElementById("fit-actions");
const FIT_IDLE_TEXT = fitText.textContent;
let calId = null;

function updateFit(cal) {
  const id = cal ? cal.id : null;
  if (id !== calId) trail = [];
  calId = id;
  fitActions.hidden = !(cal && cal.result);
  fitStart.hidden = Boolean(cal && cal.result);
  if (!cal) {
    fitStart.textContent = "Fit area to my reach";
    fitText.textContent = FIT_IDLE_TEXT;
  } else if (cal.active) {
    fitStart.textContent = "Cancel";
    fitText.textContent = `Keep sweeping. ${Math.ceil(cal.remaining)} s left.`;
  } else if (cal.error) {
    fitStart.textContent = "Try again";
    fitText.textContent = cal.error;
  } else {
    fitText.textContent = "Solid box: fitted to your reach. Dashed: your current area.";
  }
}

function plateText(s) {
  const cal = s.calibration;
  if (cal && cal.active) {
    return [STATES.FITTING[0], `${STATES.FITTING[1]} ${Math.ceil(cal.remaining)} s`];
  }
  if (cal && cal.result) return STATES.FITTED;
  if (cal && cal.error) return [STATES.FIT_FAILED[0], cal.error];
  if (s.error) return ["Camera unavailable", s.error];
  const mode = s.enabled ? (s.hand ? s.mode : "NO_HAND") : "PAUSED";
  return STATES[mode] || STATES.POINT;
}

function onState(s) {
  latest = s;
  if (s.camera === false && video) {      // camera switched off: drop the last picture
    video.close();
    video = null;
  }
  const mode = s.enabled ? (s.hand ? s.mode : "NO_HAND") : "PAUSED";
  const [title, hint] = plateText(s);
  updateFit(s.calibration || null);
  if (s.notice) showToast(s.notice);
  if (stateEl.textContent !== title) stateEl.textContent = title;
  if (hintEl.textContent !== hint) hintEl.textContent = hint;
  plate.classList.toggle("is-active", ACTIVE_MODES.has(mode));
  fpsEl.textContent = s.fps ? `${Math.round(s.fps)} fps` : "–";
  handEl.textContent = s.hand ? `${s.hand.handedness}` : "None";
  enabledEl.checked = s.enabled;

  const gesture = MODE_TO_GESTURE[mode];
  for (const li of document.querySelectorAll(".gestures li")) {
    li.classList.toggle("active", li.dataset.gesture === gesture);
  }

  for (const ev of s.events || []) {
    if (EVENT_TOAST[ev]) showToast(EVENT_TOAST[ev]);
    if (["LEFT_CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"].includes(ev) && s.aim && !reducedMotion) {
      ripples.push({ x: s.aim[0], y: s.aim[1], t: performance.now() });
    }
  }
}

let glyphsDrawn = false;

function onMessage(msg) {
  if (msg.type === "hello") {
    connections = msg.connections;
    config = msg.config;
    buildSliders(msg.schema);
    if (typeof msg.autostart === "boolean") {
      autostartRow.hidden = false;
      autostartEl.checked = msg.autostart;
    }
    if (!glyphsDrawn) {
      glyphsDrawn = true;
      fetch("glyphs.json").then((r) => r.json()).then(drawGlyphs);
    }
  } else if (msg.type === "config") {
    config = msg.config;
    for (const [name, value] of Object.entries(config)) {
      const input = document.getElementById(`s-${name}`);
      if (input && document.activeElement !== input) {
        input.value = value;
        input.dispatchEvent(new Event("sync"));
      }
    }
  } else if (msg.type === "state") {
    onState(msg);
  } else if (msg.type === "offline") {
    latest = null;
    video = null;
    stateEl.textContent = STATES.OFFLINE[0];
    hintEl.textContent = STATES.OFFLINE[1];
    plate.classList.remove("is-active");
  } else if (msg.type === "error") {
    showToast(msg.message);
  } else if (msg.type === "autostart") {
    autostartEl.checked = msg.enabled;
  }
}

function onFrame(blob) {
  createImageBitmap(blob).then((bmp) => {
    if (video) video.close();
    video = bmp;
  }).catch(() => {});
}

enabledEl.addEventListener("change", () => {
  source.send({ type: "set_enabled", enabled: enabledEl.checked });
});

const autostartEl = document.getElementById("autostart");
const autostartRow = document.getElementById("autostart-row");
autostartEl.addEventListener("change", () => {
  source.send({ type: "set_autostart", enabled: autostartEl.checked });
});

fitStart.addEventListener("click", () => {
  const cal = latest && latest.calibration;
  source.send({ type: cal && cal.active ? "calibrate_cancel" : "calibrate" });
});
document.getElementById("fit-use").addEventListener("click", () => {
  const cal = latest && latest.calibration;
  if (cal && cal.result) source.send({ type: "set_config", values: cal.result });
  source.send({ type: "calibrate_cancel" });
  showToast("Area updated");
});
document.getElementById("fit-keep").addEventListener("click", () => {
  source.send({ type: "calibrate_cancel" });
});

document.getElementById("reset").addEventListener("click", () => {
  source.send({ type: "reset_config" });
  if (source instanceof DemoSource) {
    config = { ...DEMO_CONFIG };
    buildSliders(DEMO_SCHEMA);
  }
});

addEventListener("resize", resizeCanvas);
resizeCanvas();
requestAnimationFrame(draw);
const token = new URLSearchParams(location.search).get("token");
source = token ? new LiveSource(onMessage, onFrame, token) : new DemoSource(onMessage);
source.start();
