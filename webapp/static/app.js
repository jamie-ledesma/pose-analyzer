// Analysis page: example, live camera and upload modes. Talks to the local Flask server in app.py.
// Everything on screen is drawn by render() from `state`, so each mode keeps its own result.

const $ = (id) => document.getElementById(id);
const MODES = ["example", "live", "upload"];
const LIVE_WINDOW_S = 30; // seconds of knee angle kept on the live trace
const MAX_POINTS = 600;   // longer videos are binned down to this many points on the trace

const state = {
  mode: "example",
  jobs: { example: null, upload: null },        // { id, status, label, report, filename, error }
  polls: { example: null, upload: null },
  jobSettings: { example: null, upload: null },  // the settings each mode's video was analyzed with
  traces: {},                                    // job id -> per-frame series parsed from its CSV
  aspect: { example: 9 / 16, live: 9 / 16, upload: 9 / 16 },
  file: null,
  at: { example: 0, upload: 0 },                 // frame the video is on, so counts follow playback
  live: { running: false, starting: false, poll: null, status: null, trace: [], t0: 0, error: null },
};

// ---------- settings ----------

function settings() {
  return { exercise: $("exercise").value, model: $("model").value, max_lean: $("max-lean").value, depth: $("depth").value };
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function onSettingsChanged(e) {
  $("max-lean-out").textContent = `${$("max-lean").value}°`;
  $("depth-out").textContent = `${$("depth").value}°`;
  if (state.live.running) post("/api/live/settings", settings()).catch(() => {});
  // A new exercise means a different analysis: redo the video on screen straight away.
  if (e?.target?.id === "exercise") {
    if (state.mode === "example") return runExample();
    if (state.mode === "upload" && state.file && state.jobs.upload?.status === "done") return uploadFile(state.file);
  }
  render();
}

for (const id of ["exercise", "model", "max-lean", "depth"]) $(id).addEventListener("input", onSettingsChanged);

async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// ---------- formatting ----------

const deg = (v) => (v == null ? null : `${Math.round(v)}°`);
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// "Rep 2: chest up, knees out" -> "Chest up, knees out"
function verdict(rep) {
  if (!rep.counted) return { text: "Too shallow", cls: "miss" };
  if (rep.good) return { text: "Good", cls: "good" };
  return { text: capital(rep.message.split(": ").slice(1).join(": ")), cls: "fault" };
}

// ---------- the current view, whatever the mode ----------

// What is on screen now: a finished report (with its per-frame trace), the live session, or nothing yet.
function view() {
  if (state.mode === "live") {
    const s = state.live.status;
    return { kind: "live", running: state.live.running, report: s?.report || null, fps: s?.fps, exercise: settings().exercise };
  }
  const job = state.jobs[state.mode];
  const done = job?.status === "done";
  const trace = done ? state.traces[job.id] : null;
  let report = done ? job.report : null, at = null;
  // A squat set counts up as the video plays: the report shows only what has happened by the current frame.
  if (done && trace?.frames && report.exercise === "squat") {
    at = Math.min(state.at[state.mode] || 0, trace.frames.length - 1);
    const f = trace.frames[at];
    const history = [];
    let n = 0;
    for (const rep of report.history) {
      if (rep.counted) { if (n >= f.reps) break; n++; history.push(rep); }
      else if (f.reps >= Math.max(1, n)) history.push(rep);
    }
    report = { ...report, reps: f.reps, phase: f.phase, history };
  }
  return {
    kind: "job",
    job,
    done,
    report,
    full: done ? job.report : null,
    at,
    trace,
    used: state.jobSettings[state.mode] || settings(),
    exercise: done ? job.report.exercise : settings().exercise,
  };
}

function render() {
  const v = view();
  const squat = v.exercise === "squat";
  for (const btn of document.querySelectorAll(".mode")) {
    if (btn.dataset.mode === state.mode) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  }
  const grid = $("grid");
  grid.classList.toggle("landscape", state.aspect[state.mode] > 1);
  grid.classList.toggle("no-reps", !squat);
  $("reps-panel").hidden = !squat;
  renderMedia();
  renderHeader(v, squat);
  renderSummary(v, squat);
  renderTrace(v, squat);
  renderReps(v, squat);
  renderThresholds(squat);
}

function renderMedia() {
  const mode = state.mode;
  const show = (id, on) => ($(id).hidden = !on);
  const job = state.jobs[mode];
  const status = mode === "live" ? null : job?.status;
  show("example-video", mode === "example" && status === "done");
  show("upload-video", mode === "upload" && status === "done");
  show("live-img", mode === "live" && state.live.running);
  show("live-off", mode === "live" && !state.live.running);
  show("progress", mode !== "live" && (status === "running" || status === "uploading"));
  show("dropzone", mode === "upload" && (!job || status === "error"));
  const error = mode === "live" ? state.live.error : status === "error" ? job.error : null;
  $("media-error").textContent = error || "";
  show("media-error", Boolean(error));
  $("live-start").disabled = state.live.starting;
  for (const m of ["example", "upload"]) if (m !== mode) $(`${m}-video`).pause();
}

function renderHeader(v, squat) {
  const mode = state.mode;
  const job = v.job;
  $("title").textContent = mode === "example" ? "Example set" : mode === "live" ? "Live set"
    : job?.filename || "Upload a video";

  let finding = "";
  if (v.kind === "live") {
    const r = v.report;
    if (!v.running && squat && r?.history?.length) finding = `Last set: ${setFinding(r)}`;
    else if (!v.running) finding = "Start the camera, then squat facing it. Nothing is recorded or saved.";
    else if (!squat) finding = "Tracking both elbow angles.";
    else if (r?.reps) finding = `${plural(r.reps, "rep")} so far. Last: ${verdict(r.history.at(-1)).text.toLowerCase()}.`;
    else finding = "Squat facing the camera. A rep counts once your knee passes the depth line.";
  } else if (v.done) {
    finding = squat ? setFinding(v.full) : `Pose found in ${v.report.detected} of ${v.report.frames} frames.`;
  } else if (job?.status === "uploading") finding = `Uploading ${job.filename}`;
  else if (job?.status === "running") finding = mode === "example" ? `${job.label}.` : `Analyzing ${job.filename}`;
  else if (job?.status === "error") finding = "That video could not be analyzed.";
  else finding = "Film a squat from the front on your phone and drop it in. It is analyzed on this computer.";
  $("finding").textContent = finding;

  const primary = $("primary");
  primary.hidden = false;
  if (mode === "example") primary.textContent = "Try it live";
  else if (mode === "live") {
    primary.hidden = !state.live.running;
    primary.textContent = "Stop camera";
  } else {
    primary.hidden = job?.status !== "done";
    primary.textContent = "Upload another";
  }

  const done = v.kind === "job" && v.done;
  $("dl-video").hidden = $("dl-csv").hidden = !done;
  if (done) {
    $("dl-video").href = `/runs/${job.id}/result.mp4?download`;
    $("dl-csv").href = `/runs/${job.id}/data.csv?download`;
  }
  $("title-actions").hidden = primary.hidden && !done;
}

// "2 reps counted, both with good form. Deepest knee 83°."
function setFinding(report) {
  const reps = report.history.filter((r) => r.counted);
  const shallow = report.history.length - reps.length;
  if (!report.history.length) return "No full squats found in this video.";
  const good = reps.filter((r) => r.good).length;
  let s = `${plural(reps.length, "rep")} counted`;
  if (shallow) s += `, ${shallow} too shallow`;
  if (reps.length) {
    if (good === reps.length) s += reps.length === 1 ? ", with good form" : reps.length === 2 ? ", both with good form" : ", all with good form";
    else s += `, ${good} of ${reps.length} with good form`;
  }
  const deepest = Math.min(...report.history.map((r) => r.bottom_knee));
  return `${s}. Deepest knee ${Math.round(deepest)}°.`;
}

function kpi(i, label, valueHtml, sub = "", subClass = "") {
  $(`k${i}-label`).textContent = label;
  const value = $(`k${i}-value`);
  value.innerHTML = valueHtml ?? "–";
  value.classList.toggle("dash", valueHtml == null);
  const s = $(`k${i}-sub`);
  s.textContent = sub;
  s.classList.toggle("bad", subClass === "bad");
}

function renderSummary(v, squat) {
  const r = v.report;
  if (v.kind === "live") {
    const running = v.running && r;
    if (squat) {
      const last = r?.history ? r : null; // after Stop, the last set's counts stay
      const counted = last ? r.history.filter((x) => x.counted) : [];
      const faults = running ? r.faults || [] : [];
      kpi(0, "Reps counted", last ? String(r.reps) : null, running ? capital(r.phase) : last ? "camera off" : "");
      kpi(1, "Knee now", running ? deg(r.knee) : null, `depth line ${settings().depth}°`);
      kpi(2, "Torso lean now", running ? deg(r.torso_lean) : null,
        faults.length ? `Now: ${faults.join(", ").toLowerCase()}` : `limit ${settings().max_lean}°`, faults.length ? "bad" : "");
      kpi(3, "Good form", last ? `${counted.filter((x) => x.good).length} of ${counted.length}` : null,
        running && v.fps ? `analyzing at ${Math.round(v.fps)} fps` : "");
    } else {
      kpi(0, "Left elbow", running ? deg(r.angles?.left_elbow) : null);
      kpi(1, "Right elbow", running ? deg(r.angles?.right_elbow) : null);
      kpi(2, "Frames with a pose", running ? `${r.detected} of ${r.frames}` : null);
      kpi(3, "Analyzing at", running && v.fps ? `${Math.round(v.fps)}<span class="unit">fps</span>` : null);
    }
    return;
  }
  const used = v.used;
  if (squat) {
    const counted = r ? r.history.filter((x) => x.counted) : [];
    const faultCounts = {};
    for (const rep of counted) if (!rep.good) for (const f of verdict(rep).text.split(", ")) faultCounts[f.toLowerCase()] = (faultCounts[f.toLowerCase()] || 0) + 1;
    const faultText = Object.entries(faultCounts).map(([f, n]) => `${n} ${f}`).join(", ");
    const deepest = r?.history.length ? Math.min(...r.history.map((x) => x.bottom_knee)) : null;
    const leans = v.at != null ? v.trace.frames.slice(0, v.at + 1).map((f) => f.torso) : v.trace?.torso || [];
    const maxLean = leans.some((x) => x != null) ? Math.max(...leans.filter((x) => x != null)) : null;
    kpi(0, "Reps counted", r ? String(counted.length) : null,
      v.at != null ? capital(r.phase) : r ? `of ${plural(r.history.length, "attempt")}` : "");
    kpi(1, "Deepest knee", deg(deepest), `depth line ${used.depth}°`);
    kpi(2, "Good form", r && counted.length ? `${counted.filter((x) => x.good).length} of ${counted.length}` : null,
      r ? (faultText || (counted.length ? "no faults" : "")) : "", faultText ? "bad" : "");
    kpi(3, "Most torso lean", deg(Number.isFinite(maxLean) ? maxLean : null), `limit ${used.max_lean}°`,
      maxLean > used.max_lean ? "bad" : "");
  } else {
    const range = (xs) => {
      const vals = xs?.filter((x) => x != null) || [];
      return vals.length ? `${Math.round(Math.min(...vals))}–${Math.round(Math.max(...vals))}°` : null;
    };
    const seconds = r ? r.frames / r.video.fps : null;
    kpi(0, "Pose found", r ? `${Math.round((100 * r.detected) / Math.max(1, r.frames))}<span class="unit">%</span>` : null,
      r ? `${r.detected} of ${r.frames} frames` : "");
    kpi(1, "Left elbow range", range(v.trace?.left));
    kpi(2, "Right elbow range", range(v.trace?.right));
    kpi(3, "Length", seconds == null ? null : `${seconds.toFixed(1)}<span class="unit">s</span>`, r ? `${Math.round(r.video.fps)} fps` : "");
  }
}

// ---------- the knee-angle trace ----------

function renderTrace(v, squat) {
  const el = $("chart");
  const empty = $("chart-empty");
  $("chart-title").textContent = squat ? "Knee angle" : "Elbow angles";
  const depth = Number(settings().depth);
  let data = null;

  if (v.kind === "live" && state.live.trace.length > 1) {
    const pts = state.live.trace;
    data = {
      labels: pts.map((p) => clock(p.t)),
      xAt: pts.map((p, i) => (i === 0 || Math.floor(p.t / 5) !== Math.floor(pts[i - 1].t / 5) ? i : -1)).filter((i) => i >= 0),
      series: squat ? [{ name: "Knee angle", values: pts.map((p) => p.knee) }]
        : [{ name: "Left", values: pts.map((p) => p.left) }, { name: "Right", values: pts.map((p) => p.right) }],
      annotations: [],
    };
  } else if (v.kind === "job" && v.trace) {
    const t = v.trace;
    data = { labels: t.labels, xAt: t.xAt, annotations: [] };
    if (squat) {
      data.series = [{ name: "Knee angle", values: t.knee }];
      const counted = v.report.history.filter((x) => x.counted);
      data.annotations = t.repBottoms.slice(0, counted.length).map((at, k) => ({
        at, label: `Rep ${k + 1} · ${Math.round(counted[k].bottom_knee)}°`, short: `${k + 1}`,
      })).filter((a) => a.at != null);
    } else data.series = [{ name: "Left", values: t.left }, { name: "Right", values: t.right }];
  }

  $("chart-meta").textContent = squat ? (data ? "Lower is deeper. A rep counts once you stand back up." : "") : "";
  if (!data || !data.series[0].values.some((x) => x != null)) {
    el.hidden = true;
    empty.hidden = false;
    empty.textContent = v.kind === "live"
      ? (v.running ? "Waiting for a pose. Step back until your whole body is in frame." : "Start the camera to trace your angles as you move.")
      : v.job?.status === "running" || v.job?.status === "uploading" ? "The trace appears once the analysis finishes."
      : "Upload a video to trace the angles over the set.";
    return;
  }
  el.hidden = false;
  empty.hidden = true;
  const all = data.series.flatMap((s) => s.values).filter((x) => x != null);
  const lo = Math.min(60, Math.floor(Math.min(...all) / 30) * 30);
  const ticks = [];
  for (let y = lo; y <= 180; y += 30) ticks.push(y);
  Charts.line(el, {
    labels: data.labels,
    series: data.series,
    today: v.at != null ? Math.floor(v.at / v.trace.bin) : null, // the line after the playhead is dashed
    todayLabel: "",
    xAt: data.xAt,
    yTicks: ticks,
    height: squat ? 180 : () => Math.max(180, el.clientHeight),
    format: (x) => `${Math.round(x)}°`,
    reference: squat ? { value: depth, label: `Depth ${depth}°` } : undefined,
    annotations: data.annotations,
    legend: squat ? null : $("chart-meta"),
  });
}

// Per-frame series from a job's CSV, binned to at most MAX_POINTS (each bin keeps its lowest knee angle).
function parseTrace(csvText, report) {
  const [head, ...lines] = csvText.trim().split("\n");
  const cols = head.split(",");
  const idx = (name) => cols.indexOf(name);
  const num = (row, name) => (idx(name) < 0 || row[idx(name)] === "" ? null : Number(row[idx(name)]));
  const rows = lines.map((l) => l.split(","));
  const fps = report.video.fps;
  const bin = Math.max(1, Math.ceil(rows.length / MAX_POINTS));
  const out = { labels: [], knee: [], torso: [], left: [], right: [], repBottoms: [], repAt: [], repLean: [], xAt: [], bin, fps,
    frames: rows.map((r) => ({ reps: Number(r[idx("reps")] || 0), phase: r[idx("phase")] || "standing", torso: num(r, "torso_lean") })) };
  const mean = (a, b) => (a == null ? b : b == null ? a : (a + b) / 2);
  const lowest = (xs) => { const v = xs.filter((x) => x != null); return v.length ? Math.min(...v) : null; };
  const highest = (xs) => { const v = xs.filter((x) => x != null); return v.length ? Math.max(...v) : null; };
  for (let i = 0; i < rows.length; i += bin) {
    const chunk = rows.slice(i, i + bin);
    out.labels.push(clock(i / fps));
    out.knee.push(lowest(chunk.map((r) => mean(num(r, "knee_left"), num(r, "knee_right")))));
    out.torso.push(highest(chunk.map((r) => num(r, "torso_lean"))));
    out.left.push(lowest(chunk.map((r) => num(r, "left_elbow_shown"))));
    out.right.push(lowest(chunk.map((r) => num(r, "right_elbow_shown"))));
  }
  // x labels on one even step of whole seconds
  const seconds = rows.length / fps;
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => seconds / s <= 6) || 600;
  for (let s = 0; s <= seconds; s += step) out.xAt.push(Math.min(out.labels.length - 1, Math.round((s * fps) / bin)));
  // where each counted rep bottomed out: the lowest knee between one rep count and the next
  if (idx("reps") >= 0) {
    let start = 0, prev = 0;
    rows.forEach((r, i) => {
      const reps = Number(r[idx("reps")] || 0);
      if (reps > prev) {
        let best = null, lean = null;
        for (let j = start; j <= i; j++) {
          const k = mean(num(rows[j], "knee_left"), num(rows[j], "knee_right"));
          if (k != null && (best == null || k < best.k)) best = { k, j };
          const t = num(rows[j], "torso_lean");
          if (t != null && (lean == null || t > lean)) lean = t;
        }
        out.repBottoms.push(best ? Math.floor(best.j / bin) : null);
        out.repAt.push(best ? clock(best.j / fps) : null);
        out.repLean.push(lean);
        start = i + 1;
        prev = reps;
      }
    });
  }
  return out;
}

// ---------- reps and set-up ----------

function renderReps(v, squat) {
  if (!squat) return;
  const history = v.report?.history || [];
  const t = v.kind === "job" ? v.trace : null;
  const dash = '<span class="dash">–</span>';
  let n = 0;
  $("rep-list").innerHTML = history.map((rep) => {
    const k = rep.counted ? n++ : null;
    const { text, cls } = verdict(rep);
    const lean = k != null && t?.repLean[k] != null ? `${Math.round(t.repLean[k])}°` : dash;
    const at = k != null && t?.repAt[k] ? t.repAt[k] : dash;
    return `<tr><td>${rep.counted ? `Rep ${k + 1}` : "Attempt"}</td><td><span class="pill ${cls}">${escapeHtml(text)}</span></td>`
      + `<td class="num">${Math.round(rep.bottom_knee)}°</td><td class="num">${lean}</td><td class="num">${at}</td></tr>`;
  }).join("");
  $("rep-table").parentElement.hidden = history.length === 0;
  $("reps-empty").hidden = history.length > 0;
  $("reps-empty").textContent = v.kind === "live" && v.running ? "Reps appear here as you finish them." : "Completed reps appear here.";
  const scroll = $("rep-table").parentElement;
  scroll.scrollTop = scroll.scrollHeight;
}

function renderThresholds(squat) {
  const mode = state.mode;
  const job = state.jobs[mode];
  const analyzed = state.jobSettings[mode];
  const canRerun = mode !== "live" && analyzed && job?.status !== "running" && job?.status !== "uploading"
    && (mode === "example" || state.file) && !same(settings(), analyzed);
  // Beside the sliders the button keeps its space while hidden, so the tracks don't move under a drag.
  $("rerun").hidden = !squat && !canRerun;
  $("rerun").style.visibility = canRerun ? "" : "hidden";
  $("live-reset").hidden = !(mode === "live" && state.live.running && squat);
  $("squat-settings").hidden = !squat;
  $("squat-settings").parentElement.hidden = !squat && !canRerun;
}

// ---------- mode switching ----------

function setMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === "live" && state.live.running) stopLive(); // the camera turns off when leaving live mode
  state.mode = mode;
  render();
  const video = $(`${mode}-video`);
  if (video && !video.hidden) video.play().catch(() => {});
}

for (const btn of document.querySelectorAll(".mode")) btn.addEventListener("click", () => setMode(btn.dataset.mode));

$("primary").addEventListener("click", () => {
  if (state.mode === "example") setMode("live");
  else if (state.mode === "live") stopLive();
  else resetUpload();
});

// The media slot takes the shape of what plays in it: portrait clips beside the data, landscape ones above it.
function setAspect(mode, w, h) {
  if (!w || !h) return;
  const a = w / h;
  if (Math.abs(a - state.aspect[mode]) > 0.01) {
    state.aspect[mode] = a;
    if (state.mode === mode) render();
  }
}
for (const m of ["example", "upload"]) {
  const video = $(`${m}-video`);
  video.addEventListener("loadedmetadata", () => setAspect(m, video.videoWidth, video.videoHeight));
}

// ---------- live mode ----------

async function startLive() {
  state.live.error = null;
  state.live.starting = true;
  render();
  try {
    await post("/api/live/start", settings());
  } catch (err) {
    state.live.starting = false;
    state.live.error = err.message;
    return render();
  }
  Object.assign(state.live, { running: true, starting: false, status: null, trace: [], t0: performance.now() });
  $("live-img").src = `/live.mjpg?t=${Date.now()}`;
  state.live.poll = setInterval(pollLive, 250);
  render();
}

async function stopLive() {
  clearInterval(state.live.poll);
  state.live.running = false; // the last set's reps and trace stay on screen until the next start
  $("live-img").removeAttribute("src");
  render();
  await post("/api/live/stop").catch(() => {});
}

async function pollLive() {
  const res = await fetch("/api/live/status").catch(() => null);
  if (!res || !res.ok || !state.live.running) return;
  const status = await res.json();
  if (!status.running) {
    await stopLive();
    if (status.error) { state.live.error = status.error; render(); }
    return;
  }
  state.live.status = status;
  const r = status.report || {};
  const t = (performance.now() - state.live.t0) / 1000;
  state.live.trace.push({ t, knee: r.knee ?? null, left: r.angles?.left_elbow ?? null, right: r.angles?.right_elbow ?? null });
  while (state.live.trace.length && state.live.trace[0].t < t - LIVE_WINDOW_S) state.live.trace.shift();
  const img = $("live-img");
  setAspect("live", img.naturalWidth, img.naturalHeight);
  if (state.mode === "live") render();
}

$("live-start").addEventListener("click", startLive);
$("live-reset").addEventListener("click", () => { state.live.trace = []; post("/api/live/reset"); });
window.addEventListener("pagehide", () => { if (state.live.running) navigator.sendBeacon("/api/live/stop"); });

// ---------- analysis jobs (example and upload modes) ----------

function setProgress(label, fraction) {
  $("progress-label").textContent = label;
  $("progress-bar").style.width = `${Math.round(fraction * 100)}%`;
  $("progress-note").textContent = `${Math.round(fraction * 100)}%`;
}

function watchJob(mode, id, label, filename) {
  clearInterval(state.polls[mode]);
  state.jobs[mode] = { id, status: "running", label, filename };
  setProgress(label, 0);
  state.polls[mode] = setInterval(() => pollJob(mode, id), 400);
  render();
  pollJob(mode, id);
}

async function pollJob(mode, id) {
  const res = await fetch(`/api/jobs/${id}`).catch(() => null);
  if (state.jobs[mode]?.id !== id || !res || !res.ok) return; // a newer job replaced this one
  const job = await res.json();
  if (job.status === "running") {
    if (state.mode === mode) setProgress(state.jobs[mode].label, job.progress);
    return;
  }
  clearInterval(state.polls[mode]);
  state.polls[mode] = null;
  if (job.status === "error") return jobFailed(mode, job.error);
  if (!state.traces[job.id] || mode === "example") {
    const csv = await fetch(`/runs/${job.id}/data.csv?t=${Date.now()}`).then((r) => (r.ok ? r.text() : "")).catch(() => "");
    if (csv) state.traces[job.id] = parseTrace(csv, job.report);
  }
  state.jobs[mode] = { ...job, filename: state.jobs[mode].filename || job.filename };
  const video = $(`${mode}-video`);
  video.poster = `/runs/${job.id}/poster.jpg?t=${Date.now()}`;
  video.src = `/runs/${job.id}/result.mp4?t=${Date.now()}`;
  render();
  if (state.mode === mode) video.play().catch(() => {});
}

function jobFailed(mode, message) {
  clearInterval(state.polls[mode]);
  state.polls[mode] = null;
  state.jobs[mode] = { id: null, status: "error", error: message, filename: state.jobs[mode]?.filename };
  render();
}

$("rerun").addEventListener("click", () => {
  if (state.mode === "example") runExample();
  else if (state.mode === "upload" && state.file) uploadFile(state.file);
});

async function runExample() {
  state.jobSettings.example = settings();
  try {
    const job = await post("/api/example", state.jobSettings.example);
    watchJob("example", job.id, "Re-analyzing the example");
  } catch (err) {
    jobFailed("example", err.message);
  }
}

// ---------- upload mode ----------

const dropzone = $("dropzone");
$("choose-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => e.target.files[0] && uploadFile(e.target.files[0]));
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

function resetUpload() {
  clearInterval(state.polls.upload);
  state.polls.upload = null;
  state.jobs.upload = null;
  state.jobSettings.upload = null;
  state.file = null;
  state.aspect.upload = 9 / 16;
  $("file-input").value = "";
  $("upload-video").pause();
  $("upload-video").removeAttribute("src");
  $("upload-video").removeAttribute("poster");
  render();
}

function uploadFile(file) {
  clearInterval(state.polls.upload);
  state.polls.upload = null;
  state.file = file;
  state.jobSettings.upload = settings();
  state.jobs.upload = { id: null, status: "uploading", filename: file.name, label: "Uploading" };
  $("upload-video").pause();
  setProgress("Uploading", 0);
  render();

  const form = new FormData();
  form.append("video", file);
  for (const [k, val] of Object.entries(state.jobSettings.upload)) form.append(k, val);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => e.lengthComputable && state.jobs.upload?.status === "uploading" && setProgress("Uploading", e.loaded / e.total);
  xhr.onload = () => {
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status !== 202) return jobFailed("upload", data.error || `Upload failed (${xhr.status})`);
    watchJob("upload", data.id, "Analyzing", file.name);
  };
  xhr.onerror = () => jobFailed("upload", "Upload failed. Is the server still running?");
  xhr.send(form);
}

// ---------- playback sync ----------
// The counters, rep table and chart follow the frame the video is on, and reset when it loops or is scrubbed back.

setInterval(() => {
  const mode = state.mode;
  if (mode === "live") return;
  const video = $(`${mode}-video`);
  const job = state.jobs[mode];
  const trace = job?.status === "done" ? state.traces[job.id] : null;
  if (!trace?.frames || video.hidden) return;
  const at = Math.max(0, Math.min(trace.frames.length - 1, Math.floor(video.currentTime * trace.fps)));
  if (at !== state.at[mode]) {
    state.at[mode] = at;
    render();
  }
}, 100);

// ---------- start ----------

onSettingsChanged();
state.jobSettings.example = settings(); // the server prepares the example with the default settings
watchJob("example", "example", "Preparing the example");
