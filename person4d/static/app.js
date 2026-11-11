"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const stages = ["probe", "generate", "adapter", "train", "export"];
  const stageNames = { queued: "Queued", probe: "Checking source", generate: "Generating viewpoints", adapter: "Preparing character", train: "Reconstructing in 4D", export: "Preparing playback", completed: "Ready to explore" };
  const statusNames = { queued: "Queued", running: "In progress", completed: "Complete", failed: "Needs attention", cancelled: "Stopped", interrupted: "Interrupted" };
  const legacyPresets = {
    preview: { views: 12, steps: 8000 },
    balanced: { views: 24, steps: 30000 },
    quality: { views: 48, steps: 50000 },
  };
  const state = { authenticated: false, tokenRequired: false, system: null, jobs: [], selected: null, selectedId: null, selectedTab: "result", file: null, objectURL: null, duration: 0, metadataUnknown: false, uploading: false, setupUploading: false, setup: null, connected: true, pollTimer: null, polling: false, logOffset: 0, logJob: null, logText: "", viewerKey: "", viewsKey: "", toastTimer: null, previewing: false, initialSelection: true };

  class APIError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  function errorText(data, fallback) {
    const value = data?.detail ?? data?.error ?? data?.message;
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map((v) => v.msg || String(v)).join("; ");
    return fallback;
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout ?? 20000);
    const request = { credentials: "same-origin", ...options, signal: controller.signal };
    delete request.timeout;
    if (request.body && !(request.body instanceof FormData) && typeof request.body !== "string") {
      request.headers = { "Content-Type": "application/json", ...request.headers };
      request.body = JSON.stringify(request.body);
    }
    try {
      const response = await fetch(path, request);
      let data = null;
      if (response.status !== 204) {
        const type = response.headers.get("content-type") || "";
        data = type.includes("application/json") ? await response.json() : await response.text();
      }
      if (!response.ok) {
        if (response.status === 401 && path !== "/api/login") showLogin();
        throw new APIError(errorText(data, `Request failed (${response.status}).`), response.status);
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new APIError("The pod took too long to respond. Check its connection and try again.", 0);
      if (error instanceof TypeError) throw new APIError("Could not reach your pod. Check the connection and try again.", 0);
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function icon(name, className = "") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    if (className) svg.setAttribute("class", className);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function bytes(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0) return "—";
    if (number === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const exponent = Math.min(Math.floor(Math.log(number) / Math.log(1024)), 4);
    return `${(number / 1024 ** exponent).toFixed(exponent > 1 ? 1 : 0)} ${units[exponent]}`;
  }

  function seconds(value) {
    const number = Math.max(0, Number(value) || 0);
    return number >= 60 ? `${Math.floor(number / 60)}:${(number % 60).toFixed(1).padStart(4, "0")}` : `${number.toFixed(1)} s`;
  }

  function dateLabel(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function shortDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function safeURL(value) {
    if (!value || typeof value !== "string") return null;
    try {
      const url = new URL(value, location.origin);
      return url.origin === location.origin && ["http:", "https:"].includes(url.protocol) ? `${url.pathname}${url.search}${url.hash}` : null;
    } catch { return null; }
  }

  function toast(message, isError = false) {
    clearTimeout(state.toastTimer);
    $("toast-text").textContent = message;
    $("toast").classList.toggle("error", isError);
    $("toast").hidden = false;
    state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, isError ? 14000 : 7000);
  }

  function setConnected(connected) {
    state.connected = connected;
    $("connection-banner").hidden = connected;
  }

  function showLogin() {
    state.authenticated = false;
    clearTimeout(state.pollTimer);
    if (!$("login-dialog").open) $("login-dialog").showModal();
    $("access-token").focus();
  }

  function openSetup() {
    renderSetup();
    if (!$("setup-dialog").open) $("setup-dialog").showModal();
  }

  function updateSubmit() {
    const validFile = state.file && (state.metadataUnknown || (state.duration > 0 && state.duration + 0.01 >= clipDuration()));
    const ready = Boolean(state.system?.readiness);
    $("submit-job").disabled = !validFile || !ready || state.uploading;
    if (state.uploading) $("submit-help").textContent = "Keep this page open until the upload finishes.";
    else if (!state.file) $("submit-help").textContent = "Upload a video to get started.";
    else if (!validFile) $("submit-help").textContent = `The video needs at least ${clipDuration().toFixed(2)} seconds at the selected frame rate.`;
    else if (!ready) $("submit-help").textContent = "Complete pod and model setup before reconstructing.";
    else if (state.system?.busy) $("submit-help").textContent = "Your job will wait for the GPU to become available.";
    else $("submit-help").textContent = "Runs on your pod. You can return to check progress.";
  }

  function renderSystem() {
    const system = state.system;
    if (!system) return;
    const gpu = system.gpu;
    const ready = Boolean(system.readiness);
    $("node-title").textContent = gpu?.name || "GPU not available";
    $("node-subtitle").textContent = ready ? (system.busy ? "GPU working · jobs can queue" : "Ready to reconstruct") : "Setup required";
    $("node-dot").className = `status-dot ${ready ? (system.busy ? "running" : "ready") : "queued"}`;
    $("setup-callout").hidden = ready;
    const checks = system.checks || [];
    const missing = checks.filter((check) => !check.ok);
    $("setup-callout-message").textContent = missing.length ? `${missing.map((check) => check.name).join(" · ")}. Open setup to finish preparing your workspace.` : "Prepare the models before starting your first reconstruction.";
    updateSubmit();
    if ($("setup-dialog").open) renderSetup();
  }

  function makeCheck(title, ok, message) {
    const row = node("div", `setup-check ${ok ? "" : "missing"}`);
    const marker = node("span", "check-icon");
    if (ok) marker.append(icon("check")); else marker.textContent = "!";
    const text = node("div");
    text.append(node("strong", "", title), node("small", "", message));
    row.append(marker, text);
    return row;
  }

  function renderSetup() {
    const system = state.system;
    const checks = $("setup-checks");
    checks.replaceChildren();
    if (!system) checks.append(makeCheck("Connecting", false, "Reading your pod's configuration…"));
    else {
      const gpu = system.gpu;
      checks.append(makeCheck("GPU", Boolean(gpu), gpu ? `${gpu.name}${gpu.memory_total_mb ? ` · ${(gpu.memory_total_mb / 1024).toFixed(0)} GB` : ""}` : "No CUDA GPU detected"));
      if (system.disk) checks.append(makeCheck("Persistent storage", system.disk.free_bytes > 10 * 1024 ** 3, `${bytes(system.disk.free_bytes)} available`));
      for (const check of system.checks || []) {
        if (/^(gpu|disk|storage)$/i.test(check.name)) continue;
        checks.append(makeCheck(check.name.replaceAll("_", " "), check.ok, check.message || (check.ok ? "Ready" : "Setup required")));
      }
      for (const warning of system.warnings || []) checks.append(makeCheck("GPU guidance", false, typeof warning === "string" ? warning : warning.message || "Check your GPU memory budget."));
    }
    const running = state.setup?.state === "running" || state.setupUploading;
    const publicReady = Boolean(system?.models_ready);
    const smplxReady = Boolean(system?.model_smplx_ready);
    $("public-models-status").textContent = publicReady ? "Installed" : "Required";
    $("public-models-status").className = `badge ${publicReady ? "ready" : "queued"}`;
    $("smplx-status").textContent = smplxReady ? "Installed" : "Required";
    $("smplx-status").className = `badge ${smplxReady ? "ready" : "queued"}`;
    $("download-models").disabled = publicReady || running || !state.authenticated;
    $("download-models").lastChild.textContent = publicReady ? "Models installed" : (state.setup?.state === "running" && state.setup.kind === "models" ? "Downloading…" : "Download models");
    $("smplx-file").disabled = running || smplxReady;
    $("smplx-upload-label").textContent = state.setupUploading ? "Uploading model…" : (smplxReady ? "Model installed" : "Upload model file");
    document.querySelector(".file-upload-button").classList.toggle("busy", running || smplxReady);
    $("setup-summary").textContent = system?.readiness ? "Your workspace is ready." : "Finish the required setup to start a job.";
    const task = state.setup;
    $("setup-task").hidden = !task || task.state === "idle";
    if (task && task.state !== "idle") {
      $("setup-task-state").textContent = task.state === "completed" ? "Complete" : task.state === "failed" ? "Failed" : "Working";
      $("setup-task-state").className = `badge ${task.state}`;
      $("setup-task-message").textContent = task.error || task.message || "Preparing models…";
      const log = $("setup-task-log");
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
      log.textContent = String(task.log || "").slice(-90000);
      log.hidden = !log.textContent;
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
  }

  function renderHistory() {
    const list = $("job-list");
    const activeElement = document.activeElement;
    const focusedId = activeElement?.closest(".job-entry")?.dataset.jobId;
    list.replaceChildren();
    $("job-count").textContent = String(state.jobs.length);
    $("history-empty").hidden = state.jobs.length > 0;
    for (const job of state.jobs) {
      const button = node("button", `job-entry ${job.id === state.selectedId ? "selected" : ""}`);
      button.type = "button";
      button.dataset.jobId = job.id;
      if (job.id === state.selectedId) button.setAttribute("aria-current", "page");
      const top = node("span", "job-entry-top");
      top.append(icon(job.status === "completed" ? "orbit" : "video"), node("strong", "", job.title || job.input?.filename || "Untitled reconstruction"));
      const meta = node("span", "job-entry-meta");
      meta.append(node("span", `status-dot ${job.status}`), node("span", "", statusNames[job.status] || job.status), node("span", "", "·"), node("span", "", shortDate(job.created_at)));
      button.append(top, meta);
      button.addEventListener("click", () => selectJob(job.id));
      list.append(button);
      if (focusedId === job.id) button.focus({ preventScroll: true });
    }
  }

  function setTab(tab) {
    if (tab === "views" && $("tab-views").disabled) return;
    state.selectedTab = tab;
    for (const name of ["result", "source", "views"]) {
      $("tab-" + name).classList.toggle("active", tab === name);
      $("tab-" + name).setAttribute("aria-selected", String(tab === name));
      $("tab-" + name).tabIndex = tab === name ? 0 : -1;
      $("pane-" + name).hidden = tab !== name;
    }
    if (tab !== "source") $("job-source-preview").pause();
    if (tab !== "views") $("generated-view-preview").pause();
  }

  function jobProgress(job) {
    if (job.status === "completed") return 100;
    const number = Number(job.progress);
    return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : 0;
  }

  function renderStages(job) {
    const current = stages.indexOf(job.stage);
    const complete = job.status === "completed";
    for (const [index, stage] of stages.entries()) {
      const item = document.querySelector(`#stage-list [data-stage="${stage}"]`);
      const done = complete || index < current;
      item.classList.toggle("done", done);
      item.classList.toggle("current", !complete && stage === job.stage && job.status === "running");
      const indicator = item.querySelector(".stage-indicator");
      indicator.replaceChildren();
      if (done) indicator.append(icon("check")); else indicator.textContent = String(index + 1);
    }
  }

  function renderArtifacts(job) {
    const artifacts = Array.isArray(job.artifacts) ? job.artifacts.filter((artifact) => safeURL(artifact.url)) : [];
    const model = artifacts.find((artifact) => (artifact.kind || artifact.type) === "model" && /\.(ftgs\.ply|tsog)(?:$|\?)/i.test(artifact.name || artifact.path || artifact.url)) || artifacts.find((artifact) => /\.ftgs\.ply(?:$|\?)/i.test(artifact.url));
    const modelURL = safeURL(model?.url);
    if (modelURL) {
      const fps = Number(job.options?.target_fps) || 24;
      const query = new URLSearchParams({ src: new URL(modelURL, location.origin).href, fps: String(fps), frames: "121", autoplay: "0", points: "all" });
      if (job.playback?.view) query.set("view", JSON.stringify(job.playback.view));
      const viewerURL = `/player/index.html?${query}`;
      const key = `${job.id}:${modelURL}`;
      if (state.viewerKey !== key) {
        $("result-viewer").src = viewerURL;
        state.viewerKey = key;
      }
      $("result-viewer").hidden = false;
      $("result-placeholder").hidden = true;
      $("open-player").hidden = false;
      $("open-player").href = viewerURL;
      $("viewer-caption").textContent = "Drag to orbit · scroll to zoom · use the timeline to play.";
    } else {
      if (state.viewerKey) $("result-viewer").removeAttribute("src");
      state.viewerKey = "";
      $("result-viewer").hidden = true;
      $("result-placeholder").hidden = false;
      $("open-player").hidden = true;
      $("viewer-caption").textContent = "Orbit, zoom and scrub through time when your scene is ready.";
    }
    const videos = artifacts.filter((artifact) => (artifact.kind || artifact.type) === "video" && /(?:generated|videos\/|view|camera)/i.test(artifact.path || artifact.name || ""));
    $("tab-views").disabled = videos.length === 0;
    if (!videos.length && state.selectedTab === "views") setTab("result");
    const viewsKey = `${job.id}:${videos.map((v) => v.url).join("|")}`;
    if (state.viewsKey !== viewsKey) {
      state.viewsKey = viewsKey;
      const select = $("generated-view-select");
      select.replaceChildren();
      for (const [index, artifact] of videos.entries()) {
        const option = node("option", "", artifact.name || `View ${String(index + 1).padStart(2, "0")}`);
        option.value = safeURL(artifact.url);
        select.append(option);
      }
      if (videos.length) $("generated-view-preview").src = safeURL(videos[0].url);
      else $("generated-view-preview").removeAttribute("src");
    }
    const downloadList = $("downloads-list");
    downloadList.replaceChildren();
    const downloads = artifacts.filter((artifact) => (artifact.kind || artifact.type) !== "image");
    $("downloads-panel").hidden = downloads.length === 0;
    const ordered = [...downloads].sort((a, b) => ((a.kind || a.type) === "model" ? -1 : 0) - ((b.kind || b.type) === "model" ? -1 : 0));
    for (const artifact of ordered) {
      const link = node("a", "download-item");
      link.href = safeURL(artifact.url);
      link.setAttribute("download", artifact.name || "");
      const copy = node("span", "download-copy");
      const kind = artifact.kind || artifact.type || "file";
      const label = kind === "model" ? "Animated 4DGS model" : kind === "checkpoint" ? "Training checkpoint" : artifact.name || "Download file";
      copy.append(node("strong", "", label), node("small", "", `${artifact.name || kind}${artifact.size !== undefined ? ` · ${bytes(artifact.size)}` : ""}`));
      link.append(icon(kind === "model" ? "orbit" : kind === "video" ? "video" : "file"), copy, icon("download", "end-icon"));
      downloadList.append(link);
    }
  }

  function renderJob() {
    const job = state.selected;
    if (!job) return;
    const title = job.title || job.input?.filename || "Untitled reconstruction";
    const options = job.options || {};
    const recipe = job.recipe;
    const defaults = legacyPresets[options.preset || "quality"];
    const views = recipe?.views_per_layer && recipe.layer_pitches?.length
      ? recipe.views_per_layer * recipe.layer_pitches.length : (options.views ?? defaults?.views);
    const steps = options.steps ?? recipe?.steps ?? defaults?.steps;
    const fps = Number(options.target_fps) || 24;
    const viewLabel = views ? `${views} views` : "";
    const stepsLabel = steps ? `${steps.toLocaleString()} steps` : "";
    $("job-title-heading").textContent = title;
    $("breadcrumb-name").textContent = title;
    $("job-status").textContent = statusNames[job.status] || job.status;
    $("job-status").className = `badge ${job.status}`;
    $("job-description").textContent = [viewLabel, stepsLabel, "121 frames", `${fps} FPS`].filter(Boolean).join(" · ");
    $("cancel-job").hidden = !["queued", "running"].includes(job.status);
    $("retry-job").hidden = !["failed", "cancelled", "interrupted"].includes(job.status);
    $("retry-job").querySelector("span").textContent = job.status === "failed" ? "Retry reconstruction" : "Resume reconstruction";
    $("job-error").hidden = !job.error;
    $("job-error").textContent = job.error || "";
    const progress = jobProgress(job);
    $("overall-percent").textContent = `${Math.round(progress)}%`;
    $("overall-progress").style.width = `${progress}%`;
    $("overall-progress").classList.toggle("indeterminate", job.status === "running" && !progress);
    $("job-message").textContent = job.message || (job.status === "queued" ? "Waiting for the GPU. Your source is saved." : stageNames[job.stage] || "");
    $("job-created").textContent = dateLabel(job.created_at);
    $("job-views").textContent = viewLabel || "—";
    const start = Number(options.start_time) || 0;
    $("job-segment").textContent = `${seconds(start)} → ${seconds(start + 121 / fps)}`;
    $("viewer-timing").textContent = `121 FRAMES · ${fps} FPS`;
    renderStages(job);
    const source = safeURL(job.input?.url);
    if (source && $("job-source-preview").getAttribute("src") !== source) $("job-source-preview").src = source;
    if (!source) $("job-source-preview").removeAttribute("src");
    const active = job.status === "running";
    $("placeholder-state").textContent = active ? (stageNames[job.stage] || "RECONSTRUCTING") : "RECONSTRUCTION SPACE";
    $("placeholder-title").textContent = job.status === "completed" ? "Your scene is ready" : job.status === "failed" ? "This run needs attention" : ["cancelled", "interrupted"].includes(job.status) ? "Your progress is saved" : job.status === "queued" ? "Your performance is in the queue" : "Your performance takes shape here";
    $("placeholder-description").textContent = job.status === "failed" ? "Read the message above or open the run log, then retry when the issue is resolved." : ["cancelled", "interrupted"].includes(job.status) ? "Resume the job to continue from the available saved stages." : job.status === "queued" ? "Reconstruction starts when the current GPU job finishes." : "Once reconstruction is complete, explore the animated scene from any angle.";
    renderArtifacts(job);
  }

  function showComposer() {
    state.selectedId = null;
    state.selected = null;
    state.initialSelection = false;
    localStorage.removeItem("person4d.selectedJob");
    $("composer").hidden = false;
    $("job-workspace").hidden = true;
    $("breadcrumb-name").textContent = "New reconstruction";
    $("job-source-preview").pause();
    $("generated-view-preview").pause();
    $("result-viewer").removeAttribute("src");
    state.viewerKey = "";
    renderHistory();
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  async function selectJob(id) {
    if (state.selectedId !== id) {
      state.logOffset = 0;
      state.logJob = null;
      state.logText = "";
      state.viewsKey = "";
      $("job-log").textContent = "";
      $("logs-details").open = false;
    }
    state.initialSelection = false;
    state.selectedId = id;
    localStorage.setItem("person4d.selectedJob", id);
    state.selected = state.jobs.find((job) => job.id === id) || null;
    $("composer").hidden = true;
    $("job-workspace").hidden = false;
    setTab("result");
    renderHistory();
    if (state.selected) renderJob();
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (state.selectedId !== id) return;
      state.selected = job;
      renderJob();
    } catch (error) { if (error.status !== 401) toast(error.message, true); }
  }

  async function pollLogs() {
    const id = state.selectedId;
    if (!id || !$("logs-details").open) return;
    if (state.logJob !== id) { state.logJob = id; state.logOffset = 0; state.logText = ""; }
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(id)}/logs?offset=${state.logOffset}`);
      if (state.selectedId !== id) return;
      const log = $("job-log");
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 55;
      state.logOffset = result.next_offset ?? state.logOffset;
      state.logText = (state.logText + (result.text || "")).slice(-200000);
      log.textContent = state.logText || "No process output yet.";
      if (atBottom) log.scrollTop = log.scrollHeight;
      $("log-state").textContent = state.selected?.status === "running" ? "Live process output" : "Saved process output";
    } catch (error) { if (error.status !== 401) $("log-state").textContent = "Log temporarily unavailable"; }
  }

  async function refresh() {
    if (!state.authenticated || state.polling) return;
    state.polling = true;
    try {
      const results = await Promise.allSettled([api("/api/system"), api("/api/jobs"), api("/api/setup/status")]);
      if (results[0].status === "fulfilled") { state.system = results[0].value; renderSystem(); }
      if (results[1].status === "fulfilled") {
        const jobsResult = results[1].value;
        state.jobs = Array.isArray(jobsResult) ? jobsResult : jobsResult.jobs || [];
        renderHistory();
        if (state.initialSelection) {
          state.initialSelection = false;
          const saved = localStorage.getItem("person4d.selectedJob");
          if (saved && state.jobs.some((job) => job.id === saved)) await selectJob(saved);
        }
      }
      if (results[2].status === "fulfilled") {
        const previous = state.setup?.state;
        state.setup = results[2].value;
        if (previous === "running" && state.setup.state === "completed") toast("Model setup completed.");
        if (previous === "running" && state.setup.state === "failed") toast(state.setup.error || "Model setup failed. Open setup to read the log.", true);
        if ($("setup-dialog").open) renderSetup();
      }
      const authFailure = results.some((result) => result.status === "rejected" && result.reason?.status === 401);
      if (authFailure) return;
      setConnected(results.some((result) => result.status === "fulfilled"));
      if (state.selectedId) {
        const id = state.selectedId;
        try {
          const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
          if (id === state.selectedId) { state.selected = job; renderJob(); }
        } catch (error) {
          if (error.status === 404) { showComposer(); toast("This reconstruction is no longer available.", true); }
        }
      }
      await pollLogs();
    } finally {
      state.polling = false;
      clearTimeout(state.pollTimer);
      if (state.authenticated) state.pollTimer = setTimeout(refresh, document.hidden ? 9000 : (state.setup?.state === "running" || state.jobs.some((job) => ["running", "queued"].includes(job.status)) ? 2200 : 6000));
    }
  }

  function clipDuration() { return 121 / (Number($("target-fps").value) || 24); }

  function setStart(value, seek = false) {
    const max = state.metadataUnknown ? 86400 : Math.max(0, state.duration - clipDuration());
    const start = Math.min(max, Math.max(0, Number(value) || 0));
    const rounded = Math.round(start * 1000) / 1000;
    $("start-time").value = String(rounded);
    $("start-range").value = String(rounded);
    $("segment-end-label").textContent = `END ${seconds(start + clipDuration())}`;
    if (seek && state.file) {
      state.previewing = false;
      $("source-preview").pause();
      $("source-preview").currentTime = start;
    }
  }

  function updateSegment() {
    const duration = clipDuration();
    const max = state.metadataUnknown ? 86400 : Math.max(0, state.duration - duration);
    $("start-time").max = max.toFixed(3);
    $("start-range").max = state.metadataUnknown ? "0" : max.toFixed(3);
    $("start-range").disabled = state.metadataUnknown;
    $("segment-duration").textContent = `121 frames · ${duration.toFixed(2)} s`;
    $("segment-help").textContent = state.metadataUnknown ? "Browser preview is unavailable. Enter a start time manually; the server will check the selected segment." : `One continuous ${duration.toFixed(2)}-second segment at ${$("target-fps").value} FPS. The rest of the video is kept as your source.`;
    setStart($("start-time").value);
    const tooShort = state.file && state.duration > 0 && state.duration + 0.01 < duration;
    $("video-error").hidden = !tooShort;
    if (tooShort) $("video-error").textContent = `This video is too short. Choose a video at least ${duration.toFixed(2)} seconds long, or increase the output frame rate.`;
    $("preview-segment").disabled = Boolean(tooShort || state.metadataUnknown);
    updateSubmit();
  }

  function chooseFile(file) {
    if (!file) return;
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name)) { toast("Choose a supported video file.", true); return; }
    state.file = file;
    state.duration = 0;
    state.metadataUnknown = false;
    state.previewing = false;
    if (state.objectURL) URL.revokeObjectURL(state.objectURL);
    state.objectURL = URL.createObjectURL(file);
    $("source-preview").src = state.objectURL;
    $("source-loaded").hidden = false;
    $("drop-zone").hidden = true;
    $("segment-controls").hidden = false;
    $("video-error").hidden = true;
    $("file-name").textContent = file.name;
    $("file-info").textContent = `${bytes(file.size)} · Reading video…`;
    $("start-time").value = "0";
    updateSubmit();
  }

  function upload(path, form, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", path);
      xhr.withCredentials = true;
      xhr.responseType = "json";
      xhr.timeout = 60 * 60 * 1000;
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) onProgress(Math.round((event.loaded / event.total) * 100));
      };
      xhr.onload = () => {
        const data = xhr.response;
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else {
          if (xhr.status === 401) showLogin();
          reject(new APIError(errorText(data, `Upload failed (${xhr.status}).`), xhr.status));
        }
      };
      xhr.onerror = () => reject(new APIError("The upload connection was interrupted. Check your pod and try again.", 0));
      xhr.ontimeout = () => reject(new APIError("Upload timed out. Try a smaller source video or check the pod's connection.", 0));
      xhr.onabort = () => reject(new APIError("Upload cancelled.", 0));
      xhr.send(form);
    });
  }

  async function submitJob(event) {
    event.preventDefault();
    if (state.uploading || !state.file) return;
    if (!state.system?.readiness) { openSetup(); return; }
    const fps = Number($("target-fps").value);
    const start = Number($("start-time").value);
    const stepsSelect = $("training-steps");
    const stepsInput = stepsSelect.value === "custom" ? $("custom-steps") : stepsSelect;
    const steps = Number(stepsInput.value);
    const seed = Number($("seed").value);
    if (!Number.isFinite(start) || start < 0 || start > 86400 || (!state.metadataUnknown && start + 121 / fps > state.duration + 0.01)) { toast("Choose a segment that fits inside the source video.", true); return; }
    if (!Number.isInteger(steps) || steps < 1000 || steps > 100000) { toast("Training steps must be a whole number between 1,000 and 100,000.", true); stepsInput.focus(); return; }
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) { toast("Enter a seed between 0 and 2,147,483,647.", true); $("seed").focus(); return; }
    const form = new FormData();
    form.append("file", state.file, state.file.name);
    form.append("title", $("job-title").value.trim());
    form.append("views", $("view-count").value);
    form.append("start_time", String(start));
    form.append("target_fps", String(fps));
    form.append("seed", String(seed));
    form.append("steps", String(steps));
    state.uploading = true;
    $("upload-progress").hidden = false;
    $("upload-bar").value = 0;
    $("upload-percent").textContent = "0%";
    $("upload-label").textContent = "Uploading video";
    updateSubmit();
    try {
      const result = await upload("/api/jobs", form, (progress) => {
        $("upload-bar").value = progress;
        $("upload-percent").textContent = `${progress}%`;
        if (progress === 100) $("upload-label").textContent = "Saving your source…";
      });
      const job = result.job || result;
      state.jobs = [job, ...state.jobs.filter((entry) => entry.id !== job.id)];
      toast("Source uploaded. Your reconstruction is queued.");
      await selectJob(job.id);
      await refresh();
    } catch (error) { toast(error.message, true); }
    finally { state.uploading = false; $("upload-progress").hidden = true; updateSubmit(); }
  }

  async function jobAction(action, button) {
    if (!state.selectedId) return;
    const id = state.selectedId;
    button.disabled = true;
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(id)}/${action}`, { method: "POST" });
      if (state.selectedId === id && result?.id) { state.selected = result; renderJob(); }
      toast(action === "cancel" ? "Stop requested. Saved work will remain available." : "The reconstruction has been queued to continue.");
      await refresh();
    } catch (error) { toast(error.message, true); }
    finally { button.disabled = false; }
  }

  $("new-job").addEventListener("click", showComposer);
  $("open-setup").addEventListener("click", openSetup);
  $("callout-setup").addEventListener("click", openSetup);
  for (const id of ["close-setup", "done-setup"]) $(id).addEventListener("click", () => $("setup-dialog").close());
  $("setup-dialog").addEventListener("click", (event) => {
    if (event.target === $("setup-dialog")) {
      const rect = $("setup-dialog").getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $("setup-dialog").close();
    }
  });
  $("login-dialog").addEventListener("cancel", (event) => event.preventDefault());
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("login-submit").disabled = true;
    $("login-error").hidden = true;
    try {
      await api("/api/login", { method: "POST", body: { token: $("access-token").value.trim() } });
      $("access-token").value = "";
      state.authenticated = true;
      $("login-dialog").close();
      $("logout").hidden = !state.tokenRequired;
      await refresh();
    } catch (error) { $("login-error").textContent = error.message; $("login-error").hidden = false; }
    finally { $("login-submit").disabled = false; }
  });
  $("logout").addEventListener("click", async () => {
    try { await api("/api/logout", { method: "POST" }); showLogin(); }
    catch (error) { toast(error.message, true); }
  });
  $("dismiss-toast").addEventListener("click", () => { $("toast").hidden = true; });
  $("video-file").addEventListener("change", (event) => chooseFile(event.target.files[0]));
  $("replace-video").addEventListener("click", () => $("video-file").click());
  $("drop-zone").addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("video-file").click(); } });
  for (const type of ["dragenter", "dragover"]) $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); $("drop-zone").classList.add("drag-over"); });
  for (const type of ["dragleave", "drop"]) $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); $("drop-zone").classList.remove("drag-over"); });
  $("drop-zone").addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
  $("source-preview").addEventListener("loadedmetadata", () => {
    const video = $("source-preview");
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      state.metadataUnknown = true;
      $("file-info").textContent = `${bytes(state.file?.size)} · Preview unavailable`;
      updateSegment();
      return;
    }
    state.duration = video.duration;
    state.metadataUnknown = false;
    $("file-info").textContent = `${bytes(state.file?.size)} · ${seconds(video.duration)} · ${video.videoWidth} × ${video.videoHeight}`;
    updateSegment();
  });
  $("source-preview").addEventListener("error", () => {
    state.metadataUnknown = true;
    state.duration = 0;
    $("file-info").textContent = `${bytes(state.file?.size)} · Preview unavailable in this browser`;
    updateSegment();
  });
  $("source-preview").addEventListener("timeupdate", () => {
    if (state.previewing && $("source-preview").currentTime >= Number($("start-time").value) + clipDuration()) {
      $("source-preview").pause();
      state.previewing = false;
    }
  });
  $("start-range").addEventListener("input", (event) => setStart(event.target.value, true));
  $("start-time").addEventListener("change", (event) => setStart(event.target.value, true));
  $("target-fps").addEventListener("change", updateSegment);
  function updateCustomSteps() {
    const custom = $("training-steps").value === "custom";
    $("custom-steps-field").hidden = !custom;
    $("custom-steps").disabled = !custom;
  }
  $("training-steps").addEventListener("change", updateCustomSteps);
  updateCustomSteps();
  $("preview-segment").addEventListener("click", async () => {
    $("source-preview").currentTime = Number($("start-time").value) || 0;
    state.previewing = true;
    try { await $("source-preview").play(); } catch { state.previewing = false; }
  });
  $("reconstruction-form").addEventListener("submit", submitJob);
  for (const tab of ["result", "source", "views"]) {
    $("tab-" + tab).addEventListener("click", () => setTab(tab));
    $("tab-" + tab).addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const tabs = ["result", "source", "views"].filter((name) => !$("tab-" + name).disabled);
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      setTab(tabs[next]);
      $("tab-" + tabs[next]).focus();
    });
  }
  $("generated-view-select").addEventListener("change", () => {
    const url = safeURL($("generated-view-select").value);
    if (url) $("generated-view-preview").src = url;
  });
  $("cancel-job").addEventListener("click", () => jobAction("cancel", $("cancel-job")));
  $("retry-job").addEventListener("click", () => jobAction(state.selected?.status === "failed" ? "retry" : "resume", $("retry-job")));
  $("logs-details").addEventListener("toggle", () => { if ($("logs-details").open) pollLogs(); });
  $("copy-logs").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(state.logText || $("job-log").textContent); toast("Run log copied."); }
    catch { toast("Copy is unavailable in this browser. Select the log text to copy it.", true); }
  });
  $("download-models").addEventListener("click", async () => {
    $("download-models").disabled = true;
    try {
      state.setup = await api("/api/setup/models", { method: "POST" });
      renderSetup();
      toast("Model download started. You can leave this dialog open to follow it.");
      await refresh();
    } catch (error) { toast(error.message, true); renderSetup(); }
  });
  $("smplx-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (!/\.(npz|zip)$/i.test(file.name)) { toast("Choose SMPLX_NEUTRAL.npz or the official model ZIP.", true); return; }
    state.setupUploading = true;
    renderSetup();
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      state.setup = await upload("/api/setup/smplx", form, (progress) => { $("smplx-upload-label").textContent = progress < 100 ? `Uploading model · ${progress}%` : "Installing model…"; });
      toast("Body model uploaded. Checking installation…");
      await refresh();
    } catch (error) { toast(error.message, true); }
    finally { state.setupUploading = false; event.target.value = ""; renderSetup(); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.authenticated) { clearTimeout(state.pollTimer); refresh(); } });
  window.addEventListener("beforeunload", (event) => { if (state.uploading || state.setupUploading) { event.preventDefault(); event.returnValue = ""; } });

  async function boot() {
    try {
      const auth = await api("/api/auth");
      state.tokenRequired = Boolean(auth.token_required);
      state.authenticated = Boolean(auth.authenticated || !auth.token_required);
      $("logout").hidden = !state.tokenRequired;
      if (!state.authenticated) showLogin(); else await refresh();
    } catch (error) {
      setConnected(false);
      $("node-title").textContent = "Waiting for your pod";
      $("node-subtitle").textContent = "Check the server is running";
      toast(error.message, true);
      setTimeout(boot, 6000);
    }
  }
  setTab("result");
  boot();
})();
