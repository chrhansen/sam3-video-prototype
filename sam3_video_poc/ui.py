INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SAM3 Video skier track POC</title>
  <style>
    :root {
      --bg: #0d1117;
      --fg: #e6edf3;
      --muted: #8b949e;
      --card: #161b22;
      --line: #30363d;
      --accent: #d2ff4d;
      --accent-dark: #111;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 20px;
      color: var(--fg);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: radial-gradient(circle at 20% -5%, #1a2435 0%, var(--bg) 40%);
    }
    .wrap { max-width: 980px; margin: 0 auto; }
    .card {
      margin: 0 0 16px 0;
      padding: 16px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    h1 { margin: 0 0 16px 0; }
    p { margin: 0; }
    .row {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .stack {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 8px 14px;
      font-weight: 700;
      cursor: pointer;
      background: var(--accent);
      color: var(--accent-dark);
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    input[type=file], input[type=text] {
      color: var(--fg);
    }
    input[type=text] {
      width: min(520px, 100%);
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #0f1520;
    }
    .muted { color: var(--muted); }
    .hidden { display: none !important; }
    .prompt-options {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #0f1520;
    }
    #frame0, #overlay {
      width: 100%;
      max-width: 820px;
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    #frame0.click-mode { cursor: crosshair; }
    #frame0.text-mode { cursor: default; }
    #links a {
      color: var(--accent);
      margin-right: 12px;
    }
    @media (max-width: 640px) {
      body { padding: 14px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SAM3 Video skier track POC</h1>

    <div class="card stack">
      <form id="uploadForm" class="row">
        <input id="videoInput" type="file" name="video" accept="video/*" required />
        <button id="uploadBtn" type="submit">Upload</button>
      </form>
      <p class="muted">Upload short clip. Then choose one prompt mode only: click frame 0, or enter a text phrase.</p>
    </div>

    <div id="promptCard" class="card stack hidden">
      <div class="prompt-options">
        <label class="chip">
          <input type="radio" name="promptMode" value="click" checked />
          <span>Click frame 0</span>
        </label>
        <label class="chip">
          <input type="radio" name="promptMode" value="text" />
          <span>Text prompt</span>
        </label>
      </div>

      <div id="clickHelp" class="muted">Click the skier/person on frame 0 to start tracking.</div>

      <form id="textPromptForm" class="row hidden">
        <input
          id="textPromptInput"
          type="text"
          placeholder='person / skier / skier in orange jacket and black pants'
        />
        <button id="textPromptBtn" type="submit">Track</button>
      </form>

      <p id="textHint" class="muted hidden">Text mode keeps the top-scoring match from SAM3 Video.</p>
      <img id="frame0" alt="frame0" class="hidden click-mode" />
    </div>

    <div class="card stack">
      <p id="status">Idle</p>
      <video id="overlay" controls playsinline class="hidden"></video>
      <div id="links"></div>
    </div>
  </div>

  <script>
    let current = null;
    let pollTimer = null;

    const statusEl = document.getElementById('status');
    const frame0El = document.getElementById('frame0');
    const overlayEl = document.getElementById('overlay');
    const linksEl = document.getElementById('links');
    const promptCardEl = document.getElementById('promptCard');
    const clickHelpEl = document.getElementById('clickHelp');
    const textPromptFormEl = document.getElementById('textPromptForm');
    const textPromptInputEl = document.getElementById('textPromptInput');
    const textHintEl = document.getElementById('textHint');
    const uploadBtn = document.getElementById('uploadBtn');
    const textPromptBtn = document.getElementById('textPromptBtn');

    function selectedMode() {
      return document.querySelector('input[name="promptMode"]:checked').value;
    }

    function setStatus(msg) {
      statusEl.textContent = msg;
    }

    function stopPolling() {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    function resetResults() {
      overlayEl.pause();
      overlayEl.removeAttribute('src');
      overlayEl.load();
      overlayEl.classList.add('hidden');
      linksEl.innerHTML = '';
    }

    function syncPromptModeUI() {
      const mode = selectedMode();
      const clickMode = mode === 'click';
      clickHelpEl.classList.toggle('hidden', !clickMode);
      textPromptFormEl.classList.toggle('hidden', clickMode);
      textHintEl.classList.toggle('hidden', clickMode);
      frame0El.classList.toggle('click-mode', clickMode);
      frame0El.classList.toggle('text-mode', !clickMode);
    }

    async function pollStatus() {
      if (!current) return;
      const res = await fetch(`/status/${current.job_id}`);
      if (!res.ok) return;

      const s = await res.json();
      const pct = Math.round((s.progress || 0) * 100);
      setStatus(`${s.state} ${pct}% ${s.message || ''}`.trim());

      if (s.state === 'done') {
        stopPolling();
        overlayEl.src = s.results.overlay_url;
        overlayEl.classList.remove('hidden');
        linksEl.innerHTML = `<a href="${s.results.overlay_url}" download>overlay.mp4</a> <a href="${s.results.masks_url}" download>masks.json</a>`;
      }

      if (s.state === 'failed') {
        stopPolling();
        setStatus(`failed: ${s.error || 'unknown error'}`);
      }
    }

    async function startPrompt(payload, label) {
      if (!current) return;

      setStatus(`starting ${label}`);
      textPromptBtn.disabled = true;
      const res = await fetch(`/prompt/${current.job_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      textPromptBtn.disabled = false;

      if (!res.ok) {
        const t = await res.text();
        setStatus(`start failed: ${t}`);
        return;
      }

      stopPolling();
      pollTimer = setInterval(pollStatus, 1000);
      pollStatus();
    }

    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const file = document.getElementById('videoInput').files[0];
      if (!file) return;

      resetResults();
      frame0El.classList.add('hidden');
      promptCardEl.classList.add('hidden');
      setStatus('uploading...');
      uploadBtn.disabled = true;

      const body = new FormData();
      body.append('video', file);
      const res = await fetch('/upload', { method: 'POST', body });
      uploadBtn.disabled = false;

      if (!res.ok) {
        const t = await res.text();
        setStatus(`upload failed: ${t}`);
        return;
      }

      current = await res.json();
      frame0El.src = `${current.frame0_url}?t=${Date.now()}`;
      frame0El.classList.remove('hidden');
      promptCardEl.classList.remove('hidden');
      syncPromptModeUI();
      setStatus('uploaded. choose click or text prompt.');
    });

    document.querySelectorAll('input[name="promptMode"]').forEach((el) => {
      el.addEventListener('change', syncPromptModeUI);
    });

    frame0El.addEventListener('click', async (e) => {
      if (!current || selectedMode() !== 'click') return;

      const rect = frame0El.getBoundingClientRect();
      const xUi = e.clientX - rect.left;
      const yUi = e.clientY - rect.top;
      const x = Math.round((xUi / rect.width) * current.width);
      const y = Math.round((yUi / rect.height) * current.height);

      await startPrompt({ mode: 'click', x, y }, `click (${x}, ${y})`);
    });

    textPromptFormEl.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!current || selectedMode() !== 'text') return;

      const text = textPromptInputEl.value.trim();
      if (!text) {
        setStatus('enter a text prompt first');
        return;
      }

      await startPrompt({ mode: 'text', text }, `"${text}"`);
    });
  </script>
</body>
</html>
"""
