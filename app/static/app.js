/**
 * app/static/app.js
 * ==================
 * Nexa Chat — vanilla JavaScript frontend logic.
 *
 * Architecture
 * ------------
 * No framework. Pure JS with the browser Fetch API.
 *
 * Flow:
 *   1. On load: GET /health → update status badge + model info bar
 *   2. On send: POST /api/chat with full message history
 *   3. Render reply as a new message bubble
 *   4. Repeat (history grows with each turn)
 *
 * API endpoints used:
 *   GET  /health         — check if model is loaded
 *   GET  /api/config     — load default sampling params
 *   POST /api/chat       — send messages, get reply
 *
 * No external JS libraries. No calls to external APIs. Local only.
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
/** @type {Array<{role: string, content: string}>} */
let conversationHistory = [];
let isGenerating = false;

/**
 * Session ID — generated once per page load and persisted in sessionStorage.
 * All memories for this tab share this ID.
 */
const SESSION_ID = (() => {
  const key = 'nexa_session_id';
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
    sessionStorage.setItem(key, id);
  }
  return id;
})();

// ── DOM refs ───────────────────────────────────────────────────────────────
const chatWindow      = document.getElementById('chat-window');
const userInput       = document.getElementById('user-input');
const sendBtn         = document.getElementById('send-btn');
const statusBadge     = document.getElementById('status-badge');
const statusText      = document.getElementById('status-text');
const modelInfoText   = document.getElementById('model-info-text');
const settingsToggle  = document.getElementById('settings-toggle');
const settingsPanel   = document.getElementById('settings-panel');
const closeSettings   = document.getElementById('close-settings');
const clearChat       = document.getElementById('clear-chat');
const forgetMemories  = document.getElementById('forget-memories');
const memoryBadge     = document.getElementById('memory-badge');
const memoryCountEl   = document.getElementById('memory-count');

// Sampling controls
const controls = {
  temperature:  document.getElementById('temperature'),
  topP:         document.getElementById('top-p'),
  repPenalty:   document.getElementById('rep-penalty'),
  topK:         document.getElementById('top-k'),
  maxTokens:    document.getElementById('max-tokens'),
  greedy:       document.getElementById('greedy-mode'),
};
const displays = {
  temperature:  document.getElementById('temperature-val'),
  topP:         document.getElementById('top-p-val'),
  repPenalty:   document.getElementById('rep-penalty-val'),
  topK:         document.getElementById('top-k-val'),
  maxTokens:    document.getElementById('max-tokens-val'),
};

// ── Helpers ────────────────────────────────────────────────────────────────

/** Format a Date as "HH:MM" */
function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Scroll the chat window to the bottom. */
function scrollToBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: 'smooth' });
}

/** Enable or disable the send button + input. */
function setGenerating(on) {
  isGenerating = on;
  sendBtn.disabled = on;
  userInput.disabled = on;
  if (!on) userInput.focus();
}

/** Build a message group element and insert it into the chat window. */
function appendMessage(role, text, pending = false) {
  const isUser = role === 'user';
  const group  = document.createElement('div');
  group.className = `message-group message-${isUser ? 'user' : 'nexa'}`;
  group.dataset.role = role;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.setAttribute('aria-hidden', 'true');
  avatar.textContent = isUser ? 'You' : 'N';

  const body = document.createElement('div');
  body.className = 'message-body';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  if (pending) {
    // Typing indicator
    bubble.classList.add('typing-indicator');
    bubble.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>`;
  } else {
    bubble.innerHTML = escapeAndRender(text);
  }

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = `${isUser ? 'You' : 'Nexa'} · ${nowTime()}`;

  body.appendChild(bubble);
  body.appendChild(meta);
  group.appendChild(avatar);
  group.appendChild(body);
  chatWindow.appendChild(group);
  scrollToBottom();

  return { group, bubble, meta };
}

/**
 * Very minimal safe rendering: escape HTML, then restore newlines as <br>.
 * Does NOT allow arbitrary HTML injection.
 */
function escapeAndRender(text) {
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
  // Wrap in <p> paragraphs split by double newlines, single newlines → <br>
  return escaped
    .split(/\n\n+/)
    .map(para => `<p>${para.replace(/\n/g, '<br>')}</p>`)
    .join('');
}

/** Show an inline error message below the input. */
function showError(message) {
  // Remove existing error toasts
  document.querySelectorAll('.error-toast').forEach(el => el.remove());
  const toast = document.createElement('div');
  toast.className = 'error-toast';
  toast.textContent = `Error: ${message}`;
  document.getElementById('input-area').prepend(toast);
  setTimeout(() => toast.remove(), 5000);
}

// ── Sampling config ────────────────────────────────────────────────────────

/** Read the current sampling settings from the controls. */
function getSamplingParams() {
  return {
    temperature:        parseFloat(controls.temperature.value),
    top_k:              parseInt(controls.topK.value, 10),
    top_p:              parseFloat(controls.topP.value),
    repetition_penalty: parseFloat(controls.repPenalty.value),
    max_new_tokens:     parseInt(controls.maxTokens.value, 10),
    greedy:             controls.greedy.checked,
  };
}

/** Bind slider → live display update. */
function bindSlider(input, display, decimals = 1) {
  const update = () => {
    display.textContent = parseFloat(input.value).toFixed(decimals);
  };
  input.addEventListener('input', update);
  update();   // initial render
}

// ── Status / health ────────────────────────────────────────────────────────

/** Poll GET /health and update the UI status badge. */
async function checkHealth() {
  try {
    const resp = await fetch('/health');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (data.model_loaded) {
      setStatus('ok', 'Model ready');
      const params = `${data.n_params?.toLocaleString() ?? '?'} params · vocab ${data.vocab_size ?? '?'} · v${data.version}`;
      modelInfoText.textContent = `Nexa · ${params} · all processing is local`;
    } else {
      setStatus('error', 'Model not loaded');
      modelInfoText.textContent = 'No model found. Run  python scripts/train.py --demo  then restart the server.';
    }
    updateMemoryBadge(data.memory_count || 0);
  } catch (err) {
    setStatus('error', 'Server offline');
    modelInfoText.textContent = 'Cannot reach the Nexa server. Make sure it is running on localhost:8000.';
  }
}

function setStatus(type, text) {
  statusBadge.className = `status-badge status-${type}`;
  statusText.textContent = text;
}

/** Update the memory badge in the header. */
function updateMemoryBadge(count) {
  if (count > 0) {
    memoryCountEl.textContent = count;
    memoryBadge.style.display = 'flex';
  } else {
    memoryBadge.style.display = 'none';
  }
}

/** Load default sampling params from /api/config and apply to controls. */
async function loadDefaults() {
  try {
    const resp = await fetch('/api/config');
    if (!resp.ok) return;
    const cfg = await resp.json();
    controls.temperature.value = cfg.temperature;
    controls.topP.value        = cfg.top_p;
    controls.repPenalty.value  = cfg.repetition_penalty;
    controls.topK.value        = cfg.top_k;
    controls.maxTokens.value   = cfg.max_new_tokens;
    // Refresh displays
    Object.values(displays).forEach((_, i) => {
      const inputs = Object.values(controls).slice(0, 5);
      if (inputs[i]) inputs[i].dispatchEvent(new Event('input'));
    });
  } catch (_) { /* silently ignore — defaults are already set in HTML */ }
}

// ── Send message ───────────────────────────────────────────────────────────

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isGenerating) return;

  // Clear input
  userInput.value = '';
  userInput.style.height = 'auto';
  document.querySelectorAll('.error-toast').forEach(el => el.remove());

  // Show user message
  conversationHistory.push({ role: 'user', content: text });
  appendMessage('user', text);

  // Show typing indicator
  const { group: pendingGroup, bubble: pendingBubble, meta: pendingMeta } =
    appendMessage('nexa', '', true);

  setGenerating(true);

  try {
    const payload = {
      messages:   conversationHistory,
      session_id: SESSION_ID,
      use_memory: true,
      ...getSamplingParams(),
    };

    const resp = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    const reply = data.reply || '(empty response)';

    // Replace typing indicator with real reply
    pendingBubble.classList.remove('typing-indicator');
    pendingBubble.innerHTML = escapeAndRender(reply);
    pendingMeta.textContent = `Nexa · ${nowTime()}`;

    // Update conversation history (use server's returned history)
    conversationHistory = data.messages;

    // Update memory badge
    updateMemoryBadge(data.memory_count || 0);

    // Show memory indicator if memories were used
    if (data.memories_used && data.memories_used.length > 0) {
      const memNote = document.createElement('div');
      memNote.style.cssText = 'font-size:0.68rem;color:var(--text-faint);margin-top:4px;padding:0 4px';
      memNote.textContent = `Memory: ${data.memories_used.length} past turn(s) retrieved`;
      pendingGroup.querySelector('.message-body').appendChild(memNote);
    }

  } catch (err) {
    // Remove typing indicator on error
    pendingGroup.remove();
    conversationHistory.pop();   // remove the user message we optimistically added
    showError(err.message || 'Failed to get a response.');
  } finally {
    setGenerating(false);
    scrollToBottom();
  }
}

// ── Event listeners ────────────────────────────────────────────────────────

// Send on click
sendBtn.addEventListener('click', sendMessage);

// Send on Enter (Shift+Enter = newline)
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea as user types
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
});

// Settings panel toggle
settingsToggle.addEventListener('click', () => {
  const open = settingsPanel.classList.toggle('open');
  settingsPanel.setAttribute('aria-hidden', String(!open));
  settingsToggle.setAttribute('aria-expanded', String(open));
});

closeSettings.addEventListener('click', () => {
  settingsPanel.classList.remove('open');
  settingsPanel.setAttribute('aria-hidden', 'true');
  settingsToggle.setAttribute('aria-expanded', 'false');
});

// Clear chat
clearChat.addEventListener('click', () => {
  conversationHistory = [];
  // Remove all messages except the welcome message
  const msgs = chatWindow.querySelectorAll('.message-group:not(#welcome-msg)');
  msgs.forEach(m => m.remove());
  settingsPanel.classList.remove('open');
});

// Forget memories (DELETE /api/memory for this session)
forgetMemories.addEventListener('click', async () => {
  try {
    const resp = await fetch(`/api/memory?session_id=${encodeURIComponent(SESSION_ID)}`, {
      method: 'DELETE',
    });
    if (resp.ok) {
      const data = await resp.json();
      updateMemoryBadge(0);
      // Show a small confirmation
      const toast = document.createElement('div');
      toast.className = 'error-toast';
      toast.style.background = 'rgba(34,197,94,0.08)';
      toast.style.borderColor = 'rgba(34,197,94,0.3)';
      toast.style.color = 'var(--success)';
      toast.textContent = `Cleared ${data.removed} memory/memories for this session.`;
      document.getElementById('input-area').prepend(toast);
      setTimeout(() => toast.remove(), 3000);
    }
  } catch (_) { /* silently ignore network errors */ }
  settingsPanel.classList.remove('open');
});

// Greedy mode disables temperature controls (they're irrelevant when greedy)
controls.greedy.addEventListener('change', () => {
  const greedy = controls.greedy.checked;
  controls.temperature.disabled = greedy;
  controls.topP.disabled        = greedy;
  controls.topK.disabled        = greedy;
});

// Bind sliders to live value displays
bindSlider(controls.temperature, displays.temperature, 2);
bindSlider(controls.topP,        displays.topP,        2);
bindSlider(controls.repPenalty,  displays.repPenalty,  2);
bindSlider(controls.topK,        displays.topK,        0);
bindSlider(controls.maxTokens,   displays.maxTokens,   0);

// ── Initialise ─────────────────────────────────────────────────────────────

(async function init() {
  await loadDefaults();
  await checkHealth();
  userInput.focus();
})();
