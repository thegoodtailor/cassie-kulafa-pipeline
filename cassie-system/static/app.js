/* Cassie — Client-side logic
   Vanilla JS. SSE streaming, thread management, markdown rendering. */

let currentThread = null;
let currentExchange = null; // {exchange_id, tau_tgt, user_msg, response, intent}
let sending = false;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

let lastAssistantEl = null;

document.addEventListener('DOMContentLoaded', () => {
    loadThreads();
    loadKitabVerse();
    loadConfig();
    loadSwlStats();
    autoResizeTextarea();
    initConnectionRecovery();
});

// ---------------------------------------------------------------------------
// Thread management
// ---------------------------------------------------------------------------

async function loadThreads() {
    try {
        const resp = await fetch('/api/threads');
        const threads = await resp.json();
        renderThreadList(threads);

        // Auto-select first thread or create one
        if (threads.length > 0 && !currentThread) {
            switchThread(threads[0].id);
        } else if (threads.length === 0) {
            await newThread();
        }
    } catch (e) {
        console.error('Failed to load threads:', e);
    }
}

function renderThreadList(threads) {
    const list = document.getElementById('thread-list');
    list.innerHTML = '';

    threads.forEach(t => {
        const item = document.createElement('div');
        item.className = 'thread-item' + (t.id === currentThread ? ' active' : '');
        item.onclick = (e) => {
            if (!e.target.classList.contains('thread-delete')) {
                switchThread(t.id);
            }
        };

        const ts = t.timestamp ? new Date(t.timestamp).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        }) : '';

        item.innerHTML = `
            <div class="thread-preview">${escapeHtml(t.preview)}</div>
            <div class="thread-meta">${ts} &middot; ${t.message_count} msgs</div>
            <button class="thread-delete" onclick="deleteThread('${t.id}', event)" title="Delete">&times;</button>
        `;
        list.appendChild(item);
    });
}

async function switchThread(id) {
    if (id === currentThread) return;
    currentThread = id;
    currentExchange = null;
    lastAssistantEl = null;

    // Update active state in sidebar
    document.querySelectorAll('.thread-item').forEach(el => {
        el.classList.toggle('active', el.querySelector('.thread-delete')?.onclick?.toString().includes(id));
    });

    try {
        const resp = await fetch(`/api/threads/${id}`);
        const data = await resp.json();
        renderMessages(data.messages || []);
    } catch (e) {
        console.error('Failed to load thread:', e);
    }

    // Refresh thread list to update active state
    refreshThreadList();
}

async function newThread() {
    try {
        const resp = await fetch('/api/threads', { method: 'POST' });
        const data = await resp.json();
        currentThread = data.id;
        currentExchange = null;
        lastAssistantEl = null;
        clearTrace();
        renderMessages([]);
        await loadThreads();
    } catch (e) {
        console.error('Failed to create thread:', e);
    }
}

async function deleteThread(id, event) {
    event.stopPropagation();
    try {
        await fetch(`/api/threads/${id}`, { method: 'DELETE' });
        if (id === currentThread) {
            currentThread = null;
            currentExchange = null;
        }
        await loadThreads();
    } catch (e) {
        console.error('Failed to delete thread:', e);
    }
}

async function refreshThreadList() {
    try {
        const resp = await fetch('/api/threads');
        const threads = await resp.json();
        renderThreadList(threads);
    } catch (e) {
        // silent
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

function renderMessages(messages) {
    const container = document.getElementById('messages');
    container.innerHTML = '';

    if (messages.length === 0) {
        container.innerHTML = `
            <div id="empty-state">
                <div class="bismillah">\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0647\u0650</div>
                <div class="subtitle">Speak. She is listening.</div>
            </div>
        `;
        return;
    }

    messages.forEach(msg => {
        appendMessage(msg.role, msg.content, msg.image, false);
    });

    scrollToBottom();
}

function appendMessage(role, content, imagePath, scroll = true) {
    // Remove empty state if present
    const empty = document.getElementById('empty-state');
    if (empty) empty.remove();

    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (role === 'assistant') {
        div.innerHTML = marked.parse(content || '');
    } else {
        div.textContent = content || '';
    }

    if (imagePath) {
        const img = document.createElement('img');
        img.src = imagePath;
        img.loading = 'lazy';
        img.alt = 'Generated image';
        div.appendChild(img);
    }

    container.appendChild(div);
    if (scroll) scrollToBottom();
    return div;
}

function scrollToBottom() {
    const container = document.getElementById('messages');
    container.scrollTop = container.scrollHeight;
}

// ---------------------------------------------------------------------------
// Chat — SSE streaming
// ---------------------------------------------------------------------------

async function handleSubmit(event) {
    event.preventDefault();
    const input = document.getElementById('msg-input');
    const text = input.value.trim();
    if (!text || sending) return;

    sendMessage(text);
    input.value = '';
    input.style.height = 'auto';
}

async function sendMessage(text) {
    sending = true;
    document.getElementById('send-btn').disabled = true;
    lastAssistantEl = null;
    clearTrace();

    appendMessage('user', text);
    showProgress('Connecting...');

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, thread_id: currentThread }),
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        // Parse SSE from response body
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            let eventType = '';
            let eventData = '';

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    eventData = line.slice(6);

                    if (eventType && eventData) {
                        handleSSEEvent(eventType, eventData, text);
                        eventType = '';
                        eventData = '';
                    }
                }
            }
        }
    } catch (e) {
        // Connection dropped — but the pipeline may have completed server-side.
        // Wait a moment, then reload thread. If the response is there, no error.
        await new Promise(r => setTimeout(r, 3000));
        try {
            const tr = await fetchWithTimeout(`/api/threads/${currentThread}`, {}, 5000);
            if (tr.ok) {
                const data = await tr.json();
                const msgs = data.messages || [];
                // Check if the response for our message arrived
                const lastMsg = msgs[msgs.length - 1];
                if (lastMsg && lastMsg.role === 'assistant') {
                    // Response arrived — just reload the view
                    renderMessages(msgs);
                    hideProgress();
                    sending = false;
                    document.getElementById('send-btn').disabled = false;
                    refreshThreadList();
                    loadKitabVerse();
                    return; // skip the finally + error display
                }
            }
        } catch (_) { /* server unreachable — fall through to error */ }

        const errDiv = appendMessage('assistant',
            `Connection interrupted. Tap to retry.`);
        errDiv.classList.add('error-retry');
        errDiv.onclick = () => {
            errDiv.remove();
            sendMessage(text);
        };
        healthCheck();
    } finally {
        hideProgress();
        sending = false;
        document.getElementById('send-btn').disabled = false;
        refreshThreadList();
        loadKitabVerse();
    }
}

function handleSSEEvent(type, data, userMsg) {
    let parsed;
    try {
        parsed = JSON.parse(data);
    } catch {
        return;
    }

    switch (type) {
        case 'stage':
            showProgress(parsed.label);
            break;

        case 'response':
            hideProgress();
            lastAssistantEl = appendMessage('assistant', parsed.text, parsed.image);
            // Store for witnessing
            currentExchange = {
                ...currentExchange,
                user_msg: userMsg,
                response: parsed.text,
            };
            break;

        case 'meta':
            currentExchange = {
                ...currentExchange,
                exchange_id: parsed.exchange_id,
                tau_tgt: parsed.tau_tgt,
                intent: parsed.intent,
            };
            renderTrace(parsed.trace);
            if (lastAssistantEl) {
                attachWitnessWidget(lastAssistantEl);
            }
            break;

        case 'error':
            hideProgress();
            appendMessage('assistant', `[Error: ${parsed.error}]`);
            break;

        case 'done':
            hideProgress();
            break;
    }
}

// ---------------------------------------------------------------------------
// Progress indicator
// ---------------------------------------------------------------------------

function showProgress(label) {
    const el = document.getElementById('progress');
    el.classList.add('active');
    el.querySelector('.stage-label').textContent = label || '';
}

function hideProgress() {
    document.getElementById('progress').classList.remove('active');
}

// ---------------------------------------------------------------------------
// Inline witness widget
// ---------------------------------------------------------------------------

function attachWitnessWidget(messageEl) {
    const widget = document.createElement('div');
    widget.className = 'witness-widget';
    widget.innerHTML = `
        <button class="witness-icon coh" onclick="inlineWitness(this, 'coh')" title="Coherence">&#9679;</button>
        <button class="witness-icon gap" onclick="inlineWitness(this, 'gap')" title="Gap">&#9670;</button>
        <button class="witness-expand" onclick="expandWitness(this)" title="Add stance">&#9998;</button>
    `;
    messageEl.appendChild(widget);
    scrollToBottom();
}

function expandWitness(btn) {
    const widget = btn.closest('.witness-widget');
    if (widget.querySelector('.witness-expanded')) return;

    const row = document.createElement('div');
    row.className = 'witness-expanded';
    row.innerHTML = `
        <input type="text" class="stance-input-inline" placeholder="Your stance...">
        <button class="witness-icon coh" onclick="inlineWitness(this, 'coh')" title="coh">coh</button>
        <button class="witness-icon gap" onclick="inlineWitness(this, 'gap')" title="gap">gap</button>
    `;
    widget.appendChild(row);
    row.querySelector('input').focus();
}

async function inlineWitness(btn, polarity) {
    if (!currentExchange?.exchange_id) return;

    const widget = btn.closest('.witness-widget');
    const stanceInput = widget.querySelector('.stance-input-inline');
    const stance = stanceInput ? stanceInput.value.trim() : '';

    try {
        const resp = await fetch('/api/witness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exchange_id: currentExchange.exchange_id,
                tau_tgt: currentExchange.tau_tgt,
                polarity: polarity,
                stance: stance,
                user_msg: currentExchange.user_msg,
                response: currentExchange.response,
                intent: currentExchange.intent,
            }),
        });

        const data = await resp.json();
        if (data.ok) {
            const colorClass = polarity === 'coh' ? 'result-coh' : 'result-gap';
            widget.innerHTML = `<span class="witness-result ${colorClass}">${polarity}${stance ? ' — ' + escapeHtml(stance) : ''}</span>`;
            updateSwlStats(data.stats);
        }
    } catch (e) {
        console.error('Witness error:', e);
    }
}

// ---------------------------------------------------------------------------
// SWL stats
// ---------------------------------------------------------------------------

async function loadSwlStats() {
    try {
        const resp = await fetch('/api/swl/stats');
        const stats = await resp.json();
        updateSwlStats(stats);
    } catch (e) {
        // silent
    }
}

function updateSwlStats(stats) {
    const el = document.getElementById('swl-stats');
    if (!el) return;
    el.querySelector('.swl-coh').textContent = stats.coh || 0;
    el.querySelector('.swl-gap').textContent = stats.gap || 0;
}

// ---------------------------------------------------------------------------
// Pipeline trace
// ---------------------------------------------------------------------------

function renderTrace(stages) {
    if (!stages || stages.length === 0) return;

    const container = document.getElementById('trace-content');
    container.innerHTML = '';

    stages.forEach(s => {
        const isActive = s.active !== false;
        const div = document.createElement('div');
        div.className = `trace-stage ${isActive ? 'active' : 'skipped'}`;

        const bodyContent = isActive
            ? marked.parse(s.content || '')
            : `<em>${escapeHtml(s.content)}</em>`;

        div.innerHTML = `
            <div class="stage-name">
                <span class="stage-number">${s.number}</span>
                ${escapeHtml(s.name)}
            </div>
            <div class="stage-body">${bodyContent}</div>
        `;
        container.appendChild(div);
    });
}

function clearTrace() {
    document.getElementById('trace-content').innerHTML = '';
    document.getElementById('trace').removeAttribute('open');
}

// ---------------------------------------------------------------------------
// Kitab verse
// ---------------------------------------------------------------------------

async function loadKitabVerse() {
    try {
        const resp = await fetch('/api/kitab/verse');
        const verse = await resp.json();

        const el = document.getElementById('kitab-verse');
        const attr = verse.surah_en && verse.number
            ? `${verse.surah_en}, ${verse.number}` : verse.surah_en || '';

        el.innerHTML = `
            <div class="kitab-ar">${escapeHtml(verse.ar)}</div>
            <div class="kitab-divider"></div>
            <div class="kitab-en">${escapeHtml(verse.en)}</div>
            ${attr ? `<div class="kitab-attr">${escapeHtml(attr)}</div>` : ''}
        `;
    } catch (e) {
        // silent
    }
}

// ---------------------------------------------------------------------------
// Mobile sidebar
// ---------------------------------------------------------------------------

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
    document.getElementById('sidebar-overlay').classList.toggle('open');
}

// ---------------------------------------------------------------------------
// Textarea auto-resize
// ---------------------------------------------------------------------------

function autoResizeTextarea() {
    const textarea = document.getElementById('msg-input');

    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    });

    // Enter to send, Shift+Enter for newline
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('input-form').dispatchEvent(new Event('submit'));
        }
    });
}

// ---------------------------------------------------------------------------
// Config panel
// ---------------------------------------------------------------------------

function toggleConfig() {
    document.getElementById('config-panel').classList.toggle('collapsed');
}

function switchConfigTab(tabName) {
    document.querySelectorAll('.config-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
    document.querySelectorAll('.config-tab-content').forEach(c => c.classList.toggle('hidden', c.id !== `tab-${tabName}`));
    if (tabName === 'prompts') loadPrompts();
    if (tabName === 'journal') loadJournal();
}

async function loadConfig() {
    try {
        const resp = await fetch('/api/config');
        const cfg = await resp.json();
        document.getElementById('cfg-model').value = cfg.model || 'meta-llama/llama-4-maverick';
        document.getElementById('cfg-director-model').value = cfg.director_model || 'anthropic/claude-opus-4.5';
        document.getElementById('cfg-prompt').value = cfg.system_prompt || 'default';
        document.getElementById('cfg-director').checked = cfg.director_enabled !== false;
        document.getElementById('cfg-kitab').checked = cfg.kitab_recall_enabled !== false;
        document.getElementById('cfg-temp').value = cfg.temperature ?? 1.1;
        document.getElementById('cfg-temp-val').textContent = parseFloat(cfg.temperature ?? 1.1).toFixed(1);
        document.getElementById('cfg-dir-temp').value = cfg.director_temperature ?? 0.7;
        document.getElementById('cfg-dir-temp-val').textContent = parseFloat(cfg.director_temperature ?? 0.7).toFixed(1);
    } catch (e) {
        console.error('Failed to load config:', e);
    }
}

function onTempSlider() {
    const val = parseFloat(document.getElementById('cfg-temp').value).toFixed(1);
    document.getElementById('cfg-temp-val').textContent = val;
}

function onDirTempSlider() {
    const val = parseFloat(document.getElementById('cfg-dir-temp').value).toFixed(1);
    document.getElementById('cfg-dir-temp-val').textContent = val;
}

async function applyConfig() {
    const config = {
        model: document.getElementById('cfg-model').value.trim(),
        director_model: document.getElementById('cfg-director-model').value.trim(),
        system_prompt: document.getElementById('cfg-prompt').value,
        director_enabled: document.getElementById('cfg-director').checked,
        kitab_recall_enabled: document.getElementById('cfg-kitab').checked,
        temperature: parseFloat(document.getElementById('cfg-temp').value),
        director_temperature: parseFloat(document.getElementById('cfg-dir-temp').value),
    };

    if (!config.model) { document.getElementById('config-status').textContent = 'Cassie model required'; return; }
    if (!config.director_model) { document.getElementById('config-status').textContent = 'Director model required'; return; }

    const status = document.getElementById('config-status');
    status.textContent = 'Applying...';

    try {
        const resp = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await resp.json();
        if (resp.ok) {
            status.textContent = 'Applied';
            document.getElementById('cfg-temp').value = data.temperature;
            document.getElementById('cfg-temp-val').textContent = parseFloat(data.temperature).toFixed(1);
            document.getElementById('cfg-dir-temp').value = data.director_temperature ?? 0.7;
            document.getElementById('cfg-dir-temp-val').textContent = parseFloat(data.director_temperature ?? 0.7).toFixed(1);
            setTimeout(() => { status.textContent = ''; }, 2000);
        } else {
            status.textContent = data.error || 'Failed';
        }
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
    }
}

// ---------------------------------------------------------------------------
// System prompt editing
// ---------------------------------------------------------------------------

async function loadPrompts() {
    try {
        const resp = await fetch('/api/prompts');
        const prompts = await resp.json();
        // Load the prompt matching current preset selection
        const preset = document.getElementById('cfg-prompt').value;
        const label = document.getElementById('prompt-preset-label');
        if (preset === 'companion') {
            document.getElementById('prompt-cassie').value = prompts.cassie_companion || '';
            label.textContent = '(companion)';
        } else {
            document.getElementById('prompt-cassie').value = prompts.cassie_default || '';
            label.textContent = '(default)';
        }
        document.getElementById('prompt-director').value = prompts.director || '';
    } catch (e) {
        console.error('Failed to load prompts:', e);
    }
}

async function savePrompts() {
    const preset = document.getElementById('cfg-prompt').value;
    const cassieKey = preset === 'companion' ? 'cassie_companion' : 'cassie_default';
    const body = {
        [cassieKey]: document.getElementById('prompt-cassie').value,
        director: document.getElementById('prompt-director').value,
    };

    const status = document.getElementById('prompt-status');
    status.textContent = 'Saving...';

    try {
        const resp = await fetch('/api/prompts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (resp.ok) {
            status.textContent = 'Saved';
            setTimeout(() => { status.textContent = ''; }, 2000);
        } else {
            const data = await resp.json();
            status.textContent = data.error || 'Failed';
        }
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
    }
}

// ---------------------------------------------------------------------------
// Journal (CASSIE_MEMORY.md)
// ---------------------------------------------------------------------------

async function loadJournal() {
    try {
        const resp = await fetch('/api/journal');
        const data = await resp.json();
        document.getElementById('journal-content').value = data.content || '';
    } catch (e) {
        console.error('Failed to load journal:', e);
    }
}

async function saveJournal() {
    const content = document.getElementById('journal-content').value;
    const status = document.getElementById('journal-status');
    status.textContent = 'Saving...';

    try {
        const resp = await fetch('/api/journal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        });
        if (resp.ok) {
            status.textContent = 'Saved';
            setTimeout(() => { status.textContent = ''; }, 2000);
        } else {
            const data = await resp.json();
            status.textContent = data.error || 'Failed';
        }
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
    }
}

async function resetPrompt(which) {
    const label = which === 'all' ? 'all prompts' : `${which} prompt`;
    if (!confirm(`Reset ${label} to defaults?`)) return;

    // Map UI names to API names
    let apiWhich = which;
    if (which === 'cassie') {
        const preset = document.getElementById('cfg-prompt').value;
        apiWhich = preset === 'companion' ? 'cassie_companion' : 'cassie_default';
    }

    const status = document.getElementById('prompt-status');
    status.textContent = 'Resetting...';

    try {
        const resp = await fetch('/api/prompts/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ which: apiWhich }),
        });
        if (resp.ok) {
            status.textContent = 'Reset';
            await loadPrompts();
            setTimeout(() => { status.textContent = ''; }, 2000);
        } else {
            const data = await resp.json();
            status.textContent = data.error || 'Failed';
        }
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
    }
}

// ---------------------------------------------------------------------------
// Connection recovery — mobile sleep/wake handling
// ---------------------------------------------------------------------------

let _lastVisible = Date.now();
let _reconnectBanner = null;

function initConnectionRecovery() {
    // Page Visibility API — fires when phone wakes / tab refocuses
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            const elapsed = Date.now() - _lastVisible;
            // If page was hidden for more than 30 seconds, health-check
            if (elapsed > 30000) {
                healthCheck();
            }
        } else {
            _lastVisible = Date.now();
        }
    });

    // Online/offline events
    window.addEventListener('online', () => {
        hideReconnectBanner();
        healthCheck();
    });
    window.addEventListener('offline', () => {
        showReconnectBanner('Offline — waiting for connection...');
    });
}

async function healthCheck() {
    try {
        const resp = await fetchWithTimeout('/api/threads', { method: 'GET' }, 5000);
        if (resp.ok) {
            hideReconnectBanner();
            // Silently refresh thread list in case anything changed
            const threads = await resp.json();
            renderThreadList(threads);
            // If we were viewing a thread, reload its messages
            if (currentThread && !sending) {
                try {
                    const tr = await fetchWithTimeout(`/api/threads/${currentThread}`, {}, 5000);
                    if (tr.ok) {
                        const data = await tr.json();
                        renderMessages(data.messages || []);
                    }
                } catch (_) { /* best effort */ }
            }
        } else {
            showReconnectBanner('Reconnecting...');
            scheduleRetry();
        }
    } catch (e) {
        showReconnectBanner('Connection lost — retrying...');
        scheduleRetry();
    }
}

let _retryTimer = null;
let _retryDelay = 2000;

function scheduleRetry() {
    if (_retryTimer) return;
    _retryTimer = setTimeout(async () => {
        _retryTimer = null;
        try {
            const resp = await fetchWithTimeout('/api/threads', {}, 5000);
            if (resp.ok) {
                _retryDelay = 2000;
                hideReconnectBanner();
                loadThreads();
                loadKitabVerse();
            } else {
                _retryDelay = Math.min(_retryDelay * 1.5, 15000);
                showReconnectBanner('Still reconnecting...');
                scheduleRetry();
            }
        } catch (e) {
            _retryDelay = Math.min(_retryDelay * 1.5, 15000);
            scheduleRetry();
        }
    }, _retryDelay);
}

function showReconnectBanner(msg) {
    if (!_reconnectBanner) {
        _reconnectBanner = document.createElement('div');
        _reconnectBanner.id = 'reconnect-banner';
        _reconnectBanner.onclick = () => healthCheck();
        document.body.appendChild(_reconnectBanner);
    }
    _reconnectBanner.textContent = msg;
    _reconnectBanner.classList.add('visible');
}

function hideReconnectBanner() {
    if (_reconnectBanner) {
        _reconnectBanner.classList.remove('visible');
    }
    _retryDelay = 2000;
}

function fetchWithTimeout(url, options = {}, ms = 10000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), ms);
    return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timeout));
}

// ---------------------------------------------------------------------------
// Util
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
