
const API_BASE = window.location.origin;
const VIEW_STORAGE_KEY = 'admin_view';
const limit = 5;
const userLimit = 15;
const sessionLimit = 20;
const sessionDetailLimit = 15;
let currentPage = 1;
let totalPages = 1;
let isReady = false;
let currentView = 'sessions';
const DEFAULT_SEARCH_TYPE = 'login_id';
const DEFAULT_USER_SEARCH_TYPE = 'login_id';
const DEFAULT_SESSION_SEARCH_TYPE = 'session_id';
let userCurrentPage = 1;
let userTotalPages = 1;
let sessionCurrentPage = 1;
let sessionTotalPages = 1;
let selectedSessionId = null;
let sessionDetailCurrentPage = 1;
let sessionDetailTotalPages = 1;
let sessionDetailKeyword = '';
let logEventSource = null;
let logIsStreaming = false;
let logPaused = false;
let logBuffer = [];
let selectedLogDate = null;
let selectedLogFile = null;
let isCurrentLogSelected = false;
const logFileCache = new Map();
const logDateNodes = new Map();

const userTableBody = document.getElementById('userTableBody');
const userCount = document.getElementById('userCount');
const userPageInfo = document.getElementById('userPageInfo');
const userPageInput = document.getElementById('userPageInput');
const generationTableBody = document.getElementById('generationTableBody');
const generationCount = document.getElementById('generationCount');
const pageInfo = document.getElementById('pageInfo');
const pageInput = document.getElementById('pageInput');
const loginOverlay = document.getElementById('loginOverlay');
const loginError = document.getElementById('loginError');
const textModal = document.getElementById('text-modal');
const textModalTitle = document.getElementById('text-modal-title');
const textModalContent = document.getElementById('text-modal-content');
const textModalClose = document.getElementById('text-modal-close');
const userModal = document.getElementById('user-modal');
const userModalTitle = document.getElementById('user-modal-title');
const userModalList = document.getElementById('user-modal-list');
const userModalClose = document.getElementById('user-modal-close');
const sessionTableBody = document.getElementById('sessionTableBody');
const sessionCount = document.getElementById('sessionCount');
const sessionPageInfo = document.getElementById('sessionPageInfo');
const sessionPageInput = document.getElementById('sessionPageInput');
const sessionDetailSubtitle = document.getElementById('sessionDetailSubtitle');
const sessionDetailEmpty = document.getElementById('sessionDetailEmpty');
const sessionDetailBody = document.getElementById('sessionDetailBody');
const sessionDetailTableBody = document.getElementById('sessionDetailTableBody');
const sessionMetaInfo = document.getElementById('sessionMetaInfo');
const sessionDetailPageInfo = document.getElementById('sessionDetailPageInfo');
const sessionDetailPageInput = document.getElementById('sessionDetailPageInput');
const logStatus = document.getElementById('logStatus');
const logSelectedPath = document.getElementById('logSelectedPath');
const logTreeList = document.getElementById('logTreeList');
const logTreeEmpty = document.getElementById('logTreeEmpty');
const logKeywordInput = document.getElementById('logKeywordInput');
const logLevelSelect = document.getElementById('logLevelSelect');
const logMaskToggle = document.getElementById('logMaskToggle');
const logFullViewBtn = document.getElementById('logFullViewBtn');
const logStreamBtn = document.getElementById('logStreamBtn');
const logPauseBtn = document.getElementById('logPauseBtn');
const logFileActions = document.getElementById('logFileActions');
const logStreamActions = document.getElementById('logStreamActions');
const logAutoScrollToggle = document.getElementById('logAutoScrollToggle');
const logOutput = document.getElementById('logOutput');
const logClearBtn = document.getElementById('logClearBtn');
const logRefreshBtn = document.getElementById('logRefreshBtn');
const logCurrentBtn = document.getElementById('logCurrentBtn');

const TEXT_PREVIEW_LIMIT = 500;
const IMAGE_SIZE = {
    thumb: 'thumb',
    full: 'full',
};

function buildImageUrl(fileHash, size = IMAGE_SIZE.full) {
    if (!fileHash) return null;
    return `${API_BASE}/images/${encodeURIComponent(fileHash)}?size=${size}`;
}

function escapeHtml(str) {
    return String(str ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function escapeRegExp(value) {
    return String(value ?? '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function safeJson(res) {
    try {
        return await res.json();
    } catch {
        return {};
    }
}

function showLogin(message) {
    loginOverlay.classList.remove('hidden');
    loginError.textContent = message || "관리자 계정으로 로그인하세요.";
}

function hideLogin() {
    loginOverlay.classList.add('hidden');
    loginError.textContent = "";
}

const viewSections = {
    sessions: document.getElementById('view-sessions'),
    'session-detail': document.getElementById('view-session-detail'),
    logs: document.getElementById('view-logs'),
    users: document.getElementById('view-users'),
    generations: document.getElementById('view-generations'),
};

function setActiveView(view) {
    currentView = view;
    try {
        sessionStorage.setItem(VIEW_STORAGE_KEY, view);
    } catch {
        // ignore storage errors
    }
    document.documentElement.dataset.view = view;
    document.querySelectorAll('.nav button').forEach((btn) => {
        const btnView = btn.dataset.view;
        btn.classList.toggle('active', btnView === view || (view === 'session-detail' && btnView === 'sessions'));
    });
    Object.entries(viewSections).forEach(([key, section]) => {
        if (!section) return;
        section.classList.toggle('hidden', key !== view);
    });
}

function getSavedView() {
    try {
        return sessionStorage.getItem(VIEW_STORAGE_KEY);
    } catch {
        return null;
    }
}

const initialView = getSavedView();
const allowedViews = ['sessions', 'logs', 'users', 'generations'];
if (allowedViews.includes(initialView)) {
    setActiveView(initialView);
} else {
    setActiveView('sessions');
}

async function fetchUsers() {
    const params = new URLSearchParams();
    params.set('limit', String(userLimit));
    params.set('offset', String((userCurrentPage - 1) * userLimit));

    const userSearchType = document.getElementById('userSearchTypeSelect').value;
    if (userSearchType === 'user_id') {
        const userId = document.getElementById('userFilterUserId').value;
        if (userId) params.set('user_id', userId);
    } else if (userSearchType === 'login_id') {
        const loginId = document.getElementById('userFilterLoginId').value;
        if (loginId) params.set('login_id', loginId);
    } else if (userSearchType === 'created_at') {
        const createdDate = document.getElementById('userFilterCreatedDate').value;
        if (createdDate) {
            params.set('start_date', createdDate);
            params.set('end_date', createdDate);
        }
    } else if (userSearchType === 'role') {
        const role = document.getElementById('userFilterRole').value;
        if (role === 'admin') {
            params.set('is_admin', 'true');
        } else if (role === 'user') {
            params.set('is_admin', 'false');
        }
    } else if (userSearchType === 'name') {
        const name = document.getElementById('userFilterName').value;
        if (name) params.set('name', name);
    }

    const res = await fetch(`${API_BASE}/admin/users?${params.toString()}`, {
        credentials: 'include',
    });
    if (!res.ok) {
        const data = await safeJson(res);
        showLogin(data.detail || "관리자 권한이 필요합니다.");
        return;
    }

    const data = await safeJson(res);
    const users = data.users || [];
    userTableBody.innerHTML = '';
    users.forEach((user) => {
        const row = document.createElement('tr');
        row.innerHTML = `
                    <td><input type="checkbox" class="user-check" data-user-id="${user.user_id}"></td>
                    <td>${escapeHtml(user.login_id)}</td>
                    <td>${user.user_id}</td>
                    <td>${escapeHtml(user.name)}</td>
                    <td>${new Date(user.created_at).toLocaleString()}</td>
                    <td>${user.is_admin ? '<span class="badge">admin</span>' : '<span class="badge badge-neutral">user</span>'}</td>
                `;
        userTableBody.appendChild(row);
    });
    const total = data.total || 0;
    userTotalPages = Math.max(1, Math.ceil(total / userLimit));
    if (userCurrentPage > userTotalPages) {
        userCurrentPage = userTotalPages;
        await fetchUsers();
        return;
    }
    userCount.textContent = `총 ${total}명`;
    userPageInfo.textContent = ` / ${userTotalPages}`;
    userPageInput.value = String(userCurrentPage);
    document.getElementById('selectAllUsers').checked = false;
    hideLogin();
    isReady = true;
}

function hasText(value) {
    return String(value ?? '').trim().length > 0;
}

function formatDateTime(value) {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '-';
    return date.toLocaleString();
}

function truncateText(text, limit = TEXT_PREVIEW_LIMIT) {
    const raw = String(text ?? '');
    if (raw.length <= limit) {
        return { preview: raw, truncated: false };
    }
    return { preview: `${raw.slice(0, limit).trimEnd()}...`, truncated: true };
}

function buildTextPreview(text, title, forceLimit = null) {
    if (!hasText(text)) return '-';
    const raw = String(text ?? '');
    const limitToUse = forceLimit !== null ? forceLimit : TEXT_PREVIEW_LIMIT;
    const { preview, truncated } = truncateText(raw, limitToUse);
    const button = truncated
        ? `<button type="button" class="text-more" data-title="${escapeHtml(title)}" data-full-text="${encodeURIComponent(raw)}">더보기</button>`
        : '';
    return `
                <div class="text-cell">
                    <span class="text-preview-row">
                        <span class="text-preview">${escapeHtml(preview)}</span>
                        ${button}
                    </span>
                </div>
            `;
}

function buildLabeledTextPreview(label, text, title) {
    if (!hasText(text)) return '';
    const raw = String(text ?? '');
    const { preview, truncated } = truncateText(raw);
    const button = truncated
        ? `<button type="button" class="text-more" data-title="${escapeHtml(title)}" data-full-text="${encodeURIComponent(raw)}">더보기</button>`
        : '';
    return `
                <div class="text-cell">
                    <span class="text-label">${escapeHtml(label)}</span>
                    <span class="text-preview-row">
                        <span class="text-preview">${escapeHtml(preview)}</span>
                        ${button}
                    </span>
                </div>
            `;
}

function buildImageCell(image) {
    if (!image || !image.file_hash) return '-';
    const thumbUrl = buildImageUrl(image.file_hash, IMAGE_SIZE.thumb);
    const fullUrl = buildImageUrl(image.file_hash, IMAGE_SIZE.full);
    return `<img class="thumb" src="${thumbUrl}" loading="lazy" alt="preview" onclick="openImageModal('${fullUrl}')">`;
}

function buildInputCell(item) {
    const parts = [];
    const rawText = buildLabeledTextPreview('사용자 입력', item.input_text, '사용자 입력');
    if (rawText) parts.push(rawText);
    const refinedText = buildLabeledTextPreview('정제 입력', item.refined_input_text, '정제 입력');
    if (refinedText) parts.push(refinedText);
    const imageHtml = buildImageCell(item.input_image);
    if (imageHtml !== '-') parts.push(imageHtml);
    if (!parts.length) return '-';
    return `<div class="cell-stack">${parts.join('')}</div>`;
}

function isMetaValuePresent(value) {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string' && value.trim() === '') return false;
    return true;
}

function formatMetaValue(value) {
    if (!isMetaValuePresent(value)) return '-';
    return escapeHtml(String(value));
}

function buildOutputCell(item) {
    const parts = [];
    const promptText = buildLabeledTextPreview('프롬프트', item.prompt, '프롬프트');
    if (promptText) parts.push(promptText);
    if (hasText(item.output_text)) {
        parts.push(buildTextPreview(item.output_text, '생성 텍스트'));
    }
    const imageHtml = buildImageCell(item.output_image);
    if (imageHtml !== '-') parts.push(imageHtml);
    if (!parts.length) return '-';
    return `<div class="cell-stack">${parts.join('')}</div>`;
}

function buildMetaCell(item) {
    const metaItems = [
        {
            label: '생성타입',
            value: item.content_type,
            format: (value) =>
                isMetaValuePresent(value)
                    ? `<span class="badge badge-tight">${escapeHtml(String(value))}</span>`
                    : '-',
        },
        { label: 'generation_method', value: item.generation_method },
        { label: 'style', value: item.style },
        { label: 'industry', value: item.industry },
        { label: 'seed', value: item.seed },
        { label: 'strength', value: item.strength },
        { label: 'aspect_ratio', value: item.aspect_ratio },
    ];

    const rows = metaItems
        .map(({ label, value, format }) => `
                    <div class="meta-row">
                        <span class="meta-label">${label}</span>
                        <span class="meta-value">${format ? format(value) : formatMetaValue(value)}</span>
                    </div>
                `)
        .join('');

    return `<div class="meta-list">${rows}</div>`;
}

async function fetchGenerations() {
    const params = new URLSearchParams();
    params.set('page', String(currentPage));
    params.set('limit', String(limit));

    const searchType = document.getElementById('searchTypeSelect').value;
    if (searchType === 'user_id') {
        const userId = document.getElementById('filterUserId').value;
        if (userId) params.set('user_id', userId);
    } else if (searchType === 'login_id') {
        const loginId = document.getElementById('filterLoginId').value;
        if (loginId) params.set('login_id', loginId);
    } else if (searchType === 'session_id') {
        const sessionId = document.getElementById('filterSessionId').value;
        if (sessionId) params.set('session_id', sessionId);
    } else if (searchType === 'content_type') {
        const contentType = document.getElementById('filterContentType').value;
        if (contentType) params.set('content_type', contentType);
    } else if (searchType === 'created_at') {
        const createdDate = document.getElementById('filterCreatedDate').value;
        if (createdDate) {
            params.set('start_date', createdDate);
            params.set('end_date', createdDate);
        }
    }

    const res = await fetch(`${API_BASE}/admin/generations?${params.toString()}`, {
        credentials: 'include',
    });
    if (!res.ok) {
        const data = await safeJson(res);
        showLogin(data.detail || "관리자 권한이 필요합니다.");
        return;
    }

    const data = await safeJson(res);
    const items = data.items || [];
    generationTableBody.innerHTML = '';

    items.forEach((item) => {
        const row = document.createElement('tr');
        const loginLabel = item.login_id ? escapeHtml(item.login_id) : 'guest';
        const userCell = `
                        <a href="#" class="user-link"
                            data-login-id="${encodeURIComponent(item.login_id || '')}"
                            data-name="${encodeURIComponent(item.name || '')}"
                            data-user-id="${encodeURIComponent(item.user_id ?? '')}"
                            data-session-id="${encodeURIComponent(item.session_id || '')}">
                            ${loginLabel}
                        </a>
                    `;
        const inputCell = buildInputCell(item);
        const outputCell = buildOutputCell(item);
        const metaCell = buildMetaCell(item);
        const createdAt = item.created_at
            ? new Date(item.created_at).toLocaleString()
            : '-';
        row.innerHTML = `
                    <td class="nowrap">${userCell}</td>
                    <td>${inputCell}</td>
                    <td>${outputCell}</td>
                    <td>${metaCell}</td>
                    <td>${createdAt}</td>
                `;
        generationTableBody.appendChild(row);
    });

    const total = data.total || 0;
    totalPages = Math.max(1, Math.ceil(total / limit));
    generationCount.textContent = `총 ${total}건`;
    pageInfo.textContent = ` / ${totalPages}`;
    pageInput.value = String(currentPage);
    hideLogin();
    isReady = true;
}

async function fetchSessions() {
    const params = new URLSearchParams();
    params.set('limit', String(sessionLimit));
    params.set('offset', String((sessionCurrentPage - 1) * sessionLimit));

    const sessionSearchType = document.getElementById('sessionSearchTypeSelect').value;
    if (sessionSearchType === 'session_id') {
        const query = document.getElementById('sessionFilterQuery').value;
        if (query) params.set('query', query);
    } else if (sessionSearchType === 'login_id') {
        const loginId = document.getElementById('sessionFilterLoginId').value;
        if (loginId) params.set('login_id', loginId);
    } else if (sessionSearchType === 'user_id') {
        const userId = document.getElementById('sessionFilterUserId').value;
        if (userId) params.set('user_id', userId);
    } else if (sessionSearchType === 'created_at') {
        const createdDate = document.getElementById('sessionFilterCreatedDate').value;
        if (createdDate) {
            params.set('from', createdDate);
            params.set('to', createdDate);
        }
    }

    const res = await fetch(`${API_BASE}/admin/sessions?${params.toString()}`, {
        credentials: 'include',
    });
    if (!res.ok) {
        const data = await safeJson(res);
        showLogin(data.detail || "관리자 권한이 필요합니다.");
        return;
    }

    const data = await safeJson(res);
    const items = data.items || [];
    sessionTableBody.innerHTML = '';

    items.forEach((item) => {
        const row = document.createElement('tr');
        row.classList.add('session-row');
        row.dataset.sessionId = encodeURIComponent(item.session_id || '');
        const loginLabel = item.login_id ? escapeHtml(item.login_id) : 'guest';
        row.innerHTML = `
                    <td><button type="button" class="link-button session-link" data-session-id="${encodeURIComponent(item.session_id || '')}">${escapeHtml(item.session_id)}</button></td>
                    <td>${loginLabel}</td>
                    <td>${item.message_count ?? 0}</td>
                    <td>${item.generation_count ?? 0}</td>
                    <td>${formatDateTime(item.created_at)}</td>
                `;
        sessionTableBody.appendChild(row);
    });

    const total = data.total || 0;
    sessionTotalPages = Math.max(1, Math.ceil(total / sessionLimit));
    if (sessionCurrentPage > sessionTotalPages) {
        sessionCurrentPage = sessionTotalPages;
        await fetchSessions();
        return;
    }
    sessionCount.textContent = `총 ${total}건`;
    sessionPageInfo.textContent = ` / ${sessionTotalPages}`;
    sessionPageInput.value = String(sessionCurrentPage);
    hideLogin();
    isReady = true;
}

function buildSessionMessageRow(message) {
    const roleLabel = message.role === 'assistant' ? 'assistant' : 'user';
    const roleBadge = roleLabel === 'assistant'
        ? '<span class="badge badge-tight">assistant</span>'
        : '<span class="badge badge-tight badge-neutral">user</span>';
    // 500자 미만은 전체 표시, 500자 이상만 더보기
    const contentCell = buildTextPreview(message.content, '메시지', TEXT_PREVIEW_LIMIT);
    const imageHtml = message.image ? buildImageCell(message.image) : '-';
    return `
                <tr class="session-message-row ${roleLabel}">
                    <td class="nowrap">${formatDateTime(message.created_at)}</td>
                    <td>${roleBadge}</td>
                    <td>${contentCell}</td>
                    <td>${imageHtml}</td>
                </tr>
            `;
}

function renderSessionMessagesTable(messages) {
    const html = messages.map(buildSessionMessageRow).join('');
    if (!html) {
        sessionDetailTableBody.innerHTML = '<tr><td colspan="4" class="empty-cell">메시지가 없습니다.</td></tr>';
        sessionDetailEmpty.classList.remove('hidden');
        sessionDetailBody.classList.add('hidden');
        return;
    }
    sessionDetailTableBody.innerHTML = html;
    sessionDetailEmpty.classList.add('hidden');
    sessionDetailBody.classList.remove('hidden');
}

function renderSessionMeta(detail) {
    const rows = [
        { label: 'session_id', value: detail.session_id },
        { label: 'login_id', value: detail.login_id || 'guest' },
        { label: 'user_id', value: detail.user_id ?? 'guest' },
        { label: 'created_at', value: formatDateTime(detail.created_at) },
        { label: 'last_message_at', value: formatDateTime(detail.last_message_at) },
        { label: 'message_count', value: detail.message_count ?? 0 },
    ];
    sessionMetaInfo.innerHTML = rows
        .map(({ label, value }) => `
                    <div class="meta-row">
                        <span class="meta-label">${escapeHtml(label)}</span>
                        <span class="meta-value">${escapeHtml(String(value ?? '-'))}</span>
                    </div>
                `)
        .join('');
}

async function fetchSessionDetail(sessionId, page = 1, keyword = '') {
    const params = new URLSearchParams();
    params.set('message_limit', String(sessionDetailLimit));
    params.set('message_offset', String((page - 1) * sessionDetailLimit));
    if (keyword) params.set('query', keyword);

    const res = await fetch(`${API_BASE}/admin/sessions/${encodeURIComponent(sessionId)}?${params.toString()}`, {
        credentials: 'include',
    });
    if (!res.ok) {
        const data = await safeJson(res);
        showLogin(data.detail || "관리자 권한이 필요합니다.");
        return false;
    }
    const data = await safeJson(res);
    const total = data.message_count || 0;
    sessionDetailTotalPages = Math.max(1, Math.ceil(total / sessionDetailLimit));

    renderSessionMeta(data);
    renderSessionMessagesTable(data.messages || []);

    sessionDetailSubtitle.textContent = `총 ${total}건`;
    sessionDetailPageInfo.textContent = ` / ${sessionDetailTotalPages}`;
    sessionDetailPageInput.value = String(page);

    hideLogin();
    isReady = true;
    return true;
}

async function selectSession(sessionId) {
    selectedSessionId = sessionId;
    sessionDetailCurrentPage = 1;
    sessionDetailKeyword = '';
    const keywordInput = document.getElementById('sessionDetailKeyword');
    if (keywordInput) keywordInput.value = '';

    setActiveView('session-detail');
    await fetchSessionDetail(sessionId, 1, '');
}

function goBackToSessionList() {
    selectedSessionId = null;
    setActiveView('sessions');
    fetchSessions();
}

function formatBytes(value) {
    if (!Number.isFinite(value)) return '-';
    if (value < 1024) return `${value} B`;
    const units = ['KB', 'MB', 'GB'];
    let size = value;
    let index = -1;
    while (size >= 1024 && index < units.length - 1) {
        size /= 1024;
        index += 1;
    }
    return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[index]}`;
}

const ANSI_REGEX = /[\u001b\u009b][[[()#;?]*([0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;

function stripAnsi(str) {
    if (typeof str !== 'string') return str;
    return str.replace(ANSI_REGEX, '');
}

function setLogStatus(message) {
    logStatus.textContent = message || '';
}

function renderLogLines(lines, { append = false } = {}) {
    if (!Array.isArray(lines)) return;
    if (!append) {
        logOutput.innerHTML = '';
    }
    if (!lines.length) return;

    const keyword = logKeywordInput.value.trim();
    const regex = keyword ? new RegExp(escapeRegExp(keyword), 'gi') : null;
    const html = lines.map((line) => {
        const cleanLine = stripAnsi(line);
        const escaped = escapeHtml(cleanLine);
        const highlighted = regex ? escaped.replace(regex, '<mark>$&</mark>') : escaped;
        return `<div class="log-line">${highlighted}</div>`;
    }).join('');

    if (append) {
        logOutput.insertAdjacentHTML('beforeend', html);
    } else {
        logOutput.innerHTML = html;
    }
    if (logAutoScrollToggle.checked) {
        logOutput.scrollTop = logOutput.scrollHeight;
    }
}

function flushLogBuffer() {
    if (!logBuffer.length) return;
    renderLogLines(logBuffer, { append: true });
    logBuffer = [];
}

function handleLogLine(line) {
    if (logPaused) {
        logBuffer.push(line);
        return;
    }
    renderLogLines([line], { append: true });
}

function updateLogSelectedPath() {
    if (!logSelectedPath) return;
    if (isCurrentLogSelected) {
        logSelectedPath.textContent = '선택: 실시간 로그';
    } else if (selectedLogDate && selectedLogFile) {
        logSelectedPath.textContent = `선택: ${selectedLogDate}/${selectedLogFile}`;
    } else {
        logSelectedPath.textContent = '';
    }
}

function updateLogActionButtons() {
    if (isCurrentLogSelected) {
        logFileActions.classList.add('hidden');
        logStreamActions.classList.remove('hidden');
    } else {
        logFileActions.classList.remove('hidden');
        logStreamActions.classList.add('hidden');
    }
}

function clearLogSelection() {
    selectedLogDate = null;
    selectedLogFile = null;
    isCurrentLogSelected = false;
    updateLogSelectedPath();
    updateLogActionButtons();
}

function createLogDateItem(date) {
    const item = document.createElement('div');
    item.className = 'log-date-item';
    item.dataset.date = date;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'log-date-button';
    button.textContent = date;
    const fileList = document.createElement('div');
    fileList.className = 'log-file-list hidden';
    button.addEventListener('click', () => {
        void selectLogDate(date);
    });
    item.append(button, fileList);
    logDateNodes.set(date, { item, fileList });
    return item;
}

async function selectLogDate(date, { preselectFile = null } = {}) {
    if (!date) return;
    isCurrentLogSelected = false;
    selectedLogDate = date;
    updateLogActionButtons();

    logDateNodes.forEach((node, key) => {
        const isActive = key === date;
        node.item.classList.toggle('is-active', isActive);
        node.fileList.classList.toggle('hidden', !isActive);
    });
    logCurrentBtn.classList.remove('is-active');
    await loadLogFilesForDate(date, { preselectFile, autoSelect: true });
}

async function loadLogFilesForDate(date, { preselectFile = null, autoSelect = true } = {}) {
    const node = logDateNodes.get(date);
    if (!node) return;
    let files = logFileCache.get(date);
    if (!files) {
        const res = await fetch(`${API_BASE}/admin/logs/files?date=${encodeURIComponent(date)}`, {
            credentials: 'include',
        });
        if (!res.ok) {
            const data = await safeJson(res);
            showLogin(data.detail || "관리자 권한이 필요합니다.");
            return;
        }
        const data = await safeJson(res);
        files = data.files || [];
        logFileCache.set(date, files);
    }
    renderLogFiles(date, files, { preselectFile, autoSelect });
}

function renderLogFiles(date, files, { preselectFile = null, autoSelect = true } = {}) {
    const node = logDateNodes.get(date);
    if (!node) return;
    const { fileList } = node;
    if (!files.length) {
        fileList.innerHTML = '<div class="empty-state">파일 없음</div>';
        if (selectedLogDate === date) {
            selectedLogFile = null;
            updateLogSelectedPath();
        }
        return;
    }
    fileList.innerHTML = files.map((item) => {
        const label = `${item.name}`;
        const sizeLabel = formatBytes(item.size_bytes);
        return `
                    <button type="button" class="log-file-button" data-date="${encodeURIComponent(date)}" data-file="${encodeURIComponent(item.name)}">
                        <span>${escapeHtml(label)}</span>
                        <span class="log-file-meta">${escapeHtml(sizeLabel)}</span>
                    </button>
                `;
    }).join('');
    fileList.querySelectorAll('.log-file-button').forEach((btn) => {
        btn.addEventListener('click', () => {
            const fileName = decodeURIComponent(btn.dataset.file || '');
            const targetDate = decodeURIComponent(btn.dataset.date || '');
            if (!fileName || !targetDate) return;
            stopLogStream();
            setSelectedLogFile(targetDate, fileName);
        });
    });
    let nextFile = null;
    if (preselectFile && files.some((item) => item.name === preselectFile)) {
        nextFile = preselectFile;
    } else if (autoSelect && files[0]) {
        nextFile = files[0].name;
    }
    if (nextFile) {
        setSelectedLogFile(date, nextFile);
    }
}

function setSelectedLogFile(date, file) {
    const isSameSelection = selectedLogDate === date && selectedLogFile === file;
    if (!isSameSelection && logIsStreaming) {
        stopLogStream('로그 파일이 변경되었습니다.');
    }
    isCurrentLogSelected = false;
    selectedLogDate = date;
    selectedLogFile = file;
    updateLogSelectedPath();
    updateLogActionButtons();
    document.querySelectorAll('.log-file-button').forEach((btn) => {
        const btnDate = decodeURIComponent(btn.dataset.date || '');
        const btnFile = decodeURIComponent(btn.dataset.file || '');
        btn.classList.toggle('is-active', btnDate === date && btnFile === file);
    });
    logCurrentBtn.classList.remove('is-active');
}

function selectCurrentLog() {
    stopLogStream();
    isCurrentLogSelected = true;
    selectedLogDate = null;
    selectedLogFile = null;
    updateLogSelectedPath();
    updateLogActionButtons();

    logDateNodes.forEach((node) => {
        node.item.classList.remove('is-active');
        node.fileList.classList.add('hidden');
    });
    document.querySelectorAll('.log-file-button').forEach((btn) => {
        btn.classList.remove('is-active');
    });
    logCurrentBtn.classList.add('is-active');
    logOutput.innerHTML = '<div class="log-line" style="color: #6b7280;">실시간 로그를 보려면 "시작" 버튼을 누르세요.</div>';
    setLogStatus('');
}

async function fetchLogDates({ preselectDate = null, preselectFile = null } = {}) {
    const res = await fetch(`${API_BASE}/admin/logs/dates`, { credentials: 'include' });
    if (!res.ok) {
        const data = await safeJson(res);
        showLogin(data.detail || "관리자 권한이 필요합니다.");
        return;
    }
    const data = await safeJson(res);
    const dates = data.dates || [];
    hideLogin();
    isReady = true;
    logTreeList.innerHTML = '';
    logDateNodes.clear();
    logFileCache.clear();
    if (!dates.length) {
        logTreeEmpty.classList.remove('hidden');
        clearLogSelection();
        return;
    }
    logTreeEmpty.classList.add('hidden');
    dates.forEach((item) => {
        logTreeList.appendChild(createLogDateItem(item));
    });

    // 기본적으로 현재 로그 선택
    selectCurrentLog();
}

async function fetchLogFull() {
    if (!selectedLogDate || !selectedLogFile) {
        setLogStatus('로그 파일을 선택하세요.');
        return;
    }
    const params = new URLSearchParams();
    params.set('date', selectedLogDate);
    params.set('file', selectedLogFile);
    const keyword = logKeywordInput.value.trim();
    const level = logLevelSelect.value;
    if (keyword) params.set('query', keyword);
    if (level) params.set('level', level);
    params.set('mask', logMaskToggle.checked ? 'true' : 'false');

    setLogStatus('로딩 중...');
    const res = await fetch(`${API_BASE}/admin/logs/full?${params.toString()}`, {
        credentials: 'include',
    });
    if (!res.ok) {
        const data = await safeJson(res);
        showLogin(data.detail || "로그를 불러올 수 없습니다.");
        return;
    }
    const data = await safeJson(res);
    logBuffer = [];
    renderLogLines(data.lines || [], { append: false });
    setLogStatus(`전체 ${data.total_lines || 0}줄`);
    hideLogin();
    isReady = true;
}

function startLogStream() {
    if (logIsStreaming) return;

    const params = new URLSearchParams();
    const keyword = logKeywordInput.value.trim();
    const level = logLevelSelect.value;
    if (keyword) params.set('query', keyword);
    if (level) params.set('level', level);
    params.set('mask', logMaskToggle.checked ? 'true' : 'false');

    const streamUrl = isCurrentLogSelected
        ? `${API_BASE}/admin/logs/stream/current?${params.toString()}`
        : `${API_BASE}/admin/logs/stream?date=${selectedLogDate}&file=${selectedLogFile}&${params.toString()}`;

    logOutput.innerHTML = '';
    logEventSource = new EventSource(streamUrl);
    logEventSource.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data || '{}');
            if (payload.line) {
                handleLogLine(payload.line);
            }
        } catch (error) {
            console.error(error);
        }
    };
    logEventSource.onerror = () => {
        stopLogStream('스트리밍 연결이 종료되었습니다.');
    };
    logIsStreaming = true;
    logPaused = false;
    logBuffer = [];
    logStreamBtn.textContent = '중지';
    logPauseBtn.disabled = false;
    logPauseBtn.textContent = '일시정지';
    setLogStatus('실시간 스트리밍 중');
    hideLogin();
    isReady = true;
}

function stopLogStream(message) {
    if (logEventSource) {
        logEventSource.close();
        logEventSource = null;
    }
    logIsStreaming = false;
    logPaused = false;
    logBuffer = [];
    logStreamBtn.textContent = '시작';
    logPauseBtn.disabled = true;
    logPauseBtn.textContent = '일시정지';
    if (message) {
        setLogStatus(message);
    }
}

function toggleLogPause() {
    if (!logIsStreaming) return;
    logPaused = !logPaused;
    logPauseBtn.textContent = logPaused ? '재개' : '일시정지';
    if (!logPaused) {
        flushLogBuffer();
    }
}

async function initLogsView({ preselectDate = null, preselectFile = null } = {}) {
    await fetchLogDates({ preselectDate, preselectFile });
}

async function deleteSelectedUsers() {
    const checked = Array.from(document.querySelectorAll('.user-check:checked'));
    const ids = checked.map((el) => Number(el.dataset.userId));
    if (!ids.length) {
        alert('삭제할 사용자를 선택하세요.');
        return;
    }
    if (!confirm(`선택한 ${ids.length}명을 삭제하시겠습니까?`)) return;

    const res = await fetch(`${API_BASE}/admin/users/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ user_ids: ids }),
    });

    const data = await safeJson(res);
    if (!res.ok) {
        alert(data.detail || '삭제에 실패했습니다.');
        return;
    }

    if (data.skipped_ids && data.skipped_ids.length) {
        alert(`삭제 제외: ${data.skipped_ids.join(', ')}`);
    }
    await fetchUsers();
}

async function handleLogin() {
    const loginId = document.getElementById('loginId').value;
    const loginPw = document.getElementById('loginPw').value;
    if (!loginId || !loginPw) {
        loginError.textContent = '아이디와 비밀번호를 입력하세요.';
        return;
    }

    const body = new URLSearchParams();
    body.set('username', loginId);
    body.set('password', loginPw);

    const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        credentials: 'include',
        body,
    });

    if (!res.ok) {
        const data = await safeJson(res);
        loginError.textContent = data.detail || '로그인 실패';
        return;
    }

    hideLogin();
    resetSessionFilters({ shouldFetch: false });
    resetUserFilters({ shouldFetch: false });
    resetFilters({ shouldFetch: false });
    if (currentView === 'sessions') {
        await fetchSessions();
    } else if (currentView === 'logs') {
        await initLogsView();
    } else if (currentView === 'generations') {
        await fetchGenerations();
    } else {
        await fetchUsers();
    }
}

async function handleLogout() {
    try {
        await fetch(`${API_BASE}/auth/logout`, { method: 'POST', credentials: 'include' });
    } catch (e) {
        console.error(e);
    }
    window.location.href = "/";
}

function updateScopedSearchFields(scope, selectId) {
    const searchType = document.getElementById(selectId).value;
    document
        .querySelectorAll(`.search-field[data-search-scope="${scope}"]`)
        .forEach((field) => {
            field.classList.toggle('hidden', field.dataset.searchType !== searchType);
        });
}

function updateSearchFields() {
    updateScopedSearchFields('generation', 'searchTypeSelect');
}

function updateUserSearchFields() {
    updateScopedSearchFields('user', 'userSearchTypeSelect');
}

function updateSessionSearchFields() {
    updateScopedSearchFields('session', 'sessionSearchTypeSelect');
}

function resetFilters({ shouldFetch = true } = {}) {
    document.getElementById('searchTypeSelect').value = DEFAULT_SEARCH_TYPE;
    document.getElementById('filterUserId').value = '';
    document.getElementById('filterLoginId').value = '';
    document.getElementById('filterSessionId').value = '';
    document.getElementById('filterContentType').value = '';
    document.getElementById('filterCreatedDate').value = '';
    currentPage = 1;
    updateSearchFields();
    if (shouldFetch && isReady) {
        fetchGenerations();
    }
}

function resetUserFilters({ shouldFetch = true } = {}) {
    document.getElementById('userSearchTypeSelect').value = DEFAULT_USER_SEARCH_TYPE;
    document.getElementById('userFilterUserId').value = '';
    document.getElementById('userFilterLoginId').value = '';
    document.getElementById('userFilterName').value = '';
    document.getElementById('userFilterCreatedDate').value = '';
    document.getElementById('userFilterRole').value = '';
    userCurrentPage = 1;
    updateUserSearchFields();
    if (shouldFetch && isReady) {
        fetchUsers();
    }
}

function resetSessionFilters({ shouldFetch = true } = {}) {
    document.getElementById('sessionSearchTypeSelect').value = DEFAULT_SESSION_SEARCH_TYPE;
    document.getElementById('sessionFilterQuery').value = '';
    document.getElementById('sessionFilterLoginId').value = '';
    document.getElementById('sessionFilterUserId').value = '';
    document.getElementById('sessionFilterCreatedDate').value = '';
    sessionCurrentPage = 1;
    selectedSessionId = null;
    sessionDetailCurrentPage = 1;
    sessionDetailKeyword = '';
    updateSessionSearchFields();
    if (shouldFetch && isReady) {
        fetchSessions();
    }
}

function openImageModal(url) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-image');
    img.src = url;
    modal.classList.remove('hidden');
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    modal.classList.add('hidden');
}

function openTextModal(title, text) {
    textModalTitle.textContent = title || '텍스트 상세';
    textModalContent.textContent = text || '';
    textModal.classList.remove('hidden');
}

function closeTextModal() {
    textModal.classList.add('hidden');
    textModalTitle.textContent = '텍스트 상세';
    textModalContent.textContent = '';
}

function openUserModal({ loginId, name, userId, sessionId }) {
    const titleSuffix = loginId ? ` (${loginId})` : '';
    userModalTitle.textContent = `사용자 정보${titleSuffix}`;
    const rows = [
        { label: '사용자 아이디', value: loginId || 'guest' },
        { label: 'name', value: name },
        { label: 'user_id', value: userId },
        { label: 'session_id', value: sessionId },
    ];
    userModalList.innerHTML = rows
        .map(({ label, value }) => `
                    <div class="meta-row">
                        <span class="meta-label">${escapeHtml(label)}</span>
                        <span class="meta-value">${formatMetaValue(value)}</span>
                    </div>
                `)
        .join('');
    userModal.classList.remove('hidden');
}

function closeUserModal() {
    userModal.classList.add('hidden');
    userModalTitle.textContent = '사용자 정보';
    userModalList.innerHTML = '';
}

updateSearchFields();
updateUserSearchFields();
updateSessionSearchFields();

document.querySelectorAll('.nav button').forEach((btn) => {
    btn.addEventListener('click', () => {
        const nextView = btn.dataset.view;
        const prevView = currentView;
        setActiveView(nextView);
        if (prevView === 'logs' && nextView !== 'logs') {
            stopLogStream();
        }
        if (nextView === 'sessions') {
            if (prevView !== 'sessions') {
                resetSessionFilters({ shouldFetch: isReady });
            } else if (isReady) {
                fetchSessions();
            }
        } else if (nextView === 'logs') {
            if (prevView !== 'logs' && isReady) {
                initLogsView();
            }
        } else if (nextView === 'generations') {
            if (prevView !== 'generations') {
                resetFilters({ shouldFetch: isReady });
            } else if (isReady) {
                fetchGenerations();
            }
        } else if (nextView === 'users') {
            if (prevView !== 'users') {
                resetUserFilters({ shouldFetch: isReady });
            } else if (isReady) {
                fetchUsers();
            }
        }
    });
});

document.getElementById('searchSessionsBtn').addEventListener('click', () => {
    sessionCurrentPage = 1;
    fetchSessions();
});
document.getElementById('resetSessionsBtn').addEventListener('click', resetSessionFilters);
document.getElementById('sessionSearchTypeSelect').addEventListener('change', updateSessionSearchFields);
document.getElementById('sessionPrevPageBtn').addEventListener('click', () => {
    if (sessionCurrentPage > 1) {
        sessionCurrentPage -= 1;
        fetchSessions();
    }
});
document.getElementById('sessionNextPageBtn').addEventListener('click', () => {
    if (sessionCurrentPage < sessionTotalPages) {
        sessionCurrentPage += 1;
        fetchSessions();
    }
});
sessionPageInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;
    const nextPage = Math.min(Math.max(1, Math.floor(value)), sessionTotalPages);
    if (nextPage !== sessionCurrentPage) {
        sessionCurrentPage = nextPage;
        fetchSessions();
    }
});
sessionPageInput.addEventListener('blur', (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) {
        event.target.value = String(sessionCurrentPage);
        return;
    }
    const nextPage = Math.min(Math.max(1, Math.floor(value)), sessionTotalPages);
    if (nextPage !== sessionCurrentPage) {
        sessionCurrentPage = nextPage;
        fetchSessions();
    } else {
        event.target.value = String(sessionCurrentPage);
    }
});
sessionTableBody.addEventListener('click', (event) => {
    const link = event.target.closest('.session-link');
    if (!link) return;
    const sessionId = decodeURIComponent(link.dataset.sessionId || '');
    if (!sessionId) return;
    selectSession(sessionId);
});

// 세션 상세 이벤트
document.getElementById('sessionBackBtn').addEventListener('click', goBackToSessionList);
document.getElementById('sessionDetailSearchBtn').addEventListener('click', () => {
    sessionDetailCurrentPage = 1;
    sessionDetailKeyword = document.getElementById('sessionDetailKeyword').value;
    fetchSessionDetail(selectedSessionId, 1, sessionDetailKeyword);
});
document.getElementById('sessionDetailResetBtn').addEventListener('click', () => {
    sessionDetailCurrentPage = 1;
    sessionDetailKeyword = '';
    document.getElementById('sessionDetailKeyword').value = '';
    fetchSessionDetail(selectedSessionId, 1, '');
});
document.getElementById('sessionDetailPrevBtn').addEventListener('click', () => {
    if (sessionDetailCurrentPage > 1) {
        sessionDetailCurrentPage -= 1;
        fetchSessionDetail(selectedSessionId, sessionDetailCurrentPage, sessionDetailKeyword);
    }
});
document.getElementById('sessionDetailNextBtn').addEventListener('click', () => {
    if (sessionDetailCurrentPage < sessionDetailTotalPages) {
        sessionDetailCurrentPage += 1;
        fetchSessionDetail(selectedSessionId, sessionDetailCurrentPage, sessionDetailKeyword);
    }
});
sessionDetailPageInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;
    const nextPage = Math.min(Math.max(1, Math.floor(value)), sessionDetailTotalPages);
    if (nextPage !== sessionDetailCurrentPage) {
        sessionDetailCurrentPage = nextPage;
        fetchSessionDetail(selectedSessionId, sessionDetailCurrentPage, sessionDetailKeyword);
    }
});
sessionDetailPageInput.addEventListener('blur', (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) {
        event.target.value = String(sessionDetailCurrentPage);
        return;
    }
    const nextPage = Math.min(Math.max(1, Math.floor(value)), sessionDetailTotalPages);
    if (nextPage !== sessionDetailCurrentPage) {
        sessionDetailCurrentPage = nextPage;
        fetchSessionDetail(selectedSessionId, sessionDetailCurrentPage, sessionDetailKeyword);
    } else {
        event.target.value = String(sessionDetailCurrentPage);
    }
});

logRefreshBtn.addEventListener('click', () => {
    stopLogStream();
    initLogsView({ preselectDate: selectedLogDate, preselectFile: selectedLogFile });
});
logCurrentBtn.addEventListener('click', selectCurrentLog);
logFullViewBtn.addEventListener('click', fetchLogFull);
logStreamBtn.addEventListener('click', () => {
    if (logIsStreaming) {
        stopLogStream('스트리밍 중지');
    } else {
        startLogStream();
    }
});
logPauseBtn.addEventListener('click', toggleLogPause);
logClearBtn.addEventListener('click', () => {
    logOutput.innerHTML = '';
    setLogStatus('');
});
logAutoScrollToggle.addEventListener('change', () => {
    if (logAutoScrollToggle.checked) {
        logOutput.scrollTop = logOutput.scrollHeight;
    }
});

document.getElementById('selectAllUsers').addEventListener('change', (event) => {
    document.querySelectorAll('.user-check').forEach((checkbox) => {
        checkbox.checked = event.target.checked;
    });
});

document.getElementById('deleteUsersBtn').addEventListener('click', deleteSelectedUsers);
document.getElementById('searchUsersBtn').addEventListener('click', () => {
    userCurrentPage = 1;
    fetchUsers();
});
document.getElementById('resetUsersBtn').addEventListener('click', resetUserFilters);
document.getElementById('userSearchTypeSelect').addEventListener('change', updateUserSearchFields);
document.getElementById('userPrevPageBtn').addEventListener('click', () => {
    if (userCurrentPage > 1) {
        userCurrentPage -= 1;
        fetchUsers();
    }
});
document.getElementById('userNextPageBtn').addEventListener('click', () => {
    if (userCurrentPage < userTotalPages) {
        userCurrentPage += 1;
        fetchUsers();
    }
});
userPageInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;
    const nextPage = Math.min(Math.max(1, Math.floor(value)), userTotalPages);
    if (nextPage !== userCurrentPage) {
        userCurrentPage = nextPage;
        fetchUsers();
    }
});
userPageInput.addEventListener('blur', (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) {
        event.target.value = String(userCurrentPage);
        return;
    }
    const nextPage = Math.min(Math.max(1, Math.floor(value)), userTotalPages);
    if (nextPage !== userCurrentPage) {
        userCurrentPage = nextPage;
        fetchUsers();
    } else {
        event.target.value = String(userCurrentPage);
    }
});
document.getElementById('searchGenerationsBtn').addEventListener('click', () => {
    currentPage = 1;
    fetchGenerations();
});
document.getElementById('searchTypeSelect').addEventListener('change', updateSearchFields);
document.getElementById('resetGenerationsBtn').addEventListener('click', resetFilters);
document.getElementById('prevPageBtn').addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage -= 1;
        fetchGenerations();
    }
});
document.getElementById('nextPageBtn').addEventListener('click', () => {
    if (currentPage < totalPages) {
        currentPage += 1;
        fetchGenerations();
    }
});
pageInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;
    const nextPage = Math.min(Math.max(1, Math.floor(value)), totalPages);
    if (nextPage !== currentPage) {
        currentPage = nextPage;
        fetchGenerations();
    }
});
pageInput.addEventListener('blur', (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) {
        event.target.value = String(currentPage);
        return;
    }
    const nextPage = Math.min(Math.max(1, Math.floor(value)), totalPages);
    if (nextPage !== currentPage) {
        currentPage = nextPage;
        fetchGenerations();
    } else {
        event.target.value = String(currentPage);
    }
});
document.getElementById('loginBtn').addEventListener('click', handleLogin);
document.getElementById('logoutBtn').addEventListener('click', handleLogout);
document.addEventListener('click', (event) => {
    const button = event.target.closest('.text-more');
    if (!button) return;
    const title = button.dataset.title || '텍스트 상세';
    const fullText = decodeURIComponent(button.dataset.fullText || '');
    openTextModal(title, fullText);
});
document.addEventListener('click', (event) => {
    const link = event.target.closest('.user-link');
    if (!link) return;
    event.preventDefault();
    const loginId = decodeURIComponent(link.dataset.loginId || '');
    const name = decodeURIComponent(link.dataset.name || '');
    const userId = decodeURIComponent(link.dataset.userId || '');
    const sessionId = decodeURIComponent(link.dataset.sessionId || '');
    openUserModal({ loginId, name, userId, sessionId });
});
textModal.addEventListener('click', closeTextModal);
userModal.addEventListener('click', closeUserModal);
document.querySelectorAll('.text-modal-panel').forEach((panel) => {
    panel.addEventListener('click', (event) => {
        event.stopPropagation();
    });
});
textModalClose.addEventListener('click', closeTextModal);
userModalClose.addEventListener('click', closeUserModal);

window.addEventListener('load', async () => {
    resetSessionFilters({ shouldFetch: false });
    resetUserFilters({ shouldFetch: false });
    resetFilters({ shouldFetch: false });
    if (currentView === 'sessions') {
        await fetchSessions();
    } else if (currentView === 'session-detail') {
        setActiveView('sessions');
        await fetchSessions();
    } else if (currentView === 'logs') {
        await initLogsView();
    } else if (currentView === 'generations') {
        await fetchGenerations();
    } else {
        await fetchUsers();
    }
});

