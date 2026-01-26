
        const API_BASE = window.location.origin;
        const VIEW_STORAGE_KEY = 'admin_view';
        const limit = 5;
        const userLimit = 15;
        const sessionLimit = 20;
        const messageSearchLimit = 20;
        let currentPage = 1;
        let totalPages = 1;
        let isReady = false;
        let currentView = 'sessions';
        const DEFAULT_SEARCH_TYPE = 'login_id';
        const DEFAULT_USER_SEARCH_TYPE = 'login_id';
        let userCurrentPage = 1;
        let userTotalPages = 1;
        let sessionCurrentPage = 1;
        let sessionTotalPages = 1;
        let selectedSessionId = null;
        let sessionMessageOffset = 0;
        const sessionMessageLimit = 200;
        let sessionMessageTotal = 0;
        let sessionLoadedCount = 0;
        let messageSearchPage = 1;
        let messageSearchTotalPages = 1;
        let logEventSource = null;
        let logIsStreaming = false;
        let logPaused = false;
        let logBuffer = [];

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
        const sessionMeta = document.getElementById('sessionMeta');
        const sessionTimeline = document.getElementById('sessionTimeline');
        const sessionGenerations = document.getElementById('sessionGenerations');
        const sessionLogJumpBtn = document.getElementById('sessionLogJumpBtn');
        const sessionLoadMoreBtn = document.getElementById('sessionLoadMoreBtn');
        const messageSearchTableBody = document.getElementById('messageSearchTableBody');
        const messageSearchCount = document.getElementById('messageSearchCount');
        const messagePageInfo = document.getElementById('messagePageInfo');
        const messagePageInput = document.getElementById('messagePageInput');
        const logStatus = document.getElementById('logStatus');
        const logDateSelect = document.getElementById('logDateSelect');
        const logFileSelect = document.getElementById('logFileSelect');
        const logLinesInput = document.getElementById('logLinesInput');
        const logKeywordInput = document.getElementById('logKeywordInput');
        const logLevelSelect = document.getElementById('logLevelSelect');
        const logMaskToggle = document.getElementById('logMaskToggle');
        const logTailBtn = document.getElementById('logTailBtn');
        const logStreamBtn = document.getElementById('logStreamBtn');
        const logPauseBtn = document.getElementById('logPauseBtn');
        const logAutoScrollToggle = document.getElementById('logAutoScrollToggle');
        const logOutput = document.getElementById('logOutput');
        const logClearBtn = document.getElementById('logClearBtn');
        const logRefreshBtn = document.getElementById('logRefreshBtn');

        const TEXT_PREVIEW_LIMIT = 120;
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
                btn.classList.toggle('active', btn.dataset.view === view);
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
        const allowedViews = Object.keys(viewSections);
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

        function buildTextPreview(text, title) {
            if (!hasText(text)) return '-';
            const raw = String(text ?? '');
            const { preview, truncated } = truncateText(raw);
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

            const query = document.getElementById('sessionFilterQuery').value;
            const userId = document.getElementById('sessionFilterUserId').value;
            const fromDate = document.getElementById('sessionFilterFromDate').value;
            const toDate = document.getElementById('sessionFilterToDate').value;
            if (query) params.set('query', query);
            if (userId) params.set('user_id', userId);
            if (fromDate) params.set('from', fromDate);
            if (toDate) params.set('to', toDate);

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
                const statusBadge = item.user_id
                    ? '<span class="badge badge-success">authed</span>'
                    : '<span class="badge badge-neutral">guest</span>';
                row.innerHTML = `
                    <td><button type="button" class="link-button session-link" data-session-id="${encodeURIComponent(item.session_id || '')}">${escapeHtml(item.session_id)}</button></td>
                    <td>${loginLabel}</td>
                    <td>${item.user_id ?? '-'}</td>
                    <td>${formatDateTime(item.created_at)}</td>
                    <td>${formatDateTime(item.last_message_at)}</td>
                    <td>${item.message_count ?? 0}</td>
                    <td>${statusBadge}</td>
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
            updateSessionSelectionHighlight();
        }

        function updateSessionSelectionHighlight() {
            document.querySelectorAll('.session-row').forEach((row) => {
                const rowId = decodeURIComponent(row.dataset.sessionId || '');
                row.classList.toggle('is-selected', rowId === selectedSessionId);
            });
        }

        function renderSessionMeta(detail) {
            const logHint = detail.log_hint ? `${detail.log_hint.date}/${detail.log_hint.file}` : '-';
            const rows = [
                { label: 'session_id', value: detail.session_id },
                { label: 'login_id', value: detail.login_id || 'guest' },
                { label: 'user_id', value: detail.user_id ?? 'guest' },
                { label: 'created_at', value: formatDateTime(detail.created_at) },
                { label: 'last_message_at', value: formatDateTime(detail.last_message_at) },
                { label: 'message_count', value: detail.message_count ?? 0 },
                { label: 'run_id', value: detail.run_id || '-' },
                { label: 'log_file', value: logHint },
            ];
            sessionMeta.innerHTML = rows
                .map(({ label, value }) => `
                    <div class="meta-row">
                        <span class="meta-label">${escapeHtml(label)}</span>
                        <span class="meta-value">${escapeHtml(String(value ?? '-'))}</span>
                    </div>
                `)
                .join('');
        }

        function buildSessionMessageItem(message) {
            const roleLabel = message.role === 'user' ? 'user' : 'assistant';
            const thumbUrl = message.image?.file_hash ? buildImageUrl(message.image.file_hash, IMAGE_SIZE.thumb) : null;
            const fullUrl = message.image?.file_hash ? buildImageUrl(message.image.file_hash, IMAGE_SIZE.full) : null;
            const imageHtml = thumbUrl
                ? `<img class="thumb timeline-thumb" src="${thumbUrl}" loading="lazy" alt="preview" onclick="openImageModal('${fullUrl}')">`
                : '';
            return `
                <div class="timeline-item ${roleLabel}">
                    <div class="timeline-meta">
                        <span class="badge badge-tight">${escapeHtml(roleLabel)}</span>
                        <span class="status-text">${formatDateTime(message.created_at)}</span>
                    </div>
                    <div class="timeline-content">${escapeHtml(message.content)}</div>
                    ${imageHtml}
                </div>
            `;
        }

        function renderSessionMessages(messages, { append = false } = {}) {
            const html = messages.map(buildSessionMessageItem).join('');
            if (append) {
                sessionTimeline.insertAdjacentHTML('afterbegin', html);
            } else {
                sessionTimeline.innerHTML = html || '<div class="empty-state">메시지가 없습니다.</div>';
            }
        }

        function renderSessionGenerations(generations) {
            if (!generations || !generations.length) {
                sessionGenerations.innerHTML = '<div class="empty-state">생성 이력이 없습니다.</div>';
                return;
            }
            const rows = generations.map((gen) => {
                const content = buildTextPreview(gen.output_text, '생성 결과');
                const imageHtml = gen.output_image ? buildImageCell(gen.output_image) : '-';
                return `
                    <div class="generation-card">
                        <div class="meta-row">
                            <span class="meta-label">type</span>
                            <span class="meta-value">${escapeHtml(gen.content_type || '-')}</span>
                        </div>
                        <div class="meta-row">
                            <span class="meta-label">created_at</span>
                            <span class="meta-value">${formatDateTime(gen.created_at)}</span>
                        </div>
                        <div class="generation-content">
                            ${content}
                            ${imageHtml !== '-' ? imageHtml : ''}
                        </div>
                    </div>
                `;
            }).join('');
            sessionGenerations.innerHTML = rows;
        }

        function updateSessionDetailSubtitle() {
            if (!selectedSessionId) {
                sessionDetailSubtitle.textContent = '';
                return;
            }
            sessionDetailSubtitle.textContent = `총 ${sessionMessageTotal}건 / 로드 ${sessionLoadedCount}건`;
        }

        function updateSessionLoadMoreState() {
            const hasMore = sessionLoadedCount < sessionMessageTotal;
            sessionLoadMoreBtn.disabled = !hasMore;
        }

        async function fetchSessionDetail(sessionId, { append = false, offset = 0 } = {}) {
            const params = new URLSearchParams();
            params.set('message_limit', String(sessionMessageLimit));
            params.set('message_offset', String(offset));
            const res = await fetch(`${API_BASE}/admin/sessions/${encodeURIComponent(sessionId)}?${params.toString()}`, {
                credentials: 'include',
            });
            if (!res.ok) {
                const data = await safeJson(res);
                showLogin(data.detail || "관리자 권한이 필요합니다.");
                return false;
            }
            const data = await safeJson(res);
            sessionMessageTotal = data.message_count || 0;

            if (!append) {
                sessionTimeline.innerHTML = '';
                sessionLoadedCount = 0;
                sessionDetailBody.classList.remove('hidden');
                sessionDetailEmpty.classList.add('hidden');
            }

            const messages = data.messages || [];
            renderSessionMessages(messages, { append });
            sessionLoadedCount += messages.length;

            renderSessionMeta(data);
            renderSessionGenerations(data.generations || []);
            updateSessionDetailSubtitle();
            updateSessionLoadMoreState();

            if (data.log_hint) {
                sessionLogJumpBtn.disabled = false;
                sessionLogJumpBtn.dataset.date = data.log_hint.date;
                sessionLogJumpBtn.dataset.file = data.log_hint.file;
            } else {
                sessionLogJumpBtn.disabled = true;
                sessionLogJumpBtn.dataset.date = '';
                sessionLogJumpBtn.dataset.file = '';
            }

            sessionMessageOffset = offset;
            hideLogin();
            isReady = true;
            return true;
        }

        async function selectSession(sessionId) {
            selectedSessionId = sessionId;
            sessionMessageOffset = 0;
            sessionLoadedCount = 0;
            sessionLogJumpBtn.disabled = true;
            sessionLogJumpBtn.dataset.date = '';
            sessionLogJumpBtn.dataset.file = '';
            updateSessionSelectionHighlight();
            await fetchSessionDetail(sessionId, { append: false, offset: 0 });
        }

        async function loadMoreSessionMessages() {
            if (!selectedSessionId) return;
            const nextOffset = sessionMessageOffset + sessionMessageLimit;
            const success = await fetchSessionDetail(selectedSessionId, { append: true, offset: nextOffset });
            if (success) {
                sessionMessageOffset = nextOffset;
            }
        }

        async function fetchMessageSearch() {
            const params = new URLSearchParams();
            params.set('limit', String(messageSearchLimit));
            params.set('offset', String((messageSearchPage - 1) * messageSearchLimit));

            const keyword = document.getElementById('messageSearchKeyword').value;
            const level = document.getElementById('messageSearchLevel').value;
            const fromDate = document.getElementById('messageSearchFromDate').value;
            const toDate = document.getElementById('messageSearchToDate').value;
            if (keyword) params.set('query', keyword);
            if (level) params.set('level', level);
            if (fromDate) params.set('from', fromDate);
            if (toDate) params.set('to', toDate);

            const res = await fetch(`${API_BASE}/admin/messages?${params.toString()}`, {
                credentials: 'include',
            });
            if (!res.ok) {
                const data = await safeJson(res);
                showLogin(data.detail || "관리자 권한이 필요합니다.");
                return;
            }

            const data = await safeJson(res);
            const items = data.items || [];
            messageSearchTableBody.innerHTML = '';

            items.forEach((item) => {
                const row = document.createElement('tr');
                const sessionButton = `
                    <button type="button" class="link-button session-link" data-session-id="${encodeURIComponent(item.session_id || '')}">
                        ${escapeHtml(item.session_id)}
                    </button>
                `;
                const roleBadge = item.role === 'assistant'
                    ? '<span class="badge badge-tight">assistant</span>'
                    : '<span class="badge badge-tight badge-neutral">user</span>';
                row.innerHTML = `
                    <td>${sessionButton}</td>
                    <td>${roleBadge}</td>
                    <td>${buildTextPreview(item.content, '메시지')}</td>
                    <td>${formatDateTime(item.created_at)}</td>
                `;
                messageSearchTableBody.appendChild(row);
            });

            const total = data.total || 0;
            messageSearchTotalPages = Math.max(1, Math.ceil(total / messageSearchLimit));
            if (messageSearchPage > messageSearchTotalPages) {
                messageSearchPage = messageSearchTotalPages;
                await fetchMessageSearch();
                return;
            }
            messageSearchCount.textContent = `총 ${total}건`;
            messagePageInfo.textContent = ` / ${messageSearchTotalPages}`;
            messagePageInput.value = String(messageSearchPage);
            hideLogin();
            isReady = true;
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
                const escaped = escapeHtml(line);
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
            if (!dates.length) {
                logDateSelect.innerHTML = '<option value="">날짜 없음</option>';
                logFileSelect.innerHTML = '<option value="">파일 없음</option>';
                return;
            }
            logDateSelect.innerHTML = dates.map((item) => `<option value="${item}">${item}</option>`).join('');
            if (preselectDate && dates.includes(preselectDate)) {
                logDateSelect.value = preselectDate;
            }
            await fetchLogFiles({ preselectFile });
        }

        async function fetchLogFiles({ preselectFile = null } = {}) {
            const date = logDateSelect.value;
            if (!date) {
                logFileSelect.innerHTML = '<option value="">파일 없음</option>';
                return;
            }
            const res = await fetch(`${API_BASE}/admin/logs/files?date=${encodeURIComponent(date)}`, {
                credentials: 'include',
            });
            if (!res.ok) {
                const data = await safeJson(res);
                showLogin(data.detail || "관리자 권한이 필요합니다.");
                return;
            }
            const data = await safeJson(res);
            const files = data.files || [];
            if (!files.length) {
                logFileSelect.innerHTML = '<option value="">파일 없음</option>';
                return;
            }
            logFileSelect.innerHTML = files.map((item) => {
                const label = `${item.name} (${formatBytes(item.size_bytes)})`;
                return `<option value="${item.name}">${label}</option>`;
            }).join('');
            if (preselectFile && files.some((item) => item.name === preselectFile)) {
                logFileSelect.value = preselectFile;
            }
        }

        async function fetchLogTail() {
            stopLogStream();
            const date = logDateSelect.value;
            const file = logFileSelect.value;
            if (!date || !file) {
                setLogStatus('로그 파일을 선택하세요.');
                return;
            }
            const params = new URLSearchParams();
            params.set('date', date);
            params.set('file', file);
            const lines = Number(logLinesInput.value) || 400;
            params.set('lines', String(lines));
            const keyword = logKeywordInput.value.trim();
            const level = logLevelSelect.value;
            if (keyword) params.set('query', keyword);
            if (level) params.set('level', level);
            params.set('mask', logMaskToggle.checked ? 'true' : 'false');

            const res = await fetch(`${API_BASE}/admin/logs/tail?${params.toString()}`, {
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
            setLogStatus(`미리보기 ${lines}줄`);
            hideLogin();
            isReady = true;
        }

        function startLogStream() {
            if (logIsStreaming) return;
            const date = logDateSelect.value;
            const file = logFileSelect.value;
            if (!date || !file) {
                setLogStatus('로그 파일을 선택하세요.');
                return;
            }
            const params = new URLSearchParams();
            params.set('date', date);
            params.set('file', file);
            const keyword = logKeywordInput.value.trim();
            const level = logLevelSelect.value;
            if (keyword) params.set('query', keyword);
            if (level) params.set('level', level);
            params.set('mask', logMaskToggle.checked ? 'true' : 'false');

            logEventSource = new EventSource(`${API_BASE}/admin/logs/stream?${params.toString()}`);
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
            logStreamBtn.textContent = 'Stop';
            logPauseBtn.disabled = false;
            logPauseBtn.textContent = 'Pause';
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
            logStreamBtn.textContent = 'Start';
            logPauseBtn.disabled = true;
            logPauseBtn.textContent = 'Pause';
            if (message) {
                setLogStatus(message);
            }
        }

        function toggleLogPause() {
            if (!logIsStreaming) return;
            logPaused = !logPaused;
            logPauseBtn.textContent = logPaused ? 'Resume' : 'Pause';
            if (!logPaused) {
                flushLogBuffer();
            }
        }

        async function initLogsView({ preselectDate = null, preselectFile = null } = {}) {
            if (!logLinesInput.value) {
                logLinesInput.value = '400';
            }
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
            resetMessageSearchFilters({ shouldFetch: false });
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
            document.getElementById('sessionFilterQuery').value = '';
            document.getElementById('sessionFilterUserId').value = '';
            document.getElementById('sessionFilterFromDate').value = '';
            document.getElementById('sessionFilterToDate').value = '';
            sessionCurrentPage = 1;
            selectedSessionId = null;
            sessionMessageOffset = 0;
            sessionMessageTotal = 0;
            sessionLoadedCount = 0;
            sessionDetailSubtitle.textContent = '';
            sessionMeta.innerHTML = '';
            sessionTimeline.innerHTML = '';
            sessionGenerations.innerHTML = '';
            sessionDetailBody.classList.add('hidden');
            sessionDetailEmpty.classList.remove('hidden');
            sessionLogJumpBtn.disabled = true;
            sessionLoadMoreBtn.disabled = true;
            updateSessionSelectionHighlight();
            if (shouldFetch && isReady) {
                fetchSessions();
            }
        }

        function resetMessageSearchFilters({ shouldFetch = true } = {}) {
            document.getElementById('messageSearchKeyword').value = '';
            document.getElementById('messageSearchLevel').value = '';
            document.getElementById('messageSearchFromDate').value = '';
            document.getElementById('messageSearchToDate').value = '';
            messageSearchPage = 1;
            messageSearchTableBody.innerHTML = '';
            messageSearchCount.textContent = '';
            messagePageInfo.textContent = '';
            messagePageInput.value = '1';
            if (shouldFetch && isReady) {
                fetchMessageSearch();
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
                        resetMessageSearchFilters({ shouldFetch: false });
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
        sessionLoadMoreBtn.addEventListener('click', loadMoreSessionMessages);
        sessionLogJumpBtn.addEventListener('click', async () => {
            const date = sessionLogJumpBtn.dataset.date;
            const file = sessionLogJumpBtn.dataset.file;
            if (!date || !file) return;
            setActiveView('logs');
            await initLogsView({ preselectDate: date, preselectFile: file });
        });

        document.getElementById('messageSearchBtn').addEventListener('click', () => {
            messageSearchPage = 1;
            fetchMessageSearch();
        });
        document.getElementById('messageResetBtn').addEventListener('click', resetMessageSearchFilters);
        document.getElementById('messagePrevPageBtn').addEventListener('click', () => {
            if (messageSearchPage > 1) {
                messageSearchPage -= 1;
                fetchMessageSearch();
            }
        });
        document.getElementById('messageNextPageBtn').addEventListener('click', () => {
            if (messageSearchPage < messageSearchTotalPages) {
                messageSearchPage += 1;
                fetchMessageSearch();
            }
        });
        messagePageInput.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            const value = Number(event.target.value);
            if (!Number.isFinite(value)) return;
            const nextPage = Math.min(Math.max(1, Math.floor(value)), messageSearchTotalPages);
            if (nextPage !== messageSearchPage) {
                messageSearchPage = nextPage;
                fetchMessageSearch();
            }
        });
        messagePageInput.addEventListener('blur', (event) => {
            const value = Number(event.target.value);
            if (!Number.isFinite(value)) {
                event.target.value = String(messageSearchPage);
                return;
            }
            const nextPage = Math.min(Math.max(1, Math.floor(value)), messageSearchTotalPages);
            if (nextPage !== messageSearchPage) {
                messageSearchPage = nextPage;
                fetchMessageSearch();
            } else {
                event.target.value = String(messageSearchPage);
            }
        });
        messageSearchTableBody.addEventListener('click', (event) => {
            const link = event.target.closest('.session-link');
            if (!link) return;
            const sessionId = decodeURIComponent(link.dataset.sessionId || '');
            if (!sessionId) return;
            setActiveView('sessions');
            selectSession(sessionId);
        });

        logRefreshBtn.addEventListener('click', () => {
            stopLogStream();
            initLogsView({ preselectDate: logDateSelect.value, preselectFile: logFileSelect.value });
        });
        logDateSelect.addEventListener('change', () => {
            stopLogStream();
            fetchLogFiles();
        });
        logFileSelect.addEventListener('change', () => {
            stopLogStream();
        });
        logTailBtn.addEventListener('click', fetchLogTail);
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
            resetMessageSearchFilters({ shouldFetch: false });
            resetUserFilters({ shouldFetch: false });
            resetFilters({ shouldFetch: false });
            sessionLoadMoreBtn.disabled = true;
            sessionLogJumpBtn.disabled = true;
            if (currentView === 'sessions') {
                await fetchSessions();
            } else if (currentView === 'logs') {
                await initLogsView();
            } else if (currentView === 'generations') {
                await fetchGenerations();
            } else {
                await fetchUsers();
            }
        });
    
