
        const API_BASE = window.location.origin;
        const VIEW_STORAGE_KEY = 'admin_view';
        const limit = 5;
        const userLimit = 15;
        let currentPage = 1;
        let totalPages = 1;
        let isReady = false;
        let currentView = 'users';
        const DEFAULT_SEARCH_TYPE = 'login_id';
        const DEFAULT_USER_SEARCH_TYPE = 'login_id';
        let userCurrentPage = 1;
        let userTotalPages = 1;

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

        const TEXT_PREVIEW_LIMIT = 120;

        function escapeHtml(str) {
            return String(str ?? "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#039;");
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
            document.getElementById('view-users').classList.toggle('hidden', view !== 'users');
            document.getElementById('view-generations').classList.toggle('hidden', view !== 'generations');
        }

        function getSavedView() {
            try {
                return sessionStorage.getItem(VIEW_STORAGE_KEY);
            } catch {
                return null;
            }
        }

        const initialView = getSavedView();
        if (initialView === 'generations') {
            setActiveView('generations');
        } else {
            setActiveView('users');
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
                    <span class="text-preview">${escapeHtml(preview)}</span>
                    ${button}
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
                    <span class="text-preview">${escapeHtml(preview)}</span>
                    ${button}
                </div>
            `;
        }

        function buildImageCell(image) {
            if (!image || !image.file_hash) return '-';
            const url = `${API_BASE}/images/${image.file_hash}`;
            return `<img class="thumb" src="${url}" alt="preview" onclick="openImageModal('${url}')">`;
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
                    label: '타입',
                    value: item.content_type,
                    format: (value) =>
                        isMetaValuePresent(value)
                            ? `<span class="badge">${escapeHtml(String(value))}</span>`
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
                const userCell = item.login_id
                    ? `
                        <a href="#" class="user-link"
                            data-login-id="${encodeURIComponent(item.login_id)}"
                            data-name="${encodeURIComponent(item.name || '')}"
                            data-user-id="${encodeURIComponent(item.user_id ?? '')}"
                            data-session-id="${encodeURIComponent(item.session_id || '')}">
                            ${escapeHtml(item.login_id)}
                        </a>
                    `
                    : 'guest';
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
            resetUserFilters({ shouldFetch: false });
            await fetchUsers();
            if (currentView === 'generations') {
                resetFilters({ shouldFetch: true });
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
                if (nextView === 'generations') {
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
            resetUserFilters({ shouldFetch: false });
            if (currentView === 'generations') {
                resetFilters({ shouldFetch: false });
            }
            await fetchUsers();
            if (currentView === 'generations' && isReady) {
                fetchGenerations();
            }
        });
    