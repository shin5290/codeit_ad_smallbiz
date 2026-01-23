
        const API_BASE = window.location.origin;
        const SCROLL_BOTTOM_THRESHOLD = 80;
        const IMAGE_GUIDE_STORAGE_KEY = "hide_image_zoom_guide";
        const IMAGE_SIZE = {
            thumb: "thumb",
            full: "full",
        };
        let currentSessionId = null;
        let isSignup = false;
        let isLoggedIn = false;
        let isSending = false;
        let selectedImageFile = null; // 단일 파일 객체 저장
        let oldestMessageId = null;
        let nextCursor = null;
        let isLoadingHistory = false;
        let hasUnreadMessages = false;
        let toastTimer = null;

        // ====== 안전 JSON 파서 ======
        async function safeJson(res) {
            try { return await res.json(); } catch { return {}; }
        }

        function shouldAutoScroll(box, threshold = SCROLL_BOTTOM_THRESHOLD) {
            if (!box) return false;
            const distance = box.scrollHeight - box.scrollTop - box.clientHeight;
            return distance <= threshold;
        }

        function scrollToBottom(box) {
            if (box) box.scrollTop = box.scrollHeight;
        }

        function updateScrollButtonState() {
            const box = document.getElementById('chatBox');
            const button = document.getElementById('scrollToBottomBtn');
            const badge = document.getElementById('scrollToBottomBadge');
            if (!box || !button) return;
            const isNearBottom = shouldAutoScroll(box);
            button.classList.toggle('hidden', isNearBottom);
            if (isNearBottom) {
                hasUnreadMessages = false;
            }
            if (badge) {
                badge.classList.toggle('hidden', isNearBottom || !hasUnreadMessages);
            }
        }

        function updateScrollButtonOffset() {
            const container = document.querySelector('.chat-container');
            const inputArea = document.querySelector('.input-area');
            if (!container || !inputArea) return;
            const height = inputArea.getBoundingClientRect().height;
            container.style.setProperty('--input-area-height', `${height}px`);
        }

        function markUnreadMessage() {
            hasUnreadMessages = true;
            updateScrollButtonState();
        }

        function scrollToLatest() {
            const box = document.getElementById('chatBox');
            if (!box) return;
            hasUnreadMessages = false;
            box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
            updateScrollButtonState();
        }

        function handleChatScroll() {
            const box = document.getElementById('chatBox');
            if (!box) return;
            if (box.scrollTop === 0) loadOlderHistory();
            updateScrollButtonState();
        }

        // ====== 채팅 렌더 유틸 ======
        function clearChatBox() {
            const content = document.getElementById('chatContent');
            const loading = document.getElementById('chatLoading');
            if (!content) return;
            content.innerHTML = '';
            if (loading) content.appendChild(loading);
        }

        function setChatLoading(isLoading) {
            const loading = document.getElementById('chatLoading');
            if (!loading) return;
            loading.classList.toggle('hidden', !isLoading);
        }

        async function fetchHistoryPage(limit, cursor) {
            const params = new URLSearchParams();
            params.set("limit", String(limit));
            if (cursor) params.set("cursor", String(cursor));

            const res = await fetch(`${API_BASE}/chat/history?${params.toString()}`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: "include",
            });

            if (!res.ok) return { items: [], next_cursor: null };
            return await safeJson(res);
        }

        function resolveImageValue(image) {
            if (!image) return null;
            if (typeof image === 'string') return image;
            if (image.image_hash) return image.image_hash;
            if (image.file_hash) return image.file_hash;
            if (image.file_directory) return image.file_directory;
            return null;
        }

        function buildSizedImageUrl(imageValue, size) {
            if (!imageValue) return null;
            if (imageValue.startsWith('data:image/')) return imageValue;
            if (!imageValue.includes('/') && !imageValue.includes(':')) {
                return `${API_BASE}/images/${encodeURIComponent(imageValue)}?size=${size}`;
            }
            if (imageValue.startsWith('http')) {
                try {
                    const url = new URL(imageValue);
                    if (url.pathname.startsWith('/images/')) {
                        url.searchParams.set('size', size);
                    }
                    return url.toString();
                } catch {
                    return imageValue;
                }
            }
            if (imageValue.startsWith('/images/')) {
                const url = new URL(`${API_BASE}${imageValue}`);
                url.searchParams.set('size', size);
                return url.toString();
            }
            if (imageValue.startsWith('/')) {
                return `${API_BASE}${imageValue}`;
            }
            return `${API_BASE}/images/${encodeURIComponent(imageValue)}?size=${size}`;
        }

        function normalizeImageUrls(imageUrls) {
            const entries = Array.isArray(imageUrls)
                ? imageUrls
                : imageUrls
                    ? [imageUrls]
                    : [];
            return entries.map((item) => {
                if (!item) return null;
                if (typeof item === 'object' && (item.thumbUrl || item.fullUrl)) {
                    const fullUrl = item.fullUrl || item.thumbUrl;
                    const thumbUrl = item.thumbUrl || item.fullUrl;
                    return { fullUrl, thumbUrl };
                }
                const resolved = resolveImageValue(item);
                if (!resolved) return null;
                const fullUrl = buildSizedImageUrl(resolved, IMAGE_SIZE.full);
                const thumbUrl = buildSizedImageUrl(resolved, IMAGE_SIZE.thumb);
                return {
                    fullUrl: fullUrl || thumbUrl,
                    thumbUrl: thumbUrl || fullUrl,
                };
            }).filter(Boolean);
        }

        function getMessageContentContainer(container) {
            if (!container) return null;
            let content = container.querySelector('.msg-content');
            if (!content) {
                content = document.createElement('div');
                content.className = 'msg-content';
                if (container.firstChild) {
                    container.insertBefore(content, container.firstChild);
                } else {
                    container.appendChild(content);
                }
            }
            return content;
        }

        function setMessageRawText(container, text) {
            if (!container) return;
            container.dataset.rawText = text || "";
        }

        function setMessageImageUrls(container, imageUrls) {
            if (!container) return;
            container.dataset.imageUrls = JSON.stringify(imageUrls || []);
        }

        function getMessageImageUrls(container) {
            if (!container) return [];
            const raw = container.dataset.imageUrls;
            if (!raw) return [];
            try {
                const parsed = JSON.parse(raw);
                return Array.isArray(parsed) ? parsed.filter(Boolean) : [];
            } catch {
                return [];
            }
        }

        function getMessageRole(container) {
            if (!container) return null;
            if (container.classList.contains('assistant')) return 'assistant';
            if (container.classList.contains('user')) return 'user';
            return null;
        }

        function getMessageWrapper(container) {
            if (!container) return null;
            return container.closest('.msg-wrap');
        }

        function getMessageStack(container) {
            if (!container) return null;
            const stack = container.closest('.msg-stack');
            if (stack) return stack;
            const wrapper = getMessageWrapper(container);
            return wrapper ? wrapper.querySelector('.msg-stack') : null;
        }

        function clearMessageProgress(container) {
            if (!container) return;
            container.querySelectorAll('.assistant-progress').forEach((node) => node.remove());
        }

        function clearMessageImages(container) {
            if (!container) return;
            container.querySelectorAll('.image-wrapper').forEach((node) => node.remove());
            Array.from(container.children).forEach((child) => {
                if (child.tagName === 'IMG') child.remove();
            });
        }

        function clearMessageActions(container) {
            if (!container) return;
            const wrapper = getMessageWrapper(container) || container;
            const stack = getMessageStack(container) || wrapper;
            stack.querySelectorAll('.msg-actions').forEach((node) => node.remove());
        }

        function showToast(message) {
            const toast = document.getElementById('toast');
            if (!toast) return;
            toast.textContent = message;
            toast.classList.add('show');
            if (toastTimer) clearTimeout(toastTimer);
            toastTimer = setTimeout(() => {
                toast.classList.remove('show');
            }, 2000);
        }

        async function copyToClipboard(text) {
            if (!text) return;
            try {
                if (navigator.clipboard && window.isSecureContext) {
                    await navigator.clipboard.writeText(text);
                } else {
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    const succeeded = document.execCommand('copy');
                    textarea.remove();
                    if (!succeeded) throw new Error('copy failed');
                }
                showToast('복사 완료');
            } catch (err) {
                console.error(err);
                showToast('복사 실패');
            }
        }

        function getImageFilename(url, index = 0) {
            const fallback = `image_${Date.now()}_${index + 1}.png`;
            if (!url) return fallback;
            if (url.startsWith('data:')) return fallback;
            try {
                const parsed = new URL(url, window.location.origin);
                const name = parsed.pathname.split('/').pop();
                if (!name) return fallback;
                if (name.includes('.')) return name;
                return `${name}.png`;
            } catch {
                return fallback;
            }
        }

        function downloadImage(url, filename) {
            if (!url) return;
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = filename || 'image.png';
            anchor.rel = 'noopener';
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
        }

        function appendTextActions(container) {
            if (!container || getMessageRole(container) !== 'assistant') return;
            const rawText = (container.dataset.rawText || "").trim();
            const imageUrls = getMessageImageUrls(container);
            const hasImage = imageUrls.length > 0;
            const wrapper = getMessageWrapper(container) || container;
            const stack = getMessageStack(container) || wrapper;

            clearMessageActions(container);
            if (!rawText && !hasImage) return;

            const actions = document.createElement('div');
            actions.className = 'msg-actions';

            if (rawText) {
                const copyBtn = document.createElement('button');
                copyBtn.type = 'button';
                copyBtn.className = 'msg-action-btn';
                copyBtn.setAttribute('data-tooltip', '복사');
                copyBtn.setAttribute('aria-label', '광고문구 복사');
                copyBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                `;
                copyBtn.addEventListener('click', () => {
                    const latestText = (container.dataset.rawText || "").trim();
                    if (!latestText) return;
                    copyToClipboard(latestText);
                });
                actions.appendChild(copyBtn);
            }

            if (hasImage) {
                const downloadBtn = document.createElement('button');
                downloadBtn.type = 'button';
                downloadBtn.className = 'msg-action-btn';
                downloadBtn.setAttribute('data-tooltip', '다운로드');
                downloadBtn.setAttribute('aria-label', '이미지 다운로드');
                downloadBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                `;
                downloadBtn.addEventListener('click', () => {
                    const urls = getMessageImageUrls(container);
                    if (!urls.length) return;
                    downloadImage(urls[0], getImageFilename(urls[0], 0));
                });
                actions.appendChild(downloadBtn);
            }

            stack.appendChild(actions);
        }

        function appendMessageImages(container, imageUrls) {
            const entries = normalizeImageUrls(imageUrls);
            clearMessageImages(container);
            const fullUrls = [];
            entries.forEach((entry) => {
                const fullUrl = entry.fullUrl || entry.thumbUrl;
                const thumbUrl = entry.thumbUrl || entry.fullUrl;
                if (!fullUrl && !thumbUrl) return;
                const wrapper = document.createElement('div');
                wrapper.className = 'image-wrapper';

                const img = document.createElement('img');
                img.src = thumbUrl || fullUrl;
                img.alt = '첨부 이미지';
                img.loading = 'lazy';
                img.onclick = () => openImageModal(fullUrl || thumbUrl);

                wrapper.appendChild(img);
                container.appendChild(wrapper);
                fullUrls.push(fullUrl || thumbUrl);
            });
            setMessageImageUrls(container, fullUrls);
            return fullUrls;
        }

        function shouldShowImageGuide() {
            return !localStorage.getItem(IMAGE_GUIDE_STORAGE_KEY);
        }

        function dismissImageGuide(guide, { persist = false } = {}) {
            if (!guide) return;
            if (persist) localStorage.setItem(IMAGE_GUIDE_STORAGE_KEY, "1");
            guide.classList.add('hidden');
            setTimeout(() => guide.remove(), 150);
        }

        function showImageGuide(container) {
            if (!container || !shouldShowImageGuide()) return;
            if (container.querySelector('.image-guide')) return;
            const guide = document.createElement('div');
            guide.className = 'image-guide';

            const text = document.createElement('span');
            text.className = 'image-guide-text';
            text.textContent = '💡 이미지를 클릭하면 크게 볼 수 있어요!';

            const dismissBtn = document.createElement('button');
            dismissBtn.type = 'button';
            dismissBtn.className = 'image-guide-dismiss';
            dismissBtn.textContent = '×';
            dismissBtn.setAttribute('aria-label', '다시 보지 않기');
            dismissBtn.title = '다시 보지 않기';
            dismissBtn.addEventListener('click', () => dismissImageGuide(guide, { persist: true }));

            guide.appendChild(text);
            guide.appendChild(dismissBtn);
            container.appendChild(guide);
        }

        function createMessageElement(role, text, imageUrls, messageId) {
            const wrapper = document.createElement('div');
            wrapper.className = `msg-wrap ${role}`;

            const stack = document.createElement('div');
            stack.className = 'msg-stack';

            const div = document.createElement('div');
            div.className = `msg ${role}`;
            if (messageId) div.dataset.messageId = String(messageId);

            renderMessageInto(div, text);
            appendMessageImages(div, imageUrls);
            stack.appendChild(div);
            wrapper.appendChild(stack);
            appendTextActions(div);
            return wrapper;
        }

        function renderMessages(items, { prepend = false } = {}) {
            const content = document.getElementById('chatContent');
            if (!content) return;

            const fragment = document.createDocumentFragment();
            let lastGuideTarget = null;
            (items || []).forEach((m) => {
                const role = (m.role === "assistant") ? "assistant" : "user";
                const imageUrls = m.image ? [m.image] : [];

                const wrapper = createMessageElement(role, m.content, imageUrls, m.id);
                if (!prepend && role === 'assistant' && imageUrls.length) {
                    const msg = wrapper.querySelector('.msg');
                    lastGuideTarget = msg || lastGuideTarget;
                }
                fragment.appendChild(wrapper);
            });

            if (prepend) {
                content.insertBefore(fragment, content.firstChild);
            } else {
                content.appendChild(fragment);
            }

            if (!prepend && lastGuideTarget) {
                showImageGuide(lastGuideTarget);
            }
        }

        async function loadInitialHistory() {
            setChatLoading(true);
            const data = await fetchHistoryPage(15, null);
            const items = Array.isArray(data.items) ? data.items : [];

            clearChatBox();
            setChatLoading(false);
            if (items.length === 0) {
                const greetId = appendAssistantPlaceholder("");
                if (greetId) streamAssistantText(greetId, "안녕하세요! 어떤 광고 아이디어가 필요하신가요?");
                nextCursor = null;
                oldestMessageId = null;
                updateScrollButtonState();
                return;
            }

            renderMessages(items);
            oldestMessageId = items[0].id ?? null;
            nextCursor = data.next_cursor ?? null;

            const box = document.getElementById('chatBox');
            if (box) box.scrollTop = box.scrollHeight;
            updateScrollButtonState();
        }

        async function loadOlderHistory() {
            if (isLoadingHistory || !nextCursor || !oldestMessageId) return;
            isLoadingHistory = true;

            const box = document.getElementById('chatBox');
            const prevHeight = box ? box.scrollHeight : 0;

            const data = await fetchHistoryPage(10, oldestMessageId);
            const items = Array.isArray(data.items) ? data.items : [];

            if (items.length > 0) {
                renderMessages(items, { prepend: true });
                oldestMessageId = items[0].id ?? oldestMessageId;
                nextCursor = data.next_cursor ?? null;

                if (box) {
                    const newHeight = box.scrollHeight;
                    box.scrollTop = newHeight - prevHeight;
                }
            } else {
                nextCursor = null;
            }

            isLoadingHistory = false;
            updateScrollButtonState();
        }


    async function initAuth() {
        const hasCachedUser = applyCachedAuthUI();
        setAuthChecking(true, hasCachedUser);
        if (!hasCachedUser) {
          setChatLoading(false);
          clearChatBox();
          const greetId = appendAssistantPlaceholder("");
          if (greetId) streamAssistantText(greetId, "안녕하세요! 어떤 광고 아이디어가 필요하신가요?");
          showLoggedOutUI();
          setAuthChecking(false);
          return;
        }
        try {
          const res = await fetch(`${API_BASE}/auth/me`, {
              method: "GET",
              headers: {
              "Content-Type": "application/json",
                    },
                    credentials: "include",
                });

                if (!res.ok) {
                    setChatLoading(false);
                    clearChatBox();
                    const greetId = appendAssistantPlaceholder("");
                    if (greetId) streamAssistantText(greetId, "안녕하세요! 어떤 광고 아이디어가 필요하신가요?");
                    showLoggedOutUI();
                    return;
                }

                const data = await safeJson(res);
                if (data.is_admin) {
                    window.location.href = "/admin";
                    return;
                }

                const sessionRes = await fetch(`${API_BASE}/chat/session`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    credentials: "include",
                    body: JSON.stringify({
                        session_id: localStorage.getItem("session_id"),
                    }),
                });
                if (sessionRes.ok) {
                    const sessionData = await safeJson(sessionRes);
                    currentSessionId = sessionData.session_id ?? null;
                    if (currentSessionId) localStorage.setItem("session_id", currentSessionId);
                }

                onLoginSuccess(data.name);
                await loadInitialHistory();
                
            } catch (e) {
                console.error(e);
                showLoggedOutUI();
            } finally {
                setAuthChecking(false);
            }
        }

        // 페이지 로드되면 자동 복구
        window.addEventListener("DOMContentLoaded", () => {
            // 기존 로그인 복구
            initAuth();
            updateScrollButtonOffset();
            window.addEventListener('resize', updateScrollButtonOffset);

            const box = document.getElementById('chatBox');
            if (box) {
                box.addEventListener('scroll', handleChatScroll);
                updateScrollButtonState();
            }
        });

        // ====== 모달 제어 ======
        function openModal() {
            if (isSignup) toggleAuth();
            document.getElementById('auth-modal').classList.remove('hidden');
        }

        function closeModal() {
            document.getElementById('auth-modal').classList.add('hidden');
            if (isSignup) toggleAuth();
            document.getElementById('loginId').value = '';
            document.getElementById('password').value = '';
            document.getElementById('nickname').value = '';
        }

        function toggleAuth() {
            isSignup = !isSignup;
            document.getElementById('auth-title').innerText = isSignup ? "회원가입" : "로그인";
            document.getElementById('main-btn').innerText = isSignup ? "가입하기" : "로그인";
            document.getElementById('signup-fields').classList.toggle('hidden');
            document.getElementById('toggle-question').innerText = isSignup ? "이미 계정이 있으신가요?" : "계정이 없으신가요?";
            document.getElementById('toggle-link').innerText = isSignup ? "로그인하기" : "회원가입하기";
        }

        async function handleAuth() {
            const loginId = document.getElementById('loginId').value;
            const loginPw = document.getElementById('password').value;

            if (isSignup) {
                const name = document.getElementById('nickname').value;
                const res = await fetch(`${API_BASE}/auth/signup`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: "include",
                    body: JSON.stringify({ login_id: loginId, login_pw: loginPw, name: name })
                });
                const data = await safeJson(res);
                if (res.ok) { alert("가입 성공! 로그인해주세요."); toggleAuth(); }
                else { alert(data.detail ?? "회원가입 실패"); }
                return;
            }

            // 로그인
            const loginForm = new URLSearchParams();
            loginForm.set("username", loginId);
            loginForm.set("password", loginPw);
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                credentials: "include",
                body: loginForm.toString(),
            });

            const data = await safeJson(res);
            if (!res.ok) {
                alert(data.detail ?? "로그인 실패");
                return;
            }

            let displayName = "사용자";
            let isAdmin = false;
            const meRes = await fetch(`${API_BASE}/auth/me`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: "include",
            });
            if (meRes.ok) {
                const me = await safeJson(meRes);
                displayName = me.name ?? displayName;
                isAdmin = Boolean(me.is_admin);
            }
            if (isAdmin) {
                localStorage.setItem("display_name", displayName);
                window.location.href = "/admin";
                return;
            }

            const sessionRes = await fetch(`${API_BASE}/chat/session`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: "include",
                body: JSON.stringify({
                    session_id: localStorage.getItem("session_id"),
                }),
            });
            if (sessionRes.ok) {
                const sessionData = await safeJson(sessionRes);
                currentSessionId = sessionData.session_id ?? null;
                if (currentSessionId) localStorage.setItem("session_id", currentSessionId);
            }

            onLoginSuccess(displayName);
            await loadInitialHistory();

            closeModal();
        }

        function onLoginSuccess(name) {
            isLoggedIn = true;
            localStorage.setItem("display_name", name);
            document.getElementById('loginTabBtn').classList.add('hidden');
            document.getElementById('profileEditBtn').classList.remove('hidden');
            document.getElementById('logoutBtn').classList.remove('hidden');
            document.getElementById('userInfo').innerText = `${name}님 환영합니다!`;
            document.getElementById('userInfo').classList.remove('hidden');
        }

        function showLoggedOutUI() {
            isLoggedIn = false;
            localStorage.removeItem("display_name");
            document.getElementById('profileEditBtn').classList.add('hidden');
            document.getElementById('logoutBtn').classList.add('hidden');
            document.getElementById('loginTabBtn').classList.remove('hidden');
            document.getElementById('userInfo').classList.add('hidden');
        }

        function setAuthChecking(isChecking, hasCachedUser) {
            const userInfo = document.getElementById('userInfo');
            if (isChecking) {
                if (!hasCachedUser) {
                    userInfo.innerText = "로그인 확인 중...";
                    userInfo.classList.remove('hidden');
                    document.getElementById('loginTabBtn').classList.add('hidden');
                    document.getElementById('profileEditBtn').classList.add('hidden');
                    document.getElementById('logoutBtn').classList.add('hidden');
                }
            } else if (!isLoggedIn) {
                userInfo.classList.add('hidden');
                document.getElementById('loginTabBtn').classList.remove('hidden');
            }
        }

        function applyCachedAuthUI() {
            const cachedName = localStorage.getItem("display_name");
            if (!cachedName) return false;

            document.getElementById('loginTabBtn').classList.add('hidden');
            document.getElementById('profileEditBtn').classList.remove('hidden');
            document.getElementById('logoutBtn').classList.remove('hidden');
            document.getElementById('userInfo').innerText = `${cachedName}님 환영합니다!`;
            document.getElementById('userInfo').classList.remove('hidden');
            return true;
        }

        async function handleLogout() {
            try {
                await fetch(`${API_BASE}/auth/logout`, {
                    method: "POST",
                    credentials: "include",
                });
            } catch (e) {
                console.error(e);
            }
            localStorage.removeItem("session_id");
            localStorage.removeItem("display_name");
            currentSessionId = null;
            showLoggedOutUI();
            location.reload();
        }

        // ====== 회원정보 수정 모달 ======
        async function openProfileEditModal() {
            try {
                const res = await fetch(`${API_BASE}/auth/me`, {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    credentials: "include",
                });

                if (!res.ok) {
                    alert("로그인이 필요합니다.");
                    return;
                }

                const data = await safeJson(res);
                document.getElementById('editName').value = data.name || '';
                document.getElementById('editCurrentPassword').value = '';
                document.getElementById('editNewPassword').value = '';
                document.getElementById('profile-edit-modal').classList.remove('hidden');
            } catch (e) {
                console.error(e);
                alert("오류가 발생했습니다.");
            }
        }

        function closeProfileEditModal() {
            document.getElementById('profile-edit-modal').classList.add('hidden');
            document.getElementById('editName').value = '';
            document.getElementById('editCurrentPassword').value = '';
            document.getElementById('editNewPassword').value = '';
        }

        async function handleProfileUpdate() {
            const name = document.getElementById('editName').value;
            const currentPassword = document.getElementById('editCurrentPassword').value;
            const newPassword = document.getElementById('editNewPassword').value;

            // 현재 비밀번호는 필수
            if (!currentPassword.trim()) {
                alert("현재 비밀번호를 입력해주세요.");
                return;
            }

            const body = {
                current_password: currentPassword.trim()
            };
            if (name.trim()) body.name = name.trim();
            if (newPassword.trim()) body.new_password = newPassword.trim();

            try {
                const res = await fetch(`${API_BASE}/auth/user`, {
                    method: "PUT",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    credentials: "include",
                    body: JSON.stringify(body),
                });

                const data = await safeJson(res);

                if (!res.ok) {
                    alert(data.detail || "수정에 실패했습니다.");
                    return;
                }

                alert("회원정보가 수정되었습니다.");
                if (body.name) {
                    document.getElementById('userInfo').innerText = `${body.name}님 환영합니다!`;
                }
                closeProfileEditModal();
            } catch (e) {
                console.error(e);
                alert("오류가 발생했습니다.");
            }
        }

        // ====== 이미지 업로드 (단일 파일) ======
        function handleImageUpload(event) {
            const files = Array.from(event.target.files || []);
            if (!files.length) return;

            // 첫 번째 파일만 선택 (기존 파일 교체)
            selectedImageFile = files[0];

            renderPreviews();
            updateSendButton();

            event.target.value = '';
        }

        function renderPreviews() {
            const strip = document.getElementById('previewStrip');
            strip.innerHTML = '';

            if (!selectedImageFile) {
                strip.classList.add('hidden');
                setImageUploadEnabled(true);
                updateScrollButtonOffset();
                return;
            }

            strip.classList.remove('hidden');
            setImageUploadEnabled(false);

            const item = document.createElement('div');
            item.className = 'preview-item';

            const img = document.createElement('img');
            img.alt = selectedImageFile.name;

            const reader = new FileReader();
            reader.onload = () => { img.src = reader.result; };
            reader.readAsDataURL(selectedImageFile);

            const removeBtn = document.createElement('div');
            removeBtn.className = 'preview-remove';
            removeBtn.innerText = '×';
            removeBtn.onclick = () => removePreview();

            item.appendChild(img);
            item.appendChild(removeBtn);
            strip.appendChild(item);
            updateScrollButtonOffset();
        }

        function removePreview() {
            selectedImageFile = null;
            renderPreviews();
            updateSendButton();
        }

        function setImageUploadEnabled(isEnabled) {
            const label = document.getElementById('imageUploadLabel');
            const input = document.getElementById('imageUpload');
            if (!label || !input) return;
            label.classList.toggle('disabled', !isEnabled);
            input.disabled = !isEnabled;
            if (!isEnabled) {
                label.setAttribute('aria-disabled', 'true');
            } else {
                label.removeAttribute('aria-disabled');
            }
        }

        function resizeChatInput(input) {
            if (!input) return;
            const style = window.getComputedStyle(input);
            const maxHeight = parseFloat(style.maxHeight) || 140;
            const minHeight = parseFloat(style.minHeight) || 0;
            const padding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
            const border = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
            const isBorderBox = style.boxSizing === 'border-box';

            if (input.value.trim().length === 0) {
                input.style.height = '';
                updateScrollButtonOffset();
                return;
            }

            input.style.height = 'auto';
            let nextHeight = input.scrollHeight;

            if (isBorderBox) {
                nextHeight += border;
            } else {
                nextHeight -= padding;
            }

            nextHeight = Math.max(minHeight, Math.min(nextHeight, maxHeight));
            input.style.height = `${nextHeight}px`;
            updateScrollButtonOffset();
        }

        function handleInputChange(event) {
            updateSendButton();
            resizeChatInput(event.target);
        }

        function handleInputKeyDown(event) {
            if (event.key !== 'Enter') return;
            if (event.isComposing) return;
            if (event.shiftKey) return;
            event.preventDefault();
            sendChat();
        }

        function updateSendButton() {
            const msgInput = document.getElementById('userMsg');
            const sendBtn = document.getElementById('sendBtn');
            if (!msgInput || !sendBtn) return;
            const msg = msgInput.value;
            sendBtn.disabled = isSending || (msg.trim().length === 0);
        }

        function setSendingState(isActive) {
            isSending = isActive;
            updateSendButton();
        }

        // ====== (추가) assistant placeholder / update ======
        function appendAssistantPlaceholder(text) {
            const content = document.getElementById('chatContent');
            const box = document.getElementById('chatBox');
            if (!content || !box) return null;
            const shouldScroll = shouldAutoScroll(box);
            const wrapper = document.createElement('div');
            const stack = document.createElement('div');
            const div = document.createElement('div');
            const id = `a_${Date.now()}_${Math.random().toString(16).slice(2)}`;
            div.id = id;
            div.className = `msg assistant`;
            renderMessageInto(div, text);
            wrapper.className = 'msg-wrap assistant';
            stack.className = 'msg-stack';
            stack.appendChild(div);
            wrapper.appendChild(stack);
            appendTextActions(div);
            content.appendChild(wrapper);
            if (shouldScroll) {
                scrollToBottom(box);
                updateScrollButtonState();
            } else {
                markUnreadMessage();
            }
            return id;
        }

        function clearStreamTimer(id) {
            const timers = streamAssistantText._timers;
            if (timers && timers.has(id)) {
                clearInterval(timers.get(id));
                timers.delete(id);
            }
        }

        function updateAssistantPlaceholder(id, text, imageUrls, options = {}) {
            const div = document.getElementById(id);
            if (!div) return;
            const box = document.getElementById('chatBox');
            const shouldScroll = shouldAutoScroll(box);
            const { showCursor = false } = options || {};
            clearStreamTimer(id);
            stopProgress(id);
            renderMessageInto(div, text, { showCursor });
            const images = appendMessageImages(div, imageUrls);
            appendTextActions(div);
            if (images.length) {
                showImageGuide(div);
            }
            if (shouldScroll) {
                scrollToBottom(box);
                updateScrollButtonState();
            } else {
                markUnreadMessage();
            }
        }

        async function streamAssistantText(id, text, { speed = 28, chunkSize = 1 } = {}) {
            const div = document.getElementById(id);
            const box = document.getElementById('chatBox');
            if (!div || !box) return;
            stopProgress(id);
            clearMessageActions(div);
            if (!text) {
                updateAssistantPlaceholder(id, "");
                return;
            }

            if (!streamAssistantText._timers) {
                streamAssistantText._timers = new Map();
            }
            const timers = streamAssistantText._timers;
            const prevTimer = timers.get(id);
            if (prevTimer) clearInterval(prevTimer);

            let index = 0;
            await new Promise((resolve) => {
                const timer = setInterval(() => {
                    if (!document.getElementById(id)) {
                        clearInterval(timer);
                        timers.delete(id);
                        resolve();
                        return;
                    }
                    const shouldScroll = shouldAutoScroll(box);
                    index = Math.min(text.length, index + chunkSize);
                    renderMessageInto(div, text.slice(0, index), {
                        showCursor: index < text.length,
                    });
                    if (shouldScroll) {
                        scrollToBottom(box);
                        updateScrollButtonState();
                    } else {
                        markUnreadMessage();
                    }
                    if (index >= text.length) {
                        clearInterval(timer);
                        timers.delete(id);
                        appendTextActions(div);
                        resolve();
                    }
                }, speed);
                timers.set(id, timer);
            });
        }

        function renderMessageContent(text) {
            if (!text) return "";
            if (window.marked && window.DOMPurify) {
                const rawHtml = marked.parse(text, { breaks: true });
                return DOMPurify.sanitize(rawHtml);
            }
            return `<p style="margin:0;">${escapeHtml(text)}</p>`;
        }

        function renderMessageInto(container, text, { showCursor = false } = {}) {
            if (!container) return;
            clearMessageProgress(container);
            setMessageRawText(container, text);
            const content = getMessageContentContainer(container);
            content.innerHTML = renderMessageContent(text);
            if (!showCursor) return;
            const cursor = document.createElement('span');
            cursor.className = 'typing-cursor';
            const walker = document.createTreeWalker(
                content,
                NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
                {
                    acceptNode: (node) => {
                        if (node.nodeType === Node.TEXT_NODE) {
                            return node.nodeValue && node.nodeValue.trim()
                                ? NodeFilter.FILTER_ACCEPT
                                : NodeFilter.FILTER_SKIP;
                        }
                        if (node.nodeType === Node.ELEMENT_NODE && node.tagName === "BR") {
                            return NodeFilter.FILTER_ACCEPT;
                        }
                        return NodeFilter.FILTER_SKIP;
                    },
                }
            );

            let lastNode = null;
            while (walker.nextNode()) {
                lastNode = walker.currentNode;
            }

            if (!lastNode) {
                content.appendChild(cursor);
                return;
            }

            if (lastNode.nodeType === Node.TEXT_NODE) {
                const parent = lastNode.parentNode || content;
                parent.insertBefore(cursor, lastNode.nextSibling);
                return;
            }

            if (lastNode.nodeType === Node.ELEMENT_NODE && lastNode.tagName === "BR") {
                const parent = lastNode.parentNode || content;
                parent.insertBefore(cursor, lastNode);
                return;
            }

            content.appendChild(cursor);
        }

        // ====== (추가) HTML 이스케이프 ======
        function escapeHtml(str) {
            return String(str ?? "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#039;");
        }

        // ====== 진행 상태 UI ======
        const progressState = new Map();
        const analysisStepsGeneric = [
            "메시지를 분석중입니다.",
            "요청을 확인중입니다.",
            "생성 옵션을 정리중입니다.",
        ];
        const analysisStepsImage = [
            "메시지를 분석중입니다.",
            "광고 생성 타입을 확정중입니다.",
            "이미지 비율을 결정중입니다.",
            "업종을 추출중입니다.",
            "스타일을 고르는 중입니다.",
        ];
        const analysisStepsText = [
            "메시지를 분석중입니다.",
            "광고 생성 타입을 확정중입니다.",
            "핵심 키워드를 정리중입니다.",
            "응답 톤을 설정중입니다.",
            "문장 길이를 계산중입니다.",
        ];
        const imageProgressSteps = [
            { percent: 0, message: "이미지 프롬프트를 생성중입니다." },
            { percent: 70, message: "광고 구성요소를 정리중입니다." },
            { percent: 92, message: "이미지를 생성중입니다." },
            { percent: 96, message: "이미지를 저장중입니다." },
            { percent: 98, message: "이미지를 로드중입니다." },
        ];
        const textProgressSteps = [
            { percent: 18, message: "응답 톤을 설정중입니다." },
            { percent: 38, message: "핵심 문구를 구성중입니다." },
            { percent: 62, message: "문장을 다듬고 있습니다." },
            { percent: 82, message: "최종 문구를 생성중입니다." },
        ];

        function getAnalysisSteps(generationType) {
            if (generationType === "text") return analysisStepsText;
            if (generationType === "image") return analysisStepsImage;
            return analysisStepsGeneric;
        }

        function renderProgressMarkup({ message, percent = 0, showSpinner = false, showBar = false } = {}) {
            const safeMessage = escapeHtml(message || "");
            const spinner = showSpinner ? `<div class="progress-spinner"></div>` : "";
            const bar = showBar ? `
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${Math.max(0, Math.min(100, percent))}%;"></div>
                </div>
                <div class="progress-meta">
                    <span>진행률</span>
                    <span>${Math.round(Math.max(0, Math.min(100, percent)))}%</span>
                </div>
            ` : "";

            return `
                <div class="assistant-progress">
                    <div class="progress-header">
                        <div class="progress-message">${safeMessage}</div>
                        ${spinner}
                    </div>
                    ${bar}
                </div>
            `;
        }

        function setProgressView(id, { message, percent = 0, showSpinner = false, showBar = false } = {}) {
            const div = document.getElementById(id);
            if (!div) return;
            const box = document.getElementById('chatBox');
            const shouldScroll = shouldAutoScroll(box);
            clearStreamTimer(id);
            div.innerHTML = renderProgressMarkup({ message, percent, showSpinner, showBar });
            if (shouldScroll) scrollToBottom(box);
        }

        function stopProgress(id) {
            const state = progressState.get(id);
            if (!state) return;
            if (state.timer) clearInterval(state.timer);
            progressState.delete(id);
        }

        function startAnalysisIndicator(id, { generationType, message } = {}) {
            if (!id) return;
            stopProgress(id);
            const steps = getAnalysisSteps(generationType);
            const initialMessage = message || steps[0] || "분석 중입니다.";
            let initialIndex = steps.indexOf(initialMessage);
            if (initialIndex < 0) initialIndex = -1;
            const state = {
                mode: "analysis",
                steps,
                stepIndex: initialIndex,
                currentMessage: initialMessage,
                timer: null,
            };

            setProgressView(id, { message: initialMessage, showSpinner: true, showBar: false });

            if (steps.length > 0) {
                state.timer = setInterval(() => {
                    if (state.stepIndex >= state.steps.length - 1) {
                        clearInterval(state.timer);
                        state.timer = null;
                        return;
                    }
                    state.stepIndex += 1;
                    state.currentMessage = state.steps[state.stepIndex];
                    setProgressView(id, { message: state.currentMessage, showSpinner: true, showBar: false });
                }, 1800);
            }

            progressState.set(id, state);
        }

        function updateAnalysisIndicator(id, generationType) {
            const state = progressState.get(id);
            if (!state || state.mode !== "analysis") return;
            const newSteps = getAnalysisSteps(generationType);
            if (!newSteps.length) return;
            const currentMessage = state.currentMessage;
            let newIndex = newSteps.indexOf(currentMessage);
            if (newIndex < 0) {
                if (state.stepIndex >= 0) {
                    newIndex = Math.min(state.stepIndex, newSteps.length - 1);
                    state.currentMessage = newSteps[newIndex];
                    setProgressView(id, { message: state.currentMessage, showSpinner: true, showBar: false });
                } else {
                    newIndex = -1;
                }
            }
            state.steps = newSteps;
            state.stepIndex = newIndex;
        }

        function startGenerationProgress(id, { generationType } = {}) {
            if (!id) return;
            stopProgress(id);
            const isText = generationType === "text";
            const steps = isText ? textProgressSteps : imageProgressSteps;
            const state = {
                mode: "generation",
                percent: 0,
                steps,
                stepIndex: 0,
                maxPercent: isText ? 92 : 98,
                timer: null,
            };

            setProgressView(id, {
                message: steps[0]?.message || "생성 중입니다.",
                percent: state.percent,
                showSpinner: false,
                showBar: true,
            });

            state.timer = setInterval(() => {
                let increment;
                if (isText) {
                    increment = Math.random() < 0.7 ? 2 : 3;
                } else if (state.percent < 70) {
                    increment = 4;
                } else if (state.percent < 90) {
                    increment = 2;
                } else {
                    increment = 1;
                }
                state.percent = Math.min(state.percent + increment, state.maxPercent);
                while (state.stepIndex < steps.length - 1
                    && state.percent >= steps[state.stepIndex + 1].percent) {
                    state.stepIndex += 1;
                }
                const stepMessage = steps[state.stepIndex]?.message || "생성 중입니다.";
                setProgressView(id, {
                    message: stepMessage,
                    percent: state.percent,
                    showSpinner: false,
                    showBar: true,
                });
            }, 700);

            progressState.set(id, state);
        }

        function completeProgress(id, { message, delayMs = 250, onComplete } = {}) {
            const state = progressState.get(id);
            if (!state) {
                if (typeof onComplete === "function") onComplete();
                return;
            }
            if (state.timer) clearInterval(state.timer);
            setProgressView(id, {
                message: message || "완료되었습니다.",
                percent: 100,
                showSpinner: false,
                showBar: state.mode === "generation",
            });
            const finalize = () => {
                stopProgress(id);
                if (typeof onComplete === "function") onComplete();
            };
            if (delayMs > 0) {
                setTimeout(finalize, delayMs);
            } else {
                finalize();
            }
        }


        // ====== sendChat ======
        async function sendChat() {
            const input = document.getElementById('userMsg');
            const msg = input.value;

            if (msg.trim().length === 0) return;
            if (isSending) return;

            setSendingState(true);
            let loadingId = null;

            try {
                const previewUrls = selectedImageFile ? [await fileToDataUrl(selectedImageFile)] : [];

                appendMsg('user', msg, previewUrls);

                input.value = '';
                resizeChatInput(input);
                updateSendButton();

                const form = new FormData();
                if (currentSessionId) {
                    form.append("session_id", currentSessionId);
                }
                form.append("message", msg);

                if (selectedImageFile) {
                    form.append("image", selectedImageFile);
                }

                selectedImageFile = null;
                renderPreviews();

                loadingId = appendAssistantPlaceholder("");
                if (loadingId) startAnalysisIndicator(loadingId);

                await sendChatStream(form, loadingId);
            } catch (e) {
                console.error(e);
                if (loadingId) {
                    stopProgress(loadingId);
                    await streamAssistantText(loadingId, "요청 처리 중 오류가 발생했습니다.");
                }
            } finally {
                setSendingState(false);
            }
        }

        async function sendChatStream(form, loadingId) {
            if (!loadingId) return;

            const res = await fetch(`${API_BASE}/chat/message/stream`, {
                method: 'POST',
                body: form,
                credentials: "include",
            });

            if (!res.ok || !res.body) {
                const err = await safeJson(res);
                throw new Error(err.detail || "stream failed");
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let assistantText = "";
            let hasChunks = false;
            let streamMeta = {
                intent: null,
                generationType: null,
                generationFallbackTimer: null,
            };

            const clearGenerationFallback = () => {
                if (!streamMeta.generationFallbackTimer) return;
                clearTimeout(streamMeta.generationFallbackTimer);
                streamMeta.generationFallbackTimer = null;
            };

            const applySessionId = (sessionId) => {
                if (!sessionId) return;
                currentSessionId = sessionId;
                localStorage.setItem("session_id", sessionId);
            };

            const handleStreamProgress = (payload) => {
                const stage = payload.stage || "";
                const generationType = payload.generation_type || streamMeta.generationType;
                if (stage === "analyzing") {
                    const currentState = progressState.get(loadingId);
                    if (currentState && currentState.mode === "analysis") {
                        if (payload.message && payload.message !== currentState.currentMessage) {
                            currentState.currentMessage = payload.message;
                            setProgressView(loadingId, {
                                message: currentState.currentMessage,
                                showSpinner: true,
                                showBar: false,
                            });
                        }
                        updateAnalysisIndicator(loadingId, generationType);
                    } else {
                        startAnalysisIndicator(loadingId, { generationType, message: payload.message });
                    }
                    return;
                }
                if (stage === "preloading") {
                    clearGenerationFallback();
                    stopProgress(loadingId);
                    setProgressView(loadingId, {
                        message: payload.message || "모델을 로드 중입니다.",
                        showSpinner: true,
                        showBar: false,
                    });
                    return;
                }
                if (stage === "generation_update") {
                    clearGenerationFallback();
                    let state = progressState.get(loadingId);
                    if (!state || state.mode !== "generation") {
                        startGenerationProgress(loadingId, { generationType });
                        state = progressState.get(loadingId);
                    }
                    if (state && state.mode === "generation") {
                        const nextPercent = Number.isFinite(payload.percent) ? payload.percent : state.percent;
                        const maxPercent = Number.isFinite(state.maxPercent) ? state.maxPercent : 100;
                        state.percent = Math.min(Math.max(state.percent, nextPercent), maxPercent);
                        while (state.stepIndex < state.steps.length - 1
                            && state.percent >= state.steps[state.stepIndex + 1].percent) {
                            state.stepIndex += 1;
                        }
                        const stepMessage = payload.message || state.steps[state.stepIndex]?.message || "생성 중입니다.";
                        setProgressView(loadingId, {
                            message: stepMessage,
                            percent: state.percent,
                            showSpinner: false,
                            showBar: true,
                        });
                    }
                    return;
                }
                if (stage === "generating") {
                    clearGenerationFallback();
                    startGenerationProgress(loadingId, { generationType });
                    return;
                }
                if (!stage) {
                    const message = payload.message || "";
                    if (message.includes("생성") || message.includes("수정")) {
                        clearGenerationFallback();
                        startGenerationProgress(loadingId, { generationType });
                        return;
                    }
                }
                setProgressView(loadingId, {
                    message: payload.message || "처리 중입니다...",
                    showSpinner: true,
                    showBar: false,
                });
            };

            const handlePayload = (payload) => {
                if (!payload) return;
                if (payload.session_id) applySessionId(payload.session_id);

                if (payload.type === "meta") {
                    if (payload.intent) streamMeta.intent = payload.intent;
                    if (payload.generation_type) streamMeta.generationType = payload.generation_type;
                    updateAnalysisIndicator(loadingId, streamMeta.generationType);
                    clearGenerationFallback();
                    if (streamMeta.intent && streamMeta.intent !== "consulting") {
                        streamMeta.generationFallbackTimer = setTimeout(() => {
                            const state = progressState.get(loadingId);
                            if (state && state.mode === "analysis") {
                                startGenerationProgress(loadingId, { generationType: streamMeta.generationType });
                            }
                        }, 1200);
                    }
                    return;
                }

                if (payload.type === "chunk") {
                    assistantText += payload.content || "";
                    hasChunks = true;
                    updateAssistantPlaceholder(loadingId, assistantText, null, { showCursor: true });
                    return;
                }

                if (payload.type === "progress") {
                    handleStreamProgress(payload);
                    return;
                }

                if (payload.type === "done") {
                    clearGenerationFallback();
                    const output = payload.output || null;
                    const outputText = payload.assistant_message
                        || (output ? output.output_text : "")
                        || assistantText;
                    const imageUrls = normalizeImageUrls(output ? output.image : null);

                    const finalizeOutput = () => {
                        if (outputText) {
                            if (hasChunks) {
                                updateAssistantPlaceholder(loadingId, outputText, imageUrls);
                            } else {
                                streamAssistantText(loadingId, outputText);
                                updateAssistantPlaceholder(loadingId, outputText, imageUrls);
                            }
                        } else if (imageUrls.length) {
                            updateAssistantPlaceholder(loadingId, "", imageUrls);
                        }
                    };

                    if (output) {
                        const contentType = output.content_type || streamMeta.generationType;
                        const doneMessage = contentType === "text"
                            ? "텍스트 생성 완료!"
                            : "이미지 생성 완료!";
                        completeProgress(loadingId, {
                            message: doneMessage,
                            delayMs: 250,
                            onComplete: finalizeOutput,
                        });
                    } else {
                        stopProgress(loadingId);
                        finalizeOutput();
                    }
                    return;
                }

                if (payload.type === "error") {
                    clearGenerationFallback();
                    stopProgress(loadingId);
                    streamAssistantText(loadingId, payload.message || "요청 실패");
                }
            };

            const handleEventBlock = (block) => {
                const lines = block.split("\n");
                const dataLines = lines.filter((line) => line.startsWith("data:"));
                if (dataLines.length === 0) return;
                const dataText = dataLines.map((line) => line.replace(/^data:\s?/, "")).join("");
                try {
                    handlePayload(JSON.parse(dataText));
                } catch (e) {
                    console.error(e);
                }
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split("\n\n");
                buffer = parts.pop();
                parts.forEach(handleEventBlock);
            }

            if (buffer.trim()) {
                handleEventBlock(buffer);
            }
        }

        function fileToDataUrl(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }

        function appendMsg(role, text, imageUrls, messageId) {
            const content = document.getElementById('chatContent');
            const box = document.getElementById('chatBox');
            if (!content || !box) return;

            const shouldScroll = shouldAutoScroll(box);
            const wrapper = createMessageElement(role, text, imageUrls, messageId);
            content.appendChild(wrapper);
            if (shouldScroll) {
                scrollToBottom(box);
                updateScrollButtonState();
            } else {
                if (role === 'assistant') {
                    markUnreadMessage();
                } else {
                    updateScrollButtonState();
                }
            }
        }



        // ====== 이미지 모달 제어 ======
        function openImageModal(imageUrl) {
            const modal = document.getElementById('image-modal');
            const modalImg = document.getElementById('modal-image');
            modalImg.src = imageUrl;
            modal.classList.remove('hidden');
        }

        function closeImageModal() {
            const modal = document.getElementById('image-modal');
            modal.classList.add('hidden');
        }
    
