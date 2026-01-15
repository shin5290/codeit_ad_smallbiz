"""
Instagram feed crawler (small business / self-employed) for RAG dataset.

Collects per post:
- caption text
- hashtags
- like_count
- comment_count
- post_url
- author username

Does NOT collect:
- share count / save count (not reliably available via public interfaces)

Excludes franchises:
- blacklist of known franchise brand keywords (Korean/English)
- username blacklist (optional)
- heuristic checks on bio/username

Adds:
- per-user cap (default 2 posts per username) to reduce bias
- location default: 전국 (nationwide), filter later in RAG via caption/hashtags

Install:
  pip install instaloader

Single mode:
  python instagram_self_employed_crawler.py --hashtags "강남카페,부산맛집,네일아트,데일리룩" --limit 100 --min-likes 30 --out ./out

Multi-job mode:
  python instagram_self_employed_crawler.py --config ./config.json --limit 100 --min-likes 30 --out ./out

Login (recommended):
  python instagram_self_employed_crawler.py --login-user "YOUR_ID" --login-pass "YOUR_PASS" --hashtags "카페투어,맛집추천" --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Set

import instaloader
from instagrapi import Client as InstaClient
import instagrapi.extractors as ig_extractors
import logging


# ----------------------------
# Config: franchise exclusion
# ----------------------------
FRANCHISE_KEYWORDS = {
    # big cafe chains / food chains
    "스타벅스", "starbucks",
    "빽다방", "paik", "paik's", "paiks",
    "이디야", "ediya",
    "투썸", "twosome",
    "메가커피", "mega coffee",
    "컴포즈", "compose",
    "할리스", "hollys",
    "폴바셋", "paul bassett",
    "탐앤탐스", "tom n toms",
    "카페베네", "caffebene",
    "공차", "gong cha",
    "노티드", "knotted",
    "던킨", "dunkin",
    "파리바게뜨", "paris baguette",
    "뚜레쥬르", "tous les jours",
    "배스킨", "baskin",
    "크리스피", "krispy",
    "도미노", "domino",
    "버거킹", "burger king",
    "맥도날드", "mcdonald",
    "kfc",
    "서브웨이", "subway",
    "롯데리아", "lotteria",
    "bbq", "bhc",
    "교촌", "kyochon",
    "본죽", "bonjuk",
    "맘스터치", "mom's touch",
    "미스터피자", "mrpizza",

    # generic franchise terms
    "가맹", "가맹점", "가맹문의",
    "프랜차이즈", "franchise",
    "official", "head office", "본사",
}

FRANCHISE_USERNAMES = {
    "starbucks",
    "starbuckskorea",
    "paikscoffee",
    "ediya.coffee",
    "mega_coffee_official",
}

EXCLUDE_KEYWORDS_DEFAULT = {
    # engagement/follower sales spam
    "팔로워", "팔로워구매", "팔로워늘리기",
    "좋아요", "좋아요구매", "좋아요늘리기", "좋아요테러",
    "맞팔", "선팔", "맞팔환영", "선팔하면맞팔", "팔로우", "팔로우미", "팔로우반사", "팔로잉",
    "f4f", "l4l", "follow4follow", "like4like",
    # sales/cta patterns common in spam
    "오픈채팅", "구매 문의", "구매문의", "프로필 클릭", "계정활성화",
    "최저가", "판매", "가격표",
}

EXCLUDE_HASHTAGS_DEFAULT = {
    "팔로워", "팔로워구매", "팔로워늘리기",
    "좋아요", "좋아요구매", "좋아요늘리기", "좋아요테러", "좋반",
    "맞팔", "맞팔환영", "선팔", "선팔하면맞팔", "팔로우", "팔로우미", "팔로우반사", "팔로잉",
    "f4f", "l4l", "follow4follow", "like4like",
}

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def prepare_region_keys(regions: Iterable[str]) -> list[tuple[str, str]]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for r in regions:
        if r is None:
            continue
        r = str(r).strip()
        if not r or r in seen:
            continue
        cleaned.append(r)
        seen.add(r)
    cleaned = sorted(cleaned, key=len, reverse=True)
    return [(r, normalize_text(r)) for r in cleaned]


def match_region(text: str, region_keys: list[tuple[str, str]]) -> Optional[str]:
    if not text or not region_keys:
        return None
    t = normalize_text(text)
    for region, norm in region_keys:
        if norm and norm in t:
            return region
    return None


def choose_location(
    default_location: str,
    region_keys: list[tuple[str, str]],
    caption: str,
    hashtags: list[str],
    location_name: str,
) -> str:
    fallback = (default_location or "").strip() or "전국"
    loc = (location_name or "").strip()
    if loc:
        return match_region(loc, region_keys) or loc
    text = " ".join([caption or "", " ".join(hashtags or [])]).strip()
    return match_region(text, region_keys) or fallback


def prepare_exclude_keywords(items: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for k in items:
        if k is None:
            continue
        k = normalize_text(str(k))
        if not k or k in seen:
            continue
        cleaned.append(k)
        seen.add(k)
    return cleaned


def prepare_exclude_hashtags(items: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for h in items:
        if h is None:
            continue
        h = normalize_text(str(h)).lstrip("#")
        if not h or h in seen:
            continue
        cleaned.append(h)
        seen.add(h)
    return cleaned


def should_exclude_post(
    caption: str,
    hashtags: list[str],
    exclude_keywords: list[str],
    exclude_hashtags: list[str],
) -> bool:
    text = normalize_text(caption)
    if any(kw in text for kw in exclude_keywords):
        return True

    for h in hashtags:
        hn = normalize_text(h)
        if any(ex in hn for ex in exclude_hashtags):
            return True
        if any(kw in hn for kw in exclude_keywords):
            return True
    return False


def looks_like_franchise(username: str, bio: str) -> bool:
    u = normalize_text(username)
    b = normalize_text(bio)

    if u in {normalize_text(x) for x in FRANCHISE_USERNAMES}:
        return True

    combined = f"{u} {b}"
    for kw in FRANCHISE_KEYWORDS:
        if normalize_text(kw) in combined:
            return True

    # Weak heuristic: branch markers
    # (Warning: can false-positive; keep if you really want aggressive filtering)
    branch_patterns = [r"\d+\s*호점", r"\d+\s*호", r"지점", r"\bbranch\b"]
    if any(re.search(pat, combined) for pat in branch_patterns):
        return True

    return False


# ----------------------------
# Data models
# ----------------------------
@dataclass
class InstaContent:
    username: str
    caption: str
    hashtags: list[str]
    post_url: str
    post_date: str  # ISO8601


@dataclass
class InstaPerformance:
    like_count: int
    comment_count: int
    engagement: int


@dataclass
class InstaMetadata:
    crawl_date: str
    source: str
    seed: str


@dataclass
class InstaItem:
    id: str
    platform: str
    industry: str
    location: str
    content: InstaContent
    performance: InstaPerformance
    metadata: InstaMetadata


# ----------------------------
# Crawler
# ----------------------------
class InstaSelfEmployedCrawler:
    def __init__(
        self,
        login_user: Optional[str] = None,
        login_pass: Optional[str] = None,
        session_dir: str = ".insta_session",
        session_file: Optional[str] = None,
        iphone_support: bool = False,
        search_fallback: bool = False,
        search_profile_limit: int = 30,
        search_posts_per_profile: int = 6,
        regions: Optional[list[str]] = None,
        exclude_keywords: Optional[list[str]] = None,
        exclude_hashtags: Optional[list[str]] = None,
        quiet: bool = False,
    ):
        self.quiet = quiet
        self.L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=quiet,
            iphone_support=iphone_support,
        )
        self.L.context.iphone_support = iphone_support

        if session_dir:
            os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.session_file = session_file
        self.region_keys = prepare_region_keys(regions or [])
        self.exclude_keywords = prepare_exclude_keywords(exclude_keywords or [])
        self.exclude_hashtags = prepare_exclude_hashtags(exclude_hashtags or [])

        self.search_fallback = search_fallback
        self.search_profile_limit = max(0, int(search_profile_limit or 0))
        self.search_posts_per_profile = max(0, int(search_posts_per_profile or 0))

        if login_user:
            self._try_login(login_user, login_pass)

    def _log(self, msg: str):
        if not self.quiet:
            print(msg)

    def _try_login(self, user: str, pw: Optional[str]):
        session_candidates: list[tuple[str, Optional[str]]] = []
        if self.session_file:
            session_candidates.append(("explicit session file", self.session_file))
        if self.session_dir:
            session_candidates.append(("session dir", os.path.join(self.session_dir, f"session-{user}")))
        session_candidates.append(("instaloader default", None))

        for label, path in session_candidates:
            try:
                if path:
                    self.L.load_session_from_file(user, filename=path)
                else:
                    self.L.load_session_from_file(user)
                self._log(f"[+] Loaded session for {user} ({label})")
                return
            except Exception:
                continue

        if not pw:
            self._log("[!] No session found and no password provided. Continuing without login (less stable).")
            return

        try:
            self.L.login(user, pw)
            self.L.save_session_to_file(filename=session_path)
            self._log(f"[+] Logged in and saved session for {user}")
        except Exception as e:
            self._log(f"[!] Login failed ({type(e).__name__}): {e}. Continuing without login.")

    @staticmethod
    def polite_sleep(base: float = 2.0, jitter: float = 2.0):
        time.sleep(base + random.random() * jitter)

    def iter_posts_from_hashtag(self, hashtag: str) -> Iterable[instaloader.Post]:
        try:
            h = instaloader.Hashtag.from_name(self.L.context, hashtag)
        except Exception as e:
            self._log(f"[!] Hashtag lookup failed for #{hashtag}: {type(e).__name__}: {e}")
            if self.search_fallback:
                self._log(f"[~] Falling back to profile search for '{hashtag}'")
                yield from self.iter_posts_from_search(
                    query=hashtag,
                    max_profiles=self.search_profile_limit,
                    max_posts_per_profile=self.search_posts_per_profile,
                )
            return

        # Prefer top posts first (reduces scanning volume)
        try:
            for p in h.get_top_posts():
                yield p
        except Exception:
            pass

        for p in h.get_posts():
            yield p

    def build_item(
        self,
        post: instaloader.Post,
        industry: str,
        location: str,
        seed: str,
    ) -> Optional[InstaItem]:
        try:
            prof = post.owner_profile
        except Exception:
            return None

        if prof is None:
            return None

        try:
            if prof.is_private:
                return None
        except Exception:
            return None

        username = getattr(prof, "username", "") or ""
        bio = getattr(prof, "biography", "") or ""

        if looks_like_franchise(username=username, bio=bio):
            return None

        caption = post.caption or ""
        hashtags = sorted({h for h in re.findall(r"#([0-9A-Za-z_가-힣]+)", caption)})

        if should_exclude_post(
            caption=caption,
            hashtags=hashtags,
            exclude_keywords=self.exclude_keywords,
            exclude_hashtags=self.exclude_hashtags,
        ):
            return None

        loc_name = ""
        try:
            loc = getattr(post, "location", None)
            if loc:
                loc_name = getattr(loc, "name", "") or ""
        except Exception:
            loc_name = ""

        like_count = int(getattr(post, "likes", 0) or 0)
        comment_count = int(getattr(post, "comments", 0) or 0)

        shortcode = post.shortcode
        post_url = f"https://www.instagram.com/p/{shortcode}/"
        post_date = post.date_utc.replace(tzinfo=timezone.utc).isoformat()

        return InstaItem(
            id=f"insta_p_{shortcode}",
            platform="instagram_feed",
            industry=industry,
            location=choose_location(
                default_location=location,
                region_keys=self.region_keys,
                caption=caption,
                hashtags=hashtags,
                location_name=loc_name,
            ),
            content=InstaContent(
                username=username,
                caption=caption,
                hashtags=hashtags,
                post_url=post_url,
                post_date=post_date,
            ),
            performance=InstaPerformance(
                like_count=like_count,
                comment_count=comment_count,
                engagement=like_count + comment_count,
            ),
            metadata=InstaMetadata(
                crawl_date=datetime.now(timezone.utc).isoformat(),
                source="instaloader",
                seed=seed,
            ),
        )

    def iter_posts_from_search(
        self,
        query: str,
        max_profiles: int,
        max_posts_per_profile: int,
    ) -> Iterable[instaloader.Post]:
        try:
            results = instaloader.TopSearchResults(self.L.context, query)
        except Exception as e:
            self._log(f"[!] Search error for '{query}': {type(e).__name__}: {e}")
            return

        collected_profiles = 0
        for prof in results.get_profiles():
            if max_profiles and collected_profiles >= max_profiles:
                break

            try:
                if prof.is_private:
                    continue
            except Exception:
                continue

            username = getattr(prof, "username", "") or ""
            bio = getattr(prof, "biography", "") or ""
            if looks_like_franchise(username=username, bio=bio):
                continue

            collected_profiles += 1
            yielded = 0
            try:
                for p in prof.get_posts():
                    yield p
                    yielded += 1
                    if max_posts_per_profile and yielded >= max_posts_per_profile:
                        break
            except Exception as e:
                self._log(f"[!] Profile posts error for {username}: {type(e).__name__}: {e}")
                continue

    def crawl(
        self,
        hashtags: list[str],
        limit: int,
        industry: str,
        location: str,
        min_likes: int = 0,
        min_comments: int = 0,
        max_posts_scanned: int = 5000,
        max_posts_per_user: int = 2,
    ) -> list[InstaItem]:
        results: list[InstaItem] = []
        seen: Set[str] = set()
        per_user_count: dict[str, int] = {}
        scanned = 0

        tag_iters = [(tag, iter(self.iter_posts_from_hashtag(tag))) for tag in hashtags]

        while len(results) < limit and scanned < max_posts_scanned and tag_iters:
            for idx in range(len(tag_iters) - 1, -1, -1):
                tag, it = tag_iters[idx]
                if len(results) >= limit or scanned >= max_posts_scanned:
                    break

                try:
                    post = next(it)
                except StopIteration:
                    tag_iters.pop(idx)
                    continue
                except Exception as e:
                    self._log(f"[!] Hashtag iterator error for #{tag}: {type(e).__name__}: {e}")
                    tag_iters.pop(idx)
                    continue

                scanned += 1

                shortcode = getattr(post, "shortcode", None)
                if not shortcode or shortcode in seen:
                    continue

                likes = int(getattr(post, "likes", 0) or 0)
                comments = int(getattr(post, "comments", 0) or 0)
                if likes < min_likes or comments < min_comments:
                    continue

                # Per-user cap (reduce bias)
                uname = ""
                if max_posts_per_user and max_posts_per_user > 0:
                    try:
                        owner = post.owner_profile
                        uname = getattr(owner, "username", "") or ""
                    except Exception:
                        uname = ""

                    if uname and per_user_count.get(uname, 0) >= max_posts_per_user:
                        continue

                try:
                    item = self.build_item(post, industry=industry, location=location, seed=f"hashtag:{tag}")
                except Exception:
                    item = None

                if not item:
                    continue

                seen.add(shortcode)
                results.append(item)
                if uname:
                    per_user_count[uname] = per_user_count.get(uname, 0) + 1

                if len(results) % 10 == 0:
                    self._log(f"[+] Collected {len(results)}/{limit} (scanned={scanned})")

                self.polite_sleep(2.0, 2.5)

        self._log(f"[+] Done. Collected {len(results)} posts (scanned={scanned})")
        return results


# ----------------------------
# Instagrapi crawler
# ----------------------------
def patch_instagrapi_media_extractor(quiet: bool = False) -> None:
    try:
        import instagrapi.mixins.hashtag as ig_hashtag
    except Exception:
        if not quiet:
            print("[!] Could not import instagrapi hashtag mixin for patching")
        return

    orig_extract = ig_extractors.extract_media_v1

    def sanitize_image_candidate(candidate):
        if isinstance(candidate, dict):
            if candidate.get("scans_profile") is None:
                candidate = dict(candidate)
                candidate["scans_profile"] = ""
        return candidate

    def safe_extract_media_v1(data):
        if isinstance(data, dict):
            if "clips_metadata" in data:
                data = dict(data)
                data.pop("clips_metadata", None)

            iv = data.get("image_versions2")
            if isinstance(iv, dict):
                iv = dict(iv)
                candidates = iv.get("candidates")
                if isinstance(candidates, list):
                    iv["candidates"] = [sanitize_image_candidate(c) for c in candidates]
                additional = iv.get("additional_candidates")
                if isinstance(additional, dict):
                    additional = dict(additional)
                    for key in ("first_frame", "igtv_first_frame", "smart_frame"):
                        if key in additional:
                            additional[key] = sanitize_image_candidate(additional.get(key))
                    iv["additional_candidates"] = additional
                data = dict(data)
                data["image_versions2"] = iv
        return orig_extract(data)

    ig_extractors.extract_media_v1 = safe_extract_media_v1
    ig_hashtag.extract_media_v1 = safe_extract_media_v1

    if not quiet:
        print("[+] Applied instagrapi media extractor patch")


class InstaGrapiCrawler:
    def __init__(
        self,
        login_user: Optional[str] = None,
        login_pass: Optional[str] = None,
        settings_path: Optional[str] = None,
        sessionid: Optional[str] = None,
        delay_min: float = 1.5,
        delay_max: float = 3.5,
        regions: Optional[list[str]] = None,
        exclude_keywords: Optional[list[str]] = None,
        exclude_hashtags: Optional[list[str]] = None,
        quiet: bool = False,
    ):
        self.quiet = quiet
        patch_instagrapi_media_extractor(quiet=quiet)
        ig_logger = logging.getLogger("instagrapi")
        ig_logger.addHandler(logging.NullHandler())
        ig_logger.propagate = False
        ig_logger.setLevel(logging.CRITICAL)
        self.client = InstaClient(logger=ig_logger)
        self.client.logger.disabled = True
        self.client.delay_range = [delay_min, delay_max]
        self.settings_path = settings_path
        self._user_bio_cache: dict[str, str] = {}
        self.region_keys = prepare_region_keys(regions or [])
        self.exclude_keywords = prepare_exclude_keywords(exclude_keywords or [])
        self.exclude_hashtags = prepare_exclude_hashtags(exclude_hashtags or [])

        if settings_path and os.path.exists(settings_path):
            try:
                self.client.load_settings(settings_path)
                self._log(f"[+] Loaded instagrapi settings: {settings_path}")
            except Exception as e:
                self._log(f"[!] Failed to load instagrapi settings ({type(e).__name__}): {e}")

        if sessionid:
            try:
                self.client.login_by_sessionid(sessionid)
                self._log("[+] Logged in via sessionid")
            except Exception as e:
                self._log(f"[!] Sessionid login failed ({type(e).__name__}): {e}")

        if login_user and login_pass:
            try:
                self.client.login(login_user, login_pass)
                if settings_path:
                    settings_dir = os.path.dirname(settings_path)
                    if settings_dir:
                        os.makedirs(settings_dir, exist_ok=True)
                    self.client.dump_settings(settings_path)
                self._log("[+] Logged in with instagrapi and saved settings")
            except Exception as e:
                self._log(f"[!] Instagrapi login failed ({type(e).__name__}): {e}")
        elif login_user:
            self._log("[!] No password provided; relying on existing instagrapi settings/session")

    def _log(self, msg: str):
        if not self.quiet:
            print(msg)

    @staticmethod
    def polite_sleep(base: float = 2.0, jitter: float = 2.0):
        time.sleep(base + random.random() * jitter)

    def _get_user_bio(self, user_id: Optional[str], username: str) -> str:
        if username in self._user_bio_cache:
            return self._user_bio_cache[username]
        bio = ""
        try:
            # Prefer private API to avoid noisy public GraphQL failures.
            if user_id:
                info = self.client.user_info_v1(user_id)
            else:
                info = self.client.user_info_by_username_v1(username)
            bio = getattr(info, "biography", "") or ""
        except Exception:
            bio = ""
        self._user_bio_cache[username] = bio
        return bio

    def iter_posts_from_hashtag(self, hashtag: str, amount: int) -> Iterable:
        tag = hashtag.lstrip("#")
        amount = max(1, int(amount or 1))
        top_amount = min(30, amount)
        recent_amount = min(200, amount)

        try:
            for m in self.client.hashtag_medias_top(tag, amount=top_amount):
                yield m
        except Exception as e:
            self._log(f"[!] Hashtag top error for #{tag}: {type(e).__name__}: {e}")

        try:
            for m in self.client.hashtag_medias_recent(tag, amount=recent_amount):
                yield m
        except Exception as e:
            self._log(f"[!] Hashtag recent error for #{tag}: {type(e).__name__}: {e}")

    def build_item(
        self,
        media,
        industry: str,
        location: str,
        seed: str,
    ) -> Optional[InstaItem]:
        user = getattr(media, "user", None)
        if not user:
            return None

        try:
            if user.is_private:
                return None
        except Exception:
            pass

        username = getattr(user, "username", "") or ""
        user_id = getattr(user, "pk", None)
        bio = self._get_user_bio(user_id=user_id, username=username) if username else ""

        if looks_like_franchise(username=username, bio=bio):
            return None

        caption = getattr(media, "caption_text", "") or ""
        hashtags = sorted({h for h in re.findall(r"#([0-9A-Za-z_가-힣]+)", caption)})

        if should_exclude_post(
            caption=caption,
            hashtags=hashtags,
            exclude_keywords=self.exclude_keywords,
            exclude_hashtags=self.exclude_hashtags,
        ):
            return None

        loc_name = ""
        try:
            loc = getattr(media, "location", None)
            if loc:
                loc_name = getattr(loc, "name", "") or ""
        except Exception:
            loc_name = ""

        like_count = int(getattr(media, "like_count", 0) or 0)
        comment_count = int(getattr(media, "comment_count", 0) or 0)

        shortcode = getattr(media, "code", "") or ""
        if not shortcode:
            return None

        post_url = f"https://www.instagram.com/p/{shortcode}/"
        taken_at = getattr(media, "taken_at", None)
        if taken_at is None:
            return None
        if taken_at.tzinfo is None:
            post_date = taken_at.replace(tzinfo=timezone.utc).isoformat()
        else:
            post_date = taken_at.astimezone(timezone.utc).isoformat()

        return InstaItem(
            id=f"insta_p_{shortcode}",
            platform="instagram_feed",
            industry=industry,
            location=choose_location(
                default_location=location,
                region_keys=self.region_keys,
                caption=caption,
                hashtags=hashtags,
                location_name=loc_name,
            ),
            content=InstaContent(
                username=username,
                caption=caption,
                hashtags=hashtags,
                post_url=post_url,
                post_date=post_date,
            ),
            performance=InstaPerformance(
                like_count=like_count,
                comment_count=comment_count,
                engagement=like_count + comment_count,
            ),
            metadata=InstaMetadata(
                crawl_date=datetime.now(timezone.utc).isoformat(),
                source="instagrapi",
                seed=seed,
            ),
        )

    def crawl(
        self,
        hashtags: list[str],
        limit: int,
        industry: str,
        location: str,
        min_likes: int = 0,
        min_comments: int = 0,
        max_posts_scanned: int = 5000,
        max_posts_per_user: int = 2,
    ) -> list[InstaItem]:
        results: list[InstaItem] = []
        seen: Set[str] = set()
        per_user_count: dict[str, int] = {}
        scanned = 0

        per_tag_amount = min(200, max_posts_scanned)
        tag_iters = [(tag, iter(self.iter_posts_from_hashtag(tag, per_tag_amount))) for tag in hashtags]

        while len(results) < limit and scanned < max_posts_scanned and tag_iters:
            for idx in range(len(tag_iters) - 1, -1, -1):
                tag, it = tag_iters[idx]
                if len(results) >= limit or scanned >= max_posts_scanned:
                    break

                try:
                    media = next(it)
                except StopIteration:
                    tag_iters.pop(idx)
                    continue
                except Exception as e:
                    self._log(f"[!] Hashtag iterator error for #{tag}: {type(e).__name__}: {e}")
                    tag_iters.pop(idx)
                    continue

                scanned += 1

                shortcode = getattr(media, "code", None)
                if not shortcode or shortcode in seen:
                    continue

                likes = int(getattr(media, "like_count", 0) or 0)
                comments = int(getattr(media, "comment_count", 0) or 0)
                if likes < min_likes or comments < min_comments:
                    continue

                uname = ""
                if max_posts_per_user and max_posts_per_user > 0:
                    user = getattr(media, "user", None)
                    uname = getattr(user, "username", "") or ""
                    if uname and per_user_count.get(uname, 0) >= max_posts_per_user:
                        continue

                try:
                    item = self.build_item(media, industry=industry, location=location, seed=f"hashtag:{tag}")
                except Exception:
                    item = None

                if not item:
                    continue

                seen.add(shortcode)
                results.append(item)
                if uname:
                    per_user_count[uname] = per_user_count.get(uname, 0) + 1

                if len(results) % 10 == 0:
                    self._log(f"[+] Collected {len(results)}/{limit} (scanned={scanned})")

                self.polite_sleep(2.0, 2.5)

        self._log(f"[+] Done. Collected {len(results)} posts (scanned={scanned})")
        return results


# ----------------------------
# Output helpers
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(items: list[InstaItem], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def write_csv(items: list[InstaItem], path: str):
    cols = [
        "id", "platform", "industry", "location",
        "username", "post_date", "post_url",
        "like_count", "comment_count", "engagement",
        "hashtags", "caption",
        "crawl_date", "seed",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for it in items:
            w.writerow(
                {
                    "id": it.id,
                    "platform": it.platform,
                    "industry": it.industry,
                    "location": it.location,
                    "username": it.content.username,
                    "post_date": it.content.post_date,
                    "post_url": it.content.post_url,
                    "like_count": it.performance.like_count,
                    "comment_count": it.performance.comment_count,
                    "engagement": it.performance.engagement,
                    "hashtags": ",".join(it.content.hashtags),
                    "caption": it.content.caption,
                    "crawl_date": it.metadata.crawl_date,
                    "seed": it.metadata.seed,
                }
            )


# ----------------------------
# Config jobs
# ----------------------------
def load_jobs_from_config(path: str) -> tuple[list[dict], list[str], list[str], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    jobs = cfg.get("jobs")
    regions_raw = cfg.get("regions", [])
    exclude_keywords_raw = cfg.get("exclude_keywords", [])
    exclude_hashtags_raw = cfg.get("exclude_hashtags", [])
    if isinstance(regions_raw, str):
        regions = [x.strip() for x in regions_raw.split(",") if x.strip()]
    elif isinstance(regions_raw, list):
        regions = [str(x).strip() for x in regions_raw if str(x).strip()]
    else:
        regions = []
    if isinstance(exclude_keywords_raw, str):
        exclude_keywords = [x.strip() for x in exclude_keywords_raw.split(",") if x.strip()]
    elif isinstance(exclude_keywords_raw, list):
        exclude_keywords = [str(x).strip() for x in exclude_keywords_raw if str(x).strip()]
    else:
        exclude_keywords = []
    if isinstance(exclude_hashtags_raw, str):
        exclude_hashtags = [x.strip().lstrip("#") for x in exclude_hashtags_raw.split(",") if x.strip()]
    elif isinstance(exclude_hashtags_raw, list):
        exclude_hashtags = [str(x).strip().lstrip("#") for x in exclude_hashtags_raw if str(x).strip()]
    else:
        exclude_hashtags = []
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Config must contain a non-empty 'jobs' list")

    norm_jobs: list[dict] = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        industry = str(j.get("industry", "unknown")).strip() or "unknown"
        location = str(j.get("location", "전국")).strip() or "전국"
        hashtags = j.get("hashtags")
        if isinstance(hashtags, str):
            hashtags_list = [x.strip().lstrip("#") for x in hashtags.split(",") if x.strip()]
        elif isinstance(hashtags, list):
            hashtags_list = [str(x).strip().lstrip("#") for x in hashtags if str(x).strip()]
        else:
            hashtags_list = []
        limit = int(j.get("limit", 0) or 0)
        if not hashtags_list or limit <= 0:
            continue
        norm_jobs.append({"industry": industry, "location": location, "hashtags": hashtags_list, "limit": limit})

    if not norm_jobs:
        raise ValueError("No valid jobs found in config (need hashtags + limit)")
    return norm_jobs, regions, exclude_keywords, exclude_hashtags


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl Instagram feed posts (self-employed accounts) for RAG")

    # Single-job mode
    p.add_argument("--hashtags", type=str, default=None, help="Comma-separated hashtags without # (single-job mode)")
    p.add_argument("--limit", type=int, default=100, help="Number of posts to collect (single-job mode) or total across jobs")

    p.add_argument("--industry", type=str, default="mixed", help="Industry label to store in schema (single-job mode)")
    p.add_argument("--location", type=str, default="전국", help="Location label to store in schema (single-job mode). Recommend: 전국")

    # Multi-job mode via JSON config
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help='Path to JSON config: {"jobs":[{"industry":"food","location":"전국","hashtags":["맛집추천"],"limit":40}, ...]}',
    )

    p.add_argument("--min-likes", type=int, default=0, help="Minimum likes to keep")
    p.add_argument("--min-comments", type=int, default=0, help="Minimum comments to keep")

    p.add_argument("--max-posts-per-user", type=int, default=2, help="Max collected posts per username (default: 2)")
    p.add_argument("--regions", type=str, default=None, help="Comma-separated region keywords for location matching")

    p.add_argument(
        "--engine",
        type=str,
        default="instagrapi",
        choices=["instagrapi", "instaloader"],
        help="Crawler engine (default: instagrapi)",
    )

    p.add_argument("--login-user", type=str, default=None, help="Instagram login username (optional, recommended)")
    p.add_argument("--login-pass", type=str, default=None, help="Instagram login password (optional)")
    p.add_argument("--session-dir", type=str, default=".insta_session", help="Where to save instaloader sessions")
    p.add_argument("--session-file", type=str, default=None, help="Load a specific instaloader session file path")
    p.add_argument("--iphone-support", action="store_true", help="Enable iPhone API support for instaloader")
    p.add_argument("--search-fallback", action="store_true", help="Fallback to profile search when hashtag fails")
    p.add_argument("--search-profile-limit", type=int, default=30, help="Max profiles per search query")
    p.add_argument("--search-posts-per-profile", type=int, default=6, help="Max posts per profile during search fallback")
    p.add_argument("--ig-settings", type=str, default=None, help="Path to save/load instagrapi settings JSON")
    p.add_argument("--ig-sessionid", type=str, default=None, help="Instagram sessionid cookie (instagrapi)")

    p.add_argument("--out", type=str, default="./out", help="Output directory")
    p.add_argument("--quiet", action="store_true", help="Less console output")
    p.add_argument("--max-posts-scanned", type=int, default=5000, help="Safety cap to limit scanning")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.out)

    jobs: list[dict] = []
    region_list: list[str] = []
    exclude_keywords: list[str] = list(EXCLUDE_KEYWORDS_DEFAULT)
    exclude_hashtags: list[str] = list(EXCLUDE_HASHTAGS_DEFAULT)
    if args.config:
        try:
            jobs, region_list, cfg_ex_keywords, cfg_ex_hashtags = load_jobs_from_config(args.config)
            exclude_keywords.extend(cfg_ex_keywords)
            exclude_hashtags.extend(cfg_ex_hashtags)
        except Exception as e:
            print(f"Config error: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
    if args.regions:
        region_list = [x.strip() for x in args.regions.split(",") if x.strip()]

    if args.engine == "instaloader":
        crawler = InstaSelfEmployedCrawler(
            login_user=args.login_user,
            login_pass=args.login_pass,
            session_dir=args.session_dir,
            session_file=args.session_file,
            iphone_support=args.iphone_support,
            search_fallback=args.search_fallback,
            search_profile_limit=args.search_profile_limit,
            search_posts_per_profile=args.search_posts_per_profile,
            regions=region_list,
            exclude_keywords=exclude_keywords,
            exclude_hashtags=exclude_hashtags,
            quiet=args.quiet,
        )
    else:
        settings_path = args.ig_settings
        if not settings_path:
            session_dir = args.session_dir or ".insta_session"
            ensure_dir(session_dir)
            if args.login_user:
                settings_path = os.path.join(session_dir, f"instagrapi-{args.login_user}.json")
            else:
                settings_path = os.path.join(session_dir, "instagrapi.json")

        crawler = InstaGrapiCrawler(
            login_user=args.login_user,
            login_pass=args.login_pass,
            settings_path=settings_path,
            sessionid=args.ig_sessionid,
            regions=region_list,
            exclude_keywords=exclude_keywords,
            exclude_hashtags=exclude_hashtags,
            quiet=args.quiet,
        )

    all_items: list[InstaItem] = []

    # Multi-job mode
    if args.config:

        total_target = int(args.limit or 0)
        if total_target <= 0:
            total_target = sum(j["limit"] for j in jobs)

        collected_total = 0
        for j in jobs:
            if collected_total >= total_target:
                break
            per_job_limit = min(j["limit"], total_target - collected_total)

            items = crawler.crawl(
                hashtags=j["hashtags"],
                limit=per_job_limit,
                industry=j["industry"],
                location=j["location"],
                min_likes=args.min_likes,
                min_comments=args.min_comments,
                max_posts_scanned=args.max_posts_scanned,
                max_posts_per_user=args.max_posts_per_user,
            )
            all_items.extend(items)
            collected_total = len(all_items)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(args.out, f"instagram_feed_multi_{ts}.jsonl")
        csv_path = os.path.join(args.out, f"instagram_feed_multi_{ts}.csv")

        write_jsonl(all_items, jsonl_path)
        write_csv(all_items, csv_path)

        print(f"Saved JSONL: {jsonl_path}")
        print(f"Saved CSV:  {csv_path}")
        print(f"Collected total: {len(all_items)}")
        return 0

    # Single-job mode
    if not args.hashtags:
        print("Error: provide --hashtags or --config", file=sys.stderr)
        return 2

    hashtags = [h.strip().lstrip("#") for h in args.hashtags.split(",") if h.strip()]
    if not hashtags:
        print("No hashtags provided.", file=sys.stderr)
        return 2

    items = crawler.crawl(
        hashtags=hashtags,
        limit=args.limit,
        industry=args.industry,
        location=args.location,
        min_likes=args.min_likes,
        min_comments=args.min_comments,
        max_posts_scanned=args.max_posts_scanned,
        max_posts_per_user=args.max_posts_per_user,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args.out, f"instagram_feed_{args.industry}_{args.location}_{ts}.jsonl")
    csv_path = os.path.join(args.out, f"instagram_feed_{args.industry}_{args.location}_{ts}.csv")

    write_jsonl(items, jsonl_path)
    write_csv(items, csv_path)

    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Collected:  {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
