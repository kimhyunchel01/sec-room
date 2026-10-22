"""AI 공시 토론방 (Streamlit + Supabase + OpenRouter)

- 방마다 사람 여러 명 + AI 여러 개(개수 제한 없음)가 함께 대화
- AI는 사용자가 "@이름"으로 부를 때만 답함 (수동)
- "자동 토론": 시작 버튼을 눌러야만 AI끼리 돌아가며 발언 (AI당 5회), 중지 버튼으로 언제든 멈춤
- SEC 공시 원문을 방에 첨부하고 AI에게 요약·반박 등을 요청

필요한 Streamlit Secrets (설정 방법은 안내서 참조):
  SEC_USER_AGENT, OPENROUTER_API_KEY, SUPABASE_URL, SUPABASE_KEY
  (선택) CREATE_PASSWORD, GLOBAL_DAILY_LIMIT
Supabase 정보가 없으면 '체험 모드'(메모리 저장, 서버 재시작 시 사라짐)로 동작.
"""

import datetime as dt
import hashlib
import math
import os
import random
import re
import string
import threading
import unicodedata
import uuid

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="AI 공시 토론방", page_icon="💬", layout="centered")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"

# 공시 종류별 한국어 설명
FORM_DESC = {
    "10-K": "연간 보고서",
    "10-Q": "분기 보고서",
    "8-K": "주요 사항 수시 공시",
    "20-F": "외국 기업 연간 보고서",
    "40-F": "캐나다 기업 연간 보고서",
    "6-K": "외국 기업 수시 공시",
    "3": "내부자 최초 지분 보고",
    "4": "내부자 지분 변동 보고",
    "5": "내부자 연간 지분 보고",
    "144": "내부자 주식 매도 예정 신고",
    "SC 13G": "5% 이상 지분 보유 보고(단순 투자)",
    "SCHEDULE 13G": "5% 이상 지분 보유 보고(단순 투자)",
    "SC 13D": "5% 이상 지분 보유 보고(경영 참여 목적)",
    "SCHEDULE 13D": "5% 이상 지분 보유 보고(경영 참여 목적)",
    "DEF 14A": "주주총회 안건 자료(위임장 권유)",
    "DEFA14A": "주주총회 추가 자료",
    "PRE 14A": "주주총회 안건 자료 예비본",
    "ARS": "주주용 연차 보고서",
    "S-1": "신규 증권 발행 신고서",
    "S-3": "간이 증권 발행 신고서",
    "S-4": "합병 관련 증권 신고서",
    "S-8": "임직원 주식 보상 등록",
    "424B2": "증권 발행 설명서",
    "424B3": "증권 발행 설명서",
    "424B4": "증권 발행 설명서",
    "424B5": "증권 발행 설명서",
    "FWP": "증권 발행 보충 자료",
    "11-K": "임직원 주식 보상 제도 연간 보고",
    "SD": "분쟁 광물 보고",
    "8-A12B": "증권 거래소 등록",
    "25-NSE": "상장 폐지 통지",
    "CORRESP": "SEC 질의에 대한 회사 답변",
    "UPLOAD": "SEC 질의 서한",
    "CERT": "거래소 상장 인증",
    "IRANNOTICE": "이란 관련 거래 공지",
    "NT 10-K": "연간 보고서 제출 지연 통지",
    "NT 10-Q": "분기 보고서 제출 지연 통지",
}


def describe_form(form: str) -> str:
    if form in FORM_DESC:
        return FORM_DESC[form]
    if form.endswith("/A") and form[:-2] in FORM_DESC:
        return FORM_DESC[form[:-2]] + " (정정)"
    return "기타 공시"


def normalize(text: str) -> str:
    """띄어쓰기·점·쉼표 제거, 소문자화"""
    return re.sub(r"[\s.,\-]", "", str(text)).lower()


def get_user_agent() -> str | None:
    try:
        ua = st.secrets.get("SEC_USER_AGENT")
    except Exception:
        ua = None
    return ua or os.environ.get("SEC_USER_AGENT")


def sec_fetch(url: str) -> requests.Response:
    headers = {"User-Agent": get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


def sec_get(url: str) -> dict:
    return sec_fetch(url).json()


def get_secret(name: str, default=None):
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val if val not in (None, "") else os.environ.get(name, default)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_companies() -> pd.DataFrame:
    data = sec_get(TICKERS_URL)
    df = pd.DataFrame(data.values())
    df = df.rename(columns={"cik_str": "cik", "ticker": "ticker", "title": "name"})
    df["ticker"] = df["ticker"].str.upper()
    return df[["cik", "ticker", "name"]]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_kr_names() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kr_names.csv")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except FileNotFoundError:
        return pd.DataFrame(columns=["kr_name", "ticker"])
    df.columns = ["kr_name", "ticker"] + list(df.columns[2:])
    df = df.dropna(subset=["kr_name", "ticker"])
    df["kr_name"] = df["kr_name"].str.strip()
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df[["kr_name", "ticker"]]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_filings(cik: int) -> pd.DataFrame:
    data = sec_get(SUBMISSIONS_URL.format(cik=cik))
    recent = data.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df
    for col in ["reportDate", "primaryDocument", "primaryDocDescription"]:
        if col not in df.columns:
            df[col] = ""

    def make_link(row):
        acc = str(row["accessionNumber"]).replace("-", "")
        doc = row["primaryDocument"]
        if doc:
            return ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc)
        return FILING_INDEX_URL.format(cik=cik, acc=acc)

    df["link"] = df.apply(make_link, axis=1)
    df["desc"] = df["form"].apply(describe_form)
    return df


def search(query: str, companies: pd.DataFrame, kr: pd.DataFrame) -> pd.DataFrame:
    """검색 결과: ticker, cik, name(영문), kr_name 목록"""
    q = normalize(query)
    if not q:
        return companies.iloc[0:0]

    kr = kr.copy()
    kr["norm"] = kr["kr_name"].apply(normalize)
    kr_hit = kr[kr["norm"].str.contains(q, regex=False)]
    # 정확히 일치하는 한글명을 앞에
    kr_hit = kr_hit.assign(exact=kr_hit["norm"] == q).sort_values("exact", ascending=False)
    tickers = list(dict.fromkeys(kr_hit["ticker"]))

    comp = companies.copy()
    comp["norm_t"] = comp["ticker"].apply(normalize)
    exact_t = comp[comp["norm_t"] == q]["ticker"].tolist()
    if len(q) >= 2:
        name_hit = comp[comp["name"].apply(normalize).str.contains(q, regex=False)]["ticker"].tolist()
    else:
        name_hit = []

    ordered = list(dict.fromkeys(exact_t + tickers + name_hit))[:30]
    result = comp.set_index("ticker").loc[[t for t in ordered if t in set(comp["ticker"])]].reset_index()
    return result[["ticker", "cik", "name"]]


def display_name(ticker: str, eng_name: str, kr: pd.DataFrame) -> str:
    hit = kr[kr["ticker"] == ticker]
    base = hit.iloc[0]["kr_name"] if not hit.empty else eng_name
    return f"{base}({ticker})"


# ---------------- 공시 본문 ----------------
PAGE_CHARS = 10_000  # 한 화면(=번역 1회)에 표시할 글자 수
SKIP_DOC = re.compile(r"^(R\d+\.htm|FilingSummary|.*\.(xsd|jpg|jpeg|png|gif|zip|js|css|json)$)", re.I)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def list_documents(cik: int, accession: str, primary: str) -> list[dict]:
    """공시 1건에 포함된 문서 목록 (본문 + 첨부). 첫 항목이 본문."""
    acc = accession.replace("-", "")
    base = FILING_INDEX_URL.format(cik=cik, acc=acc)
    docs = []
    try:
        items = sec_get(base + "index.json").get("directory", {}).get("item", [])
        for it in items:
            name = it.get("name", "")
            if not name or SKIP_DOC.match(name) or name.endswith("-index.htm") or name.endswith("-index-headers.html"):
                continue
            if not re.search(r"\.(htm|html|txt|xml)$", name, re.I):
                continue
            docs.append({"name": name, "url": base + name})
    except Exception:
        pass
    # 본문을 맨 앞으로
    if primary:
        prim_name = primary.split("/")[-1]
        docs = [d for d in docs if d["name"] != prim_name]
        docs.insert(0, {"name": primary, "url": base + primary})
    return docs


def doc_label(name: str, is_primary: bool) -> str:
    if is_primary:
        return f"본문 · {name}"
    low = name.lower()
    if re.search(r"ex[-_]?99", low):
        return f"첨부(보도자료 등) · {name}"
    if re.search(r"ex[-_]?(31|32)", low):
        return f"첨부(경영진 인증서) · {name}"
    if re.search(r"ex[-_]?\d+", low):
        return f"첨부 · {name}"
    return name


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False, max_entries=50)
def fetch_text(url: str) -> str:
    """SEC 문서에서 글자만 추출"""
    raw = sec_fetch(url).content
    if url.lower().endswith(".txt"):
        text = raw.decode("utf-8", errors="ignore")
    else:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "head", "ix:header"]):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
        text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if ln:
            out.append(ln)
            blank = 0
        else:
            blank += 1
            if blank == 1:
                out.append("")
    return "\n".join(out).strip()


def split_pages(text: str, size: int = PAGE_CHARS) -> list[str]:
    """줄 경계를 지키며 size 글자 단위로 나눔"""
    pages, cur, cur_len = [], [], 0
    for ln in text.split("\n"):
        if cur_len + len(ln) > size and cur:
            pages.append("\n".join(cur))
            cur, cur_len = [], 0
        while len(ln) > size:  # 매우 긴 한 줄
            pages.append(ln[:size])
            ln = ln[size:]
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        pages.append("\n".join(cur))
    return pages or [""]


# =====================================================================
# 설정값 (필요하면 숫자만 바꾸십시오)
# =====================================================================
KST = dt.timezone(dt.timedelta(hours=9))
CONTEXT_N = 20            # AI가 읽는 최근 대화 개수
FILING_CHARS = 40_000     # 공시 원문을 AI에게 보내는 최대 글자 수(한 쪽)
MAX_TOKENS_MANUAL = 1500  # 수동 호출 시 AI 답변 최대 길이
MAX_TOKENS_AUTO = 700     # 자동 토론 시 AI 답변 최대 길이
AUTO_PER_AI = 5           # 자동 토론에서 AI 1개당 발언 횟수
DEFAULT_ROOM_LIMIT = 60   # 방당 하루 AI 호출 기본 한도
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODELS_URL = "https://openrouter.ai/api/v1/models"
UNIQUE = {"rooms": ("code",), "members": ("room_id", "nickname"), "ais": ("room_id", "label")}


def now_iso() -> str:
    return dt.datetime.now(KST).isoformat()


def today_start_iso() -> str:
    n = dt.datetime.now(KST)
    return n.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


# =====================================================================
# 저장소: Supabase(REST) 또는 체험 모드(메모리)
# =====================================================================
class DupError(Exception):
    pass


class DB:
    def __init__(self):
        self.url = (get_secret("SUPABASE_URL") or "").rstrip("/")
        self.key = get_secret("SUPABASE_KEY") or ""
        self.mem_mode = not (self.url and self.key)
        self.tables = {t: [] for t in ("rooms", "members", "ais", "messages")}
        self.counter = {t: 0 for t in self.tables}
        self.lock = threading.Lock()

    # ---------- Supabase ----------
    def _h(self, extra=None):
        h = {"apikey": self.key, "Content-Type": "application/json"}
        if not self.key.startswith("sb_secret_"):
            h["Authorization"] = f"Bearer {self.key}"
        if extra:
            h.update(extra)
        return h

    def _req(self, method, table, params=None, body=None, extra=None):
        r = requests.request(
            method, f"{self.url}/rest/v1/{table}", params=params, json=body,
            headers=self._h(extra), timeout=30,
        )
        if r.status_code == 409:
            raise DupError()
        if r.status_code >= 400:
            raise RuntimeError(f"저장소 오류({r.status_code}): {r.text[:300]}")
        return r

    @staticmethod
    def _params(filters):
        return [(c, f"{op}.{v}") for c, op, v in filters]

    # ---------- 메모리 ----------
    @staticmethod
    def _match(row, filters):
        for c, op, v in filters:
            a = row.get(c)
            if op == "eq" and str(a) != str(v):
                return False
            if op == "gte" and not (a is not None and str(a) >= str(v)):
                return False
            if op == "gt" and not (a is not None and a > int(v)):
                return False
        return True

    # ---------- 공통 ----------
    def select(self, table, filters=(), order=None, limit=None):
        if self.mem_mode:
            with self.lock:
                rows = [dict(r) for r in self.tables[table] if self._match(r, filters)]
            if order:
                col, _, direction = order.partition(".")
                rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=(direction == "desc"))
            return rows[:limit] if limit else rows
        params = [("select", "*")] + self._params(filters)
        if order:
            params.append(("order", order))
        if limit:
            params.append(("limit", str(limit)))
        return self._req("GET", table, params).json()

    def insert(self, table, row):
        if self.mem_mode:
            with self.lock:
                for col_set in [UNIQUE.get(table)] if UNIQUE.get(table) else []:
                    for r in self.tables[table]:
                        if all(r.get(c) == row.get(c) for c in col_set):
                            raise DupError()
                self.counter[table] += 1
                new = dict(row)
                new["id"] = self.counter[table]
                new.setdefault("created_at", now_iso())
                self.tables[table].append(new)
                return dict(new)
        r = self._req("POST", table, body=row, extra={"Prefer": "return=representation"})
        return r.json()[0]

    def update(self, table, filters, values):
        if self.mem_mode:
            with self.lock:
                for r in self.tables[table]:
                    if self._match(r, filters):
                        r.update(values)
            return
        self._req("PATCH", table, self._params(filters), values)

    def delete(self, table, filters):
        if self.mem_mode:
            with self.lock:
                doomed = [r for r in self.tables[table] if self._match(r, filters)]
                self.tables[table] = [r for r in self.tables[table] if r not in doomed]
                if table == "rooms":
                    ids = {r["id"] for r in doomed}
                    for t in ("members", "ais", "messages"):
                        self.tables[t] = [r for r in self.tables[t] if r["room_id"] not in ids]
            return
        self._req("DELETE", table, self._params(filters))

    def count(self, table, filters=()):
        if self.mem_mode:
            return len(self.select(table, filters))
        params = [("select", "id"), ("limit", "1")] + self._params(filters)
        r = self._req("GET", table, params, extra={"Prefer": "count=exact"})
        total = r.headers.get("content-range", "*/0").split("/")[-1]
        return int(total) if total.isdigit() else 0


@st.cache_resource
def get_db() -> DB:
    return DB()


db = get_db()


# =====================================================================
# 도우미
# =====================================================================
def hash_pin(room_id, nick, pin) -> str:
    return hashlib.sha256(f"{room_id}:{nick}:{pin}".encode()).hexdigest()


def new_room_code() -> str:
    alphabet = string.ascii_uppercase.replace("O", "").replace("I", "") + "23456789"
    return "".join(random.choice(alphabet) for _ in range(6))


def fmt_time(iso: str) -> str:
    try:
        return dt.datetime.fromisoformat(str(iso)).astimezone(KST).strftime("%m-%d %H:%M")
    except Exception:
        return ""


def get_room(rid):
    rows = db.select("rooms", [("id", "eq", rid)])
    return rows[0] if rows else None


def get_ais(rid):
    return db.select("ais", [("room_id", "eq", rid)], order="position.asc")


def get_members(rid):
    return db.select("members", [("room_id", "eq", rid)], order="id.asc")


def add_system_msg(rid, text):
    db.insert("messages", {"room_id": rid, "author": "시스템", "kind": "system", "content": text})


def find_mentions(text, labels):
    found = []
    ordered = sorted(labels, key=len, reverse=True)
    for m in re.finditer(r"@(\S+)", text):
        tok = m.group(1)
        for lab in ordered:
            if tok.startswith(lab):
                if lab not in found:
                    found.append(lab)
                break
    return found


def flash(kind, text):
    st.session_state.setdefault("flash", []).append((kind, text))


@st.cache_data(ttl=3600, show_spinner=False)
def load_models():
    r = requests.get(OR_MODELS_URL, timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        p = m.get("pricing") or {}
        try:
            pin = float(p.get("prompt", 0)) * 1e6
            pout = float(p.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            pin = pout = None
        out.append({"id": m["id"], "in": pin, "out": pout})
    return sorted(out, key=lambda x: x["id"])


PROVIDER_KO = {"anthropic": "앤트로픽", "google": "구글", "openai": "오픈AI", "meta-llama": "메타", "deepseek": "딥시크",
               "qwen": "알리바바", "x-ai": "xAI", "mistralai": "미스트랄", "moonshotai": "문샷", "z-ai": "지푸",
               "nvidia": "엔비디아", "microsoft": "마이크로소프트", "cohere": "코히어", "perplexity": "퍼플렉시티",
               "amazon": "아마존", "minimax": "미니맥스"}
FAMILY_KO = [("claude", "클로드"), ("gemini", "제미나이"), ("gemma", "젬마"), ("llama", "라마"), ("deepseek", "딥시크"),
             ("qwen", "큐웬"), ("grok", "그록"), ("mistral", "미스트랄"), ("kimi", "키미"), ("glm", "GLM"), ("gpt", "GPT")]


def model_ko_name(model_id: str) -> str:
    """'anthropic/claude-sonnet-4.5' -> '클로드 sonnet 4.5' 식의 한국어 표기."""
    prov, _, name = model_id.partition("/")
    if not name:
        prov, name = "", prov
    low = name.lower()
    for en, ko in FAMILY_KO:
        if low.startswith(en):
            name = ko + name[len(en):]
            break
    name = name.replace("-", " ").replace(":free", " 무료")
    return name


def model_label(m):
    ko = model_ko_name(m["id"])
    if m["in"] is None:
        return f"{ko} · {m['id']}"
    return f"{ko} · {m['id']} · 100만 토큰당 입력 ${m['in']:.2f} / 출력 ${m['out']:.2f}"


# =====================================================================
# 사용 한도·통계
# =====================================================================
def room_calls_today(rid) -> int:
    return db.count("messages", [("room_id", "eq", rid), ("kind", "eq", "ai"), ("created_at", "gte", today_start_iso())])


def global_calls_today() -> int:
    return db.count("messages", [("kind", "eq", "ai"), ("created_at", "gte", today_start_iso())])


def global_limit() -> int:
    try:
        return int(get_secret("GLOBAL_DAILY_LIMIT", 300))
    except (TypeError, ValueError):
        return 300


def room_stats_today(rid):
    rows = db.select("messages", [("room_id", "eq", rid), ("kind", "eq", "ai"), ("created_at", "gte", today_start_iso())])
    tin = sum(int(r.get("in_tok") or 0) for r in rows)
    tout = sum(int(r.get("out_tok") or 0) for r in rows)
    cost = sum(float(r.get("cost") or 0) for r in rows)
    has_cost = any(r.get("cost") is not None for r in rows)
    return len(rows), tin, tout, (cost if has_cost else None)


def check_limits(room):
    n = room_calls_today(room["id"])
    if n >= int(room["daily_limit"]):
        raise RuntimeError(f"이 방의 오늘 AI 호출 한도({room['daily_limit']}회)에 도달했습니다. 방장이 한도를 올릴 수 있습니다.")
    if global_calls_today() >= global_limit():
        raise RuntimeError("앱 전체의 오늘 AI 호출 한도에 도달했습니다. 내일 다시 이용하십시오.")


# =====================================================================
# AI 호출 (OpenRouter)
# =====================================================================
def call_openrouter(model, system, user_text, max_tokens):
    key = get_secret("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 추가하십시오.")
    resp = requests.post(
        OR_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AI Disclosure Room"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            "max_tokens": max_tokens,
            "usage": {"include": True},
        },
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"AI 호출 실패({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"AI 호출 실패: {str(data['error'])[:300]}")
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise RuntimeError("AI 응답 형식이 올바르지 않습니다.")
    u = data.get("usage") or {}
    return text.strip(), u.get("prompt_tokens"), u.get("completion_tokens"), u.get("cost")


def build_system(ai, room):
    s = (
        f"당신은 미국 기업 공시를 놓고 토론하는 대화방의 AI 참여자 '{ai['label']}'입니다.\n"
        "규칙: 한국어로 답합니다. 사용자는 투자 비전문가이므로 쉬운 말로 간결하게 설명합니다. "
        "특정 종목의 매수·매도를 권유하지 않습니다. 공시에 없는 내용을 지어내지 않고, 모르면 모른다고 말합니다. "
        "숫자와 고유명사는 공시 원문 그대로 인용합니다."
    )
    if ai.get("role"):
        s += f"\n당신에게 지정된 역할: {ai['role']}"
    return s


def filing_block(room, with_text):
    f = room.get("filing")
    if not f:
        return ""
    head = f"[첨부 공시] {f.get('name', '')} · {f.get('form', '')}({f.get('desc', '')}) · {f.get('date', '')}"
    if not with_text:
        return head + "\n(원문은 이번에 보내지 않습니다. 앞선 대화에 요약이 있으면 그것을 참고하십시오.)"
    pages = split_pages(fetch_text(f["url"]), FILING_CHARS)
    pg = min(int(f.get("page", 0)), len(pages) - 1)
    return f"{head}\n(영문 원문 {pg + 1}/{len(pages)}쪽)\n{pages[pg]}"


def transcript(rid):
    rows = db.select("messages", [("room_id", "eq", rid)], order="id.desc", limit=CONTEXT_N)
    rows.reverse()
    lines = []
    for r in rows:
        if r["kind"] == "system":
            continue
        lines.append(f"[{r['author']}] {str(r['content'])[:3000]}")
    return "\n".join(lines)


def run_ai(room, ai, request, with_filing, max_tokens):
    """AI 1회 호출 → 결과를 대화에 저장"""
    check_limits(room)
    user_text = (
        f"{filing_block(room, with_filing)}\n\n[최근 대화]\n{transcript(room['id'])}\n\n"
        f"[요청]\n{request}\n\n위 요청 중 '{ai['label']}'에게 해당하는 부분에 답하십시오."
    ).strip()
    text, tin, tout, cost = call_openrouter(ai["model"], build_system(ai, room), user_text, max_tokens)
    if not text:
        text = "(빈 답변이 돌아왔습니다. 모델을 바꿔 보십시오.)"
    db.insert("messages", {
        "room_id": room["id"], "author": ai["label"], "kind": "ai", "content": text,
        "in_tok": tin, "out_tok": tout, "cost": cost, "model": ai["model"],
    })
    if with_filing and room.get("filing"):
        db.update("rooms", [("id", "eq", room["id"])], {"filing_sent": True})
        room["filing_sent"] = True
    return text


# =====================================================================
# 화면: 입장 / 방 만들기
# =====================================================================
def enter_room(room, nick):
    st.session_state["room_id"] = room["id"]
    st.session_state["nick"] = nick
    st.session_state.setdefault("sid", uuid.uuid4().hex)
    st.rerun()


def room_join_hash(code, pw) -> str:
    return hashlib.sha256(f"join:{code}:{pw}".encode()).hexdigest()


def screen_entry():
    st.title("💬 AI 공시 토론방")
    st.caption("사람 여러 명과 AI 여러 개가 미국 기업 공시를 놓고 대화하는 방입니다. AI는 @이름으로 부를 때만 답합니다.")
    if db.mem_mode:
        st.warning("체험 모드: 저장소(Supabase)가 연결되지 않아 서버가 재시작되면 대화가 사라집니다.")
    tab_join, tab_new = st.tabs(["방 목록", "방 만들기"])

    with tab_join:
        rooms, mem_count, ai_count = [], {}, {}
        try:
            rooms = db.select("rooms", order="id.desc", limit=50)
            for m in db.select("members"):
                mem_count[m["room_id"]] = mem_count.get(m["room_id"], 0) + 1
            for a in db.select("ais"):
                ai_count[a["room_id"]] = ai_count.get(a["room_id"], 0) + 1
        except Exception as e:
            st.error(str(e))
        if not rooms:
            st.info("아직 만들어진 방이 없습니다. '방 만들기' 탭에서 새 방을 만드십시오.")
        else:
            def room_label(r):
                lock = "🔒" if r.get("join_hash") else "🔓"
                return (f"{lock} {r['name']} · 방장 {r['creator']} · "
                        f"{mem_count.get(r['id'], 0)}/{r['capacity']}명 · AI {ai_count.get(r['id'], 0)}개")

            labels = [room_label(r) for r in rooms]
            pick = st.selectbox("입장할 방 (🔒 = 입장 암호 필요)", labels)
            room = rooms[labels.index(pick)]
            with st.form("join"):
                nick = st.text_input("닉네임").strip()
                pw = st.text_input("입장 암호 (🔒 방에 처음 들어갈 때만)", type="password") if room.get("join_hash") else ""
                ok = st.form_submit_button("입장")
            if ok:
                try:
                    mems = get_members(room["id"])
                    nick = unicodedata.normalize("NFC", nick)
                    mine = [m for m in mems if unicodedata.normalize("NFC", m["nickname"]).strip() == nick]
                    if not nick:
                        st.error("닉네임을 입력하십시오.")
                    elif mine:
                        enter_room(room, mine[0]["nickname"])
                    elif room.get("join_hash") and room_join_hash(room["code"], pw) != room["join_hash"]:
                        st.error("입장 암호가 맞지 않습니다.")
                    elif len(mems) >= int(room["capacity"]):
                        st.error(f"정원({room['capacity']}명)이 가득 찼습니다.")
                    else:
                        db.insert("members", {"room_id": room["id"], "nickname": nick, "pin_hash": ""})
                        add_system_msg(room["id"], f"{nick}님이 입장했습니다.")
                        enter_room(room, nick)
                except DupError:
                    st.error("이미 사용 중인 닉네임입니다.")
                except Exception as e:
                    st.error(str(e))

    with tab_new:
        need_pw = get_secret("CREATE_PASSWORD")
        with st.form("new"):
            name = st.text_input("방 이름").strip()
            nick = st.text_input("내 닉네임(방장)").strip()
            cap = st.slider("정원(최대 20명)", 2, 20, 10)
            join_pw = st.text_input("입장 암호 (선택 — 비우면 누구나 입장할 수 있는 열린 방)", type="password")
            pw = st.text_input("방 만들기 암호", type="password") if need_pw else ""
            ok = st.form_submit_button("방 만들기")
        if ok:
            if need_pw and pw != need_pw:
                st.error("방 만들기 암호가 맞지 않습니다.")
            elif not name or not nick:
                st.error("방 이름과 닉네임을 입력하십시오.")
            else:
                try:
                    room = None
                    for _ in range(5):
                        code = new_room_code()
                        try:
                            room = db.insert("rooms", {
                                "code": code, "name": name, "creator": nick, "capacity": cap,
                                "daily_limit": DEFAULT_ROOM_LIMIT, "filing": None, "filing_sent": False, "auto": None,
                                "join_hash": room_join_hash(code, join_pw) if join_pw else None,
                            })
                            break
                        except DupError:
                            room = None
                    if not room:
                        st.error("방 생성에 실패했습니다. 다시 시도하십시오.")
                    else:
                        db.insert("members", {"room_id": room["id"], "nickname": nick, "pin_hash": ""})
                        add_system_msg(room["id"], "방이 만들어졌습니다.")
                        enter_room(room, nick)
                except Exception as e:
                    st.error(str(e))


# =====================================================================
# 화면: 사이드바 (공시 첨부 / AI 관리 / 참여자 / 설정)
# =====================================================================
def sidebar_filing(room):
    with st.sidebar.expander("📄 공시 첨부", expanded=not room.get("filing")):
        f = room.get("filing")
        if f:
            st.caption(f"현재: {f.get('name')} · {f.get('form')} · {f.get('date')}")
        if not get_user_agent():
            st.error("SEC_USER_AGENT 설정이 필요합니다.")
            return
        q = st.text_input("종목 검색(한글명·영문명·티커)", key="fq")
        if not q:
            return
        try:
            comps, kr = load_companies(), load_kr_names()
            res = search(q, comps, kr)
        except Exception as e:
            st.error(f"종목 검색 실패: {e}")
            return
        if res.empty:
            st.caption("검색 결과가 없습니다.")
            return
        opts = {display_name(r.ticker, r.name, kr): int(r.cik) for r in res.itertuples()}
        pick = st.selectbox("종목 선택", list(opts), key="fpick")
        cik = opts[pick]
        try:
            fl = load_filings(cik).head(40)
        except Exception as e:
            st.error(f"공시 목록 조회 실패: {e}")
            return
        if fl.empty:
            st.caption("공시가 없습니다.")
            return
        rows = {f"{r.filingDate} · {r.form}({r.desc})": r for r in fl.itertuples()}
        fkey = st.selectbox("공시 선택", list(rows), key="ffil")
        r = rows[fkey]
        try:
            docs = list_documents(cik, r.accessionNumber, r.primaryDocument)
        except Exception as e:
            st.error(f"문서 목록 조회 실패: {e}")
            return
        if not docs:
            st.caption("읽을 수 있는 문서가 없습니다.")
            return
        dmap = {doc_label(d["name"], i == 0): d for i, d in enumerate(docs)}
        dkey = st.selectbox("문서 선택", list(dmap), key="fdoc")
        d = dmap[dkey]
        try:
            n_pages = len(split_pages(fetch_text(d["url"]), FILING_CHARS))
        except Exception as e:
            st.error(f"원문 가져오기 실패: {e}")
            return
        page = 1
        if n_pages > 1:
            page = st.number_input(f"원문 쪽 번호 (1~{n_pages}, 쪽당 약 {FILING_CHARS // 1000}천 자)", 1, n_pages, 1, key="fpg")
        if st.button("이 공시를 방에 첨부", key="fattach"):
            filing = {
                "name": pick, "ticker": pick, "form": r.form, "desc": r.desc, "date": str(r.filingDate),
                "url": d["url"], "page": int(page) - 1,
            }
            db.update("rooms", [("id", "eq", room["id"])], {"filing": filing, "filing_sent": False})
            add_system_msg(room["id"], f"공시 첨부: {pick} · {r.form}({r.desc}) · {r.filingDate} · 원문 {int(page)}/{n_pages}쪽")
            st.rerun()


def sidebar_ais(room, is_owner):
    ais = get_ais(room["id"])
    with st.sidebar.expander(f"🤖 AI 목록 ({len(ais)}개)", expanded=not ais):
        for ai in ais:
            st.markdown(f"**@{ai['label']}** · {model_ko_name(ai['model'])}")
            if is_owner:
                role = st.text_area("역할", value=ai.get("role") or "", key=f"role_{ai['id']}", height=70)
                c1, c2 = st.columns(2)
                if c1.button("역할 저장", key=f"rs_{ai['id']}"):
                    db.update("ais", [("id", "eq", ai["id"])], {"role": role})
                    st.rerun()
                if c2.button("삭제", key=f"rd_{ai['id']}"):
                    db.delete("ais", [("id", "eq", ai["id"])])
                    st.rerun()
            elif ai.get("role"):
                st.caption(f"역할: {ai['role']}")
        if is_owner:
            st.divider()
            st.markdown("**AI 추가**")
            label = st.text_input("부를 이름 (띄어쓰기 없이, 예: 클로드)", key="new_label", max_chars=12).strip()
            try:
                models = load_models()
                labels = [model_label(m) for m in models]
                mk = st.selectbox("모델 (글자를 입력해 검색)", labels, key="new_model")
                model_id = models[labels.index(mk)]["id"]
            except Exception:
                model_id = st.text_input("모델 ID (OpenRouter 모델 목록 조회 실패 — 직접 입력)", key="new_model_txt").strip()
            role = st.text_area("역할 (선택, 예: 공시 원문 요약 담당)", key="new_role", height=70)
            if st.button("AI 추가", key="add_ai"):
                if not label or re.search(r"\s", label) or not model_id:
                    st.error("이름(띄어쓰기 없음)과 모델을 입력하십시오.")
                else:
                    try:
                        db.insert("ais", {"room_id": room["id"], "label": label, "model": model_id, "role": role, "position": len(ais)})
                        st.rerun()
                    except DupError:
                        st.error("같은 이름의 AI가 이미 있습니다.")
    return ais


def sidebar_members(room, is_owner, nick):
    mems = get_members(room["id"])
    with st.sidebar.expander(f"👥 참여자 ({len(mems)}/{room['capacity']})"):
        for m in mems:
            c1, c2 = st.columns([3, 1])
            c1.write(m["nickname"] + (" 👑" if m["nickname"] == room["creator"] else ""))
            if is_owner and m["nickname"] != nick and c2.button("내보내기", key=f"kick_{m['id']}"):
                db.delete("members", [("id", "eq", m["id"])])
                add_system_msg(room["id"], f"{m['nickname']}님이 내보내졌습니다.")
                st.rerun()


def sidebar_owner(room):
    with st.sidebar.expander("⚙️ 방 설정 (방장)"):
        lim = st.number_input("방 하루 AI 호출 한도", 1, 1000, int(room["daily_limit"]), key="lim")
        if st.button("한도 저장"):
            db.update("rooms", [("id", "eq", room["id"])], {"daily_limit": int(lim)})
            st.rerun()
        st.divider()
        did = st.number_input("삭제할 글 번호(#)", 0, 10**9, 0, key="delid")
        if st.button("글 삭제") and did:
            db.delete("messages", [("id", "eq", int(did)), ("room_id", "eq", room["id"])])
            st.rerun()
        st.divider()
        sure = st.checkbox("방을 완전히 삭제합니다", key="sure_del")
        if st.button("방 삭제") and sure:
            db.delete("rooms", [("id", "eq", room["id"])])
            leave()


def leave():
    for k in ("room_id", "nick"):
        st.session_state.pop(k, None)
    st.rerun()


# =====================================================================
# 화면: 대화 / 자동 토론
# =====================================================================
def render_messages_body(rid):
    try:
        rows = db.select("messages", [("room_id", "eq", rid)], order="id.desc", limit=100)
    except Exception as e:
        st.error(str(e))
        return
    rows.reverse()
    for m in rows:
        if m["kind"] == "system":
            st.caption(f"📢 {m['content']}")
            continue
        role = "assistant" if m["kind"] == "ai" else "user"
        with st.chat_message(role):
            tag = " 🤖" if m["kind"] == "ai" else ""
            st.markdown(f"**{m['author']}**{tag} · {fmt_time(m.get('created_at'))} · #{m['id']}")
            st.markdown(m["content"])


def render_messages(rid, auto_refresh):
    if auto_refresh:
        st.fragment(run_every=5)(render_messages_body)(rid)
    else:
        render_messages_body(rid)


def auto_panel(room, ais, sid, nick):
    auto = room.get("auto") or {}
    active = bool(auto.get("active"))
    with st.expander("🗣️ AI 자동 토론", expanded=active):
        if active:
            st.progress(min(auto["turn"] / max(auto["total"], 1), 1.0), text=f"주제: {auto['topic']} · 진행 {auto['turn']}/{auto['total']}회")
            c1, c2 = st.columns(2)
            if c1.button("⏹ 중지", type="primary", key="auto_stop"):
                db.update("rooms", [("id", "eq", room["id"])], {"auto": {**auto, "active": False}})
                add_system_msg(room["id"], f"{nick}님이 자동 토론을 중지했습니다. ({auto['turn']}/{auto['total']}회)")
                st.rerun()
            if auto.get("runner") != sid and c2.button("▶ 이 화면에서 이어서 진행", key="auto_take"):
                db.update("rooms", [("id", "eq", room["id"])], {"auto": {**auto, "runner": sid}})
                st.rerun()
            if auto.get("runner") != sid:
                st.caption("다른 사람의 화면에서 진행 중입니다. 그 화면이 닫히면 '이어서 진행'을 누르십시오.")
        else:
            if len(ais) < 2:
                st.caption("자동 토론은 AI가 2개 이상일 때 사용할 수 있습니다.")
                return
            topic = st.text_area("토론 주제 (예: 이 공시가 회사 실적에 긍정적인지 토론)", key="auto_topic", height=70)
            total = len(ais) * AUTO_PER_AI
            st.caption(f"AI {len(ais)}개 × 각 {AUTO_PER_AI}회 = 총 {total}회 호출. 순서: " + " → ".join(a["label"] for a in ais))
            if st.button("▶ 토론 시작", key="auto_start"):
                if not topic.strip():
                    st.error("주제를 입력하십시오.")
                else:
                    state = {"active": True, "topic": topic.strip(), "turn": 0, "total": total,
                             "runner": sid, "order": [a["label"] for a in ais]}
                    db.update("rooms", [("id", "eq", room["id"])], {"auto": state})
                    add_system_msg(room["id"], f"{nick}님이 자동 토론을 시작했습니다. 주제: {topic.strip()} (총 {total}회)")
                    st.rerun()


def run_auto_turn(room, sid):
    auto = room.get("auto") or {}
    if not (auto.get("active") and auto.get("runner") == sid):
        return
    rid = room["id"]
    ais = {a["label"]: a for a in get_ais(rid)}
    turn, total, order = auto["turn"], auto["total"], auto["order"]

    def finish(msg):
        db.update("rooms", [("id", "eq", rid)], {"auto": {**auto, "active": False}})
        add_system_msg(rid, msg)
        st.rerun()

    if turn >= total:
        finish(f"자동 토론이 끝났습니다. (총 {total}회)")
    label = order[turn % len(order)]
    ai = ais.get(label)
    if not ai:
        finish(f"AI '{label}'이(가) 목록에 없어 자동 토론을 중단했습니다.")
    k = turn // len(order) + 1
    request = (
        f"[자동 토론] 주제: {auto['topic']}\n지금은 '{label}'의 {k}번째 발언입니다(AI당 총 {AUTO_PER_AI}회). "
        "앞선 발언을 읽고, 자신에게 지정된 역할에 맞게 의견을 말하십시오. 이미 나온 말을 반복하지 말고 5문장 이내로 쓰십시오."
    )
    with_filing = bool(room.get("filing")) and not room.get("filing_sent")
    try:
        with st.spinner(f"{label} 발언 중... ({turn + 1}/{total})"):
            run_ai(room, ai, request, with_filing, MAX_TOKENS_AUTO)
    except Exception as e:
        flash("error", f"자동 토론 중단: {e}")
        finish(f"오류로 자동 토론을 중단했습니다. ({turn}/{total}회 완료)")
    fresh = get_room(rid)
    fa = (fresh or {}).get("auto") or {}
    if fa.get("active") and fa.get("runner") == sid:
        db.update("rooms", [("id", "eq", rid)], {"auto": {**fa, "turn": turn + 1}})
        st.rerun()


# =====================================================================
# 화면: 방
# =====================================================================
def screen_room():
    rid, nick = st.session_state["room_id"], st.session_state["nick"]
    sid = st.session_state.setdefault("sid", uuid.uuid4().hex)
    room = get_room(rid)
    mine = [m for m in get_members(rid)] if room else []
    if not room or nick not in [m["nickname"] for m in mine]:
        st.session_state["flash"] = [("error", "방이 삭제되었거나 더 이상 참여자가 아닙니다.")]
        leave()
        return
    is_owner = nick == room["creator"]

    sidebar_filing(room)
    ais = sidebar_ais(room, is_owner)
    sidebar_members(room, is_owner, nick)
    if is_owner:
        sidebar_owner(room)
    st.sidebar.divider()
    auto_refresh = st.sidebar.checkbox("5초마다 자동 새로고침", value=False)
    if st.sidebar.button("🔄 새로고침"):
        st.rerun()
    if st.sidebar.button("방 나가기"):
        leave()

    st.title(f"💬 {room['name']}")
    n, tin, tout, cost = room_stats_today(rid)
    stat = f"오늘 AI 호출 {n}/{room['daily_limit']}회 · 토큰 입력 {tin:,} / 출력 {tout:,}"
    if cost is not None:
        stat += f" · 비용 약 ${cost:.4f}"
    st.caption(f"{'🔒' if room.get('join_hash') else '🔓'} {nick}{' (방장)' if is_owner else ''} · {stat}")
    if db.mem_mode:
        st.warning("체험 모드: 서버가 재시작되면 대화가 사라집니다.")
    for kind, text in st.session_state.pop("flash", []):
        (st.error if kind == "error" else st.info)(text)
    if room.get("filing"):
        f = room["filing"]
        st.info(f"📄 첨부 공시: {f.get('name')} · {f.get('form')}({f.get('desc')}) · {f.get('date')}")

    auto_panel(room, ais, sid, nick)
    render_messages(rid, auto_refresh)
    st.caption("AI 답변은 참고용이며 투자 권유가 아닙니다. 투자 판단과 책임은 본인에게 있습니다.")

    # 자동 토론은 화면 표시 뒤에 1회씩 진행
    run_auto_turn(room, sid)

    with_filing = False
    if room.get("filing"):
        with_filing = st.checkbox(
            f"공시 원문 함께 보내기 (글자 수만큼 비용 증가, 약 {FILING_CHARS // 1000}천 자)",
            value=not room.get("filing_sent"),
            key=f"wf_{rid}_{int(bool(room.get('filing_sent')))}_{room['filing'].get('url')}",
        )
    msg = st.chat_input("메시지 입력 (AI를 부르려면 @이름)")
    if msg:
        msg = msg.strip()[:2000]
        if (room.get("auto") or {}).get("active"):
            st.warning("자동 토론이 진행 중입니다. 중지한 뒤 입력하십시오.")
            return
        db.insert("messages", {"room_id": rid, "author": nick, "kind": "user", "content": msg})
        labels = [a["label"] for a in ais]
        mentions = find_mentions(msg, labels)
        if "@" in msg and not mentions and labels:
            flash("info", "일치하는 AI 이름이 없습니다. 오른쪽 목록의 @이름을 확인하십시오.")
        by_label = {a["label"]: a for a in ais}
        used_filing = with_filing
        for lab in mentions:
            try:
                with st.spinner(f"{lab} 답변 중..."):
                    run_ai(room, by_label[lab], msg, used_filing, MAX_TOKENS_MANUAL)
                used_filing = False  # 같은 메시지에서 두 번째 AI부터는 앞 AI 답변을 읽게 함
            except Exception as e:
                flash("error", f"{lab}: {e}")
                break
        st.rerun()


# =====================================================================
# 시작
# =====================================================================
if "room_id" in st.session_state:
    screen_room()
else:
    for kind, text in st.session_state.pop("flash", []):
        (st.error if kind == "error" else st.info)(text)
    screen_entry()
