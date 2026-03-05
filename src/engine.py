"""
Generating response with Gemini API (google.genai)
"""

import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"


# 日本語文字（ひらがな・カタカナ・漢字・全角スペース）の Unicode ブロック
_JAPANESE_CHAR = r"\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3000"
# 句読点を除く（ひらがな・カタカナ・漢字のみ）。句読点の前にはスペースを入れない
_JAPANESE_LETTER = r"\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff"

# 句読点（、。）以外の記号・カッコを半角に統一（全角 → 半角）
_FULL_TO_HALF_SYMBOLS = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "＜": "<",
        "＞": ">",
        "「": '"',
        "」": '"',
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "～": "~",
        "＠": "@",
        "＃": "#",
        "＄": "$",
        "％": "%",
        "＆": "&",
        "＊": "*",
        "＋": "+",
        "－": "-",
        "＝": "=",
        "＾": "^",
        "＿": "_",
        "＼": "\\",
        "｜": "|",
    }
)


def _normalize_symbols(text: str) -> str:
    """句読点（、。）以外の記号・カッコを半角に統一する。"""
    if not text:
        return text
    return text.translate(_FULL_TO_HALF_SYMBOLS)


def _ensure_halfwidth_spaces(text: str) -> str:
    """
    半角英数字・単語の前後に半角スペースを入れる。
    日本語文字に隣接している場合のみ挿入。バッククォート内（ロール名・権限名など）は対象外。
    バッククォートの前後で日本語に隣接する場合は半角スペースを入れるが、句読点の前には入れない。
    """
    if not text:
        return text
    parts = text.split("`")
    for i in range(len(parts)):
        if i % 2 == 1:
            continue
        # バッククォート外のみ処理
        p = parts[i]
        p = re.sub(rf"([{_JAPANESE_CHAR}])([0-9A-Za-z]+)", r"\1 \2", p)
        p = re.sub(rf"([0-9A-Za-z]+)([{_JAPANESE_CHAR}])", r"\1 \2", p)
        parts[i] = p
    text = "`".join(parts)
    # バッククォートの前で日本語の直後ならスペース挿入
    text = re.sub(rf"([{_JAPANESE_CHAR}])`", r"\1 `", text)
    # 閉じバッククォートの直後が日本語の「文字」（句読点以外）のときだけスペース挿入。句読点（、。）の前にはスペース不要
    text = re.sub(rf"`([{_JAPANESE_LETTER}])", r"` \1", text)
    # カッコで囲んだバッククォート表記（例: 管理者(`roles/...`)）の前後に半角スペースを挿入（閉じ括弧の直後が " の場合は入れない）
    text = re.sub(r"([^\s])\(`", r"\1 (`", text)
    text = re.sub(r"`\)([^\s\"])", r"`) \1", text)
    # 開き括弧 ( の前で、直後が ` でないときスペース。例: 選択します(デフォルトは → 選択します (デフォルトは
    text = re.sub(r"([^\s])\((?!`)", r"\1 (", text)
    # 開き " の前のみスペース（" と単語の間には入れない）。例: 対する"Storage → 対する "Storage、ば、"Create → ば、"Create
    text = re.sub(r"([^\s])\"([A-Za-z])", r'\1 "\2', text)
    # 閉じ " の直後に日本語が続くときスペース。例: "Create" を、"Continue" を
    text = re.sub(rf'"([{_JAPANESE_LETTER}])', r'" \1', text)
    # ") の後に非スペースのときスペース。例: ")" IAM
    text = re.sub(r'\)"([^\s])', r')" \1', text)
    # (オプション) のように、`)` でない閉じ括弧の直後に日本語が続くときスペース。例: (オプション) アクセス
    text = re.sub(rf"(?<![`])\)([{_JAPANESE_LETTER}])", r") \1", text)
    return text.strip()


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def expand_query_for_search(
    user_question: str, thread_context: str | None = None
) -> str:
    """
    ユーザーの質問が短い・抽象的だとコンテキスト不足になりがちなため、
    Gemini で推論して必要そうな情報を付け加えた検索向けの質問に拡張する。
    thread_context がある場合はスレッド内の会話を考慮する。
    出力は質問と同じ言語で、検索クエリとして使うのに適した形にする。
    """
    client = _get_client()
    context_block = ""
    if thread_context and thread_context.strip():
        context_block = f"""Previous messages in this thread (for context):
{thread_context.strip()}

Use this to better understand what the user is asking. Now expand the following question.

"""
    prompt = f"""You are helping to improve a search query for Google Cloud documentation.
The user asked a short or vague question. Expand it into a more specific query by adding inferred context so that document search can find relevant pages. For example:
- If they ask about "制限" (limits), add "Google Cloud" or the service name if obvious from context.
- If they ask about a feature without naming the product, add the likely product or service name.
- Keep the expanded query concise (one or two sentences, or a few keywords). Do not answer the question yourself.
Output only the expanded query in the same language as the user, no explanation or quotation marks.

{context_block}User question:
{user_question}

Expanded query:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    out = (response.text or "").strip().strip("\"'")
    return out if out else user_question


def translate_to_english(user_question: str) -> str:
    """
    Translate the user's question to English for document search.
    If the question is already in English, return it as-is (or with minimal edits).
    """
    client = _get_client()
    prompt = f"""Translate the following user question into English for searching technical documentation.
If the text is already in English, return it unchanged or with minimal wording improvements.
Output only the English search query, no explanation or quotation marks.

User question:
{user_question}

English search query:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    out = (response.text or "").strip().strip("\"'")
    return out if out else user_question


def format_query_for_search_documents(english_query: str) -> str:
    """
    search_documents がヒットしやすい形にクエリを整形する。
    長い文や説明文は短いキーワード列にし、製品名・サービス名とトピックを明確にする。
    """
    if not english_query or len(english_query.strip()) < 50:
        return english_query.strip()
    client = _get_client()
    prompt = f"""Rewrite this as a short search query for a documentation search API (e.g. Google Cloud docs).
Rules:
- Output 3 to 8 key terms only, separated by spaces. No full sentences.
- Include the product or service name (e.g. Cloud Storage, GCS, Compute Engine) and the main topic (e.g. limits, quotas, best practices).
- Use English. Use terms that appear in official documentation (e.g. "bucket" not "container" for GCS).
- Output nothing else: no explanation, no quotes, no punctuation at the end.

Input:
{english_query}

Search query:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    out = (response.text or "").strip().strip("\"'.,;")
    return out if out else english_query.strip()


def _format_references(context_docs: str) -> str:
    """
    context_docs から「## 出典: ...」の行を抽出し、出典セクションの文字列を組み立てる。
    documents/... は https URL に変換。すでに https:// の場合はそのまま。
    """
    seen: set[str] = set()
    lines: list[str] = []
    for line in context_docs.splitlines():
        line = line.strip()
        if not line.startswith("##") or "出典" not in line:
            continue
        # 「## 出典:」の後ろを取得（URL 内の : と区別するため正規表現で出典直後のコロンのみ区切り）
        m = re.search(r"##\s*出典\s*:\s*(.+)", line)
        if not m:
            continue
        r = m.group(1).strip()
        if not r or r in seen:
            continue
        seen.add(r)
        if r.startswith("https://"):
            lines.append(f"- {r}")
        elif r.startswith("documents/"):
            if r.startswith("documents/docs.cloud.google.com/"):
                url = (
                    "https://cloud.google.com/"
                    + r[len("documents/docs.cloud.google.com/") :]
                )
            else:
                url = "https://" + r.replace("documents/", "", 1)
            lines.append(f"- {url}")
    if not lines:
        return ""
    return "\n\n**出典**\n本回答は以下のドキュメントを参照しています。\n\n" + "\n".join(
        lines
    )


def _strip_inline_links(text: str) -> str:
    """回答内の [text](url) 形式のインラインリンクを削除し、リンクテキストのみ残す。出典は末尾一覧で示すため。"""
    return re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)


# コマンド・環境変数設定行とみなしてバッククォートで囲むパターン（行頭が $ / export / set VAR= / gcloud 等）
_COMMAND_LINE_RE = re.compile(
    r"^(?:\s*)(?:\$\s+)?(?:export\s+\w+|set\s+\w+=|gcloud\s+|gsutil\s+|bq\s+|kubectl\s+|[A-Za-z_][A-Za-z0-9_]*\s*=\s*[\"'].*[\"'])(.*)$",
    re.MULTILINE,
)


def _wrap_bare_command_lines(text: str) -> str:
    """
    バッククォートで囲まれていないコマンド・環境変数行を `...` で囲む。
    すでに ` または ``` で囲まれている行は触らない。
    """
    lines = text.split("\n")
    result: list[str] = []
    in_fenced = False
    fenced_char = ""
    for line in lines:
        stripped = line.strip()
        # 既に ``` ブロック内ならそのまま
        if stripped.startswith("```"):
            in_fenced = not in_fenced
            fenced_char = "```" if in_fenced else ""
            result.append(line)
            continue
        if in_fenced:
            result.append(line)
            continue
        # 既に ` で始まる行（インラインコードのみの行）はそのまま
        if stripped.startswith("`") and stripped.endswith("`") and "`" in stripped[1:-1]:
            result.append(line)
            continue
        if re.match(r"^`[^`]+`\s*$", stripped):
            result.append(line)
            continue
        # コマンド風の行を検出して囲む（export VAR=, set VAR=, $ command, gcloud ... 等）
        if _COMMAND_LINE_RE.match(line) and "`" not in line:
            # 行頭の空白を保持して、内容を ` で囲む
            leading = line[: len(line) - len(line.lstrip())]
            result.append(leading + "`" + line.strip() + "`")
        else:
            result.append(line)
    return "\n".join(result)


def _normalize_line_breaks(text: str) -> str:
    """
    改行を整理する。
    - 回答本文に含まれる「## 出典:」行は削除（出典は末尾でまとめて付与するため）。
    - 連続する空行は最大2つに抑える。
    - 句点の直後に番号付きステップ（1. 2. など）が続く場合は、その前に改行2つを入れる。
    - 番号付き見出し（■ N. または N. ）の直前に空行が無い場合は空行1行を入れる。
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##") and "出典" in stripped:
            continue
        out.append(line)
    text = "\n".join(out)
    # 句点の直後に「数字. 」（ステップ）が始まる箇所の前に改行を入れる
    text = re.sub(r"([。.])\s*(\d+\.\s+)", r"\1\n\n\2", text)
    # 番号付き見出しの直前に空行が無い場合は空行を1行入れる（単一改行のときだけ）
    # キャプチャは (\s*) (■\s*)? (\d+\.\s+) の3つのみ
    text = re.sub(
        r"(?<!\n)\n(?!\n)(\s*)(■\s*)?(\d+\.\s+)",
        r"\n\n\1\2\3",
        text,
    )
    # 連続空行を最大2つに
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 番号付き見出しの行（■ 6. タイトル または 6. タイトル）にマッチ
_SECTION_HEADING_RE = re.compile(r"^(\s*)(■\s*)?(\d+)(\.\s+)(.*)$", re.MULTILINE)


def _fix_section_numbering(text: str) -> str:
    """
    番号付きセクション（■ 1. 〜 や 6. 〜）が逆行しないようにする。
    直前の番号より小さいまたは同じ番号が出た場合は、直前+1 に置き換える。
    """
    last_num = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal last_num
        prefix = m.group(1)  # 行頭の空白
        bullet = m.group(2) or ""  # ■ があれば
        num_str = m.group(3)
        dot_rest = m.group(4)  # ". "
        rest = m.group(5)
        num = int(num_str)
        if num <= last_num:
            num = last_num + 1
        last_num = num
        return f"{prefix}{bullet}{num}{dot_rest}{rest}"

    return _SECTION_HEADING_RE.sub(repl, text)


def generate_answer(user_question: str, context_docs: str) -> str:
    """
    Generate an answer to a given user question using the Gemini API.
    """
    client = _get_client()
    prompt = f"""You are a Google Cloud documentation expert. Answer the user's question using only the context documents below.
- If the information is not in the context, say "関連するドキュメントが見つかりませんでした｡" (or the equivalent in the answer language). Do not guess.
- Respond in the same language as the user's question (e.g. if the question is in Japanese, answer in Japanese; if in English, answer in English).
- Do NOT include markdown links [text](URL) or inline URLs in the answer body. Reference links will be listed at the end of the response as 出典.

**長さ（必ず守ること）:**
- 回答は簡潔に。Slack で読みやすい長さに抑え、要点・手順の骨子に絞る。
- 重複説明や「〜することもできます」などの補足は最小限に。必要な場合のみ1文で補足する。

**太字・記号（必ず守ること）:**
- 太字は「見出し」だけに使う。見出しとは番号付きの大見出し（1. 〇〇、2. △△）の行のみ。サブ項目（・〇〇:）や本文中の語句は太字にしない。
- 番号付きの見出しの先頭には ■ を付ける。例: ■ 1. クラスタのセキュリティを強化する
- 箇条書き（サブ項目）にはアスタリスク * を使わず、・ または • を使う。

**コード・コマンド（必ず守ること）:**
- コマンド、環境変数の設定例、パス、コード片は必ずバッククォートで囲む。例: `export PUBSUB_EMULATOR_HOST=[::1]:8432` または複数行は ``` で囲む。
- Linux/macOS と Windows で異なるコマンドがある場合は「Linux / macOS:」「Windows:」のようにラベルを付けた上で、それぞれのコマンドを `...` で囲む。

**レイアウト・セクション（必ず守ること）:**
- 手順は番号付きリストで書く。各メインステップ（■ 1. 〜、■ 2. 〜）の**前には必ず空行を1行入れる**。ステップ同士がつながらないようにする。
- 番号は連続させる（1, 2, 3...）。サブ手順は「・」や「Linux / macOS:」「Windows:」などで示し、大見出しの番号を重複させない。
- 各番号項目の中では、サブポイントごとに改行し、必要に応じて空行を1行入れて区切る。2〜3文ごとに改行を挟む。行頭に余白（インデント）を付けない。すべての行は左揃えで書く。

## User question:
{user_question}

## Context documents:
{context_docs}

## Answer:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw = response.text if response.text else "(回答を生成できませんでした)"
    text = _strip_inline_links(raw)
    text = _wrap_bare_command_lines(text)
    text = _normalize_line_breaks(text)
    text = _normalize_symbols(text)
    text = _ensure_halfwidth_spaces(text)
    text = _fix_section_numbering(text)
    refs = _format_references(context_docs)
    if refs:
        text = text.rstrip() + "\n\n" + refs
    return text
