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
    out = (response.text or "").strip().strip('"\'')
    return out if out else user_question


def _format_references(context_docs: str) -> str:
    """
    context_docs から「## 出典: documents/...」を抽出し、参照セクションの文字列を組み立てる。
    documents/docs.cloud.google.com/... → https://cloud.google.com/... に変換する。
    """
    refs = re.findall(r"^## 出典:\s*(documents/[^\s\n]+)", context_docs, re.MULTILINE)
    seen = set()
    lines = []
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        if r.startswith("documents/docs.cloud.google.com/"):
            url = "https://cloud.google.com/" + r[len("documents/docs.cloud.google.com/") :]
        else:
            url = "https://" + r.replace("documents/", "", 1)
        lines.append(f"- {url}")
    if not lines:
        return ""
    return "\n\n**参照**\n" + "\n".join(lines)


def generate_answer(user_question: str, context_docs: str) -> str:
    """
    Generate an answer to a given user question using the Gemini API.
    """
    client = _get_client()
    prompt = f"""You are a Google Cloud documentation expert. Answer the user's question using only the context documents below.
- If the information is not in the context, say "ドキュメントに記載がありません。" (or the equivalent in the answer language). Do not guess.
- Respond in the same language as the user's question (e.g. if the question is in Japanese, answer in Japanese; if in English, answer in English).

## User question:
{user_question}

## Context documents:
{context_docs}

## Answer:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw = response.text if response.text else "(回答を生成できませんでした)"
    text = _normalize_symbols(raw)
    text = _ensure_halfwidth_spaces(text)
    refs = _format_references(context_docs)
    if refs and refs not in text:
        text = text.rstrip() + "\n\n" + refs
    return text
