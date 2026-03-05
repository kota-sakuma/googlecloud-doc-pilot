"""
Main entry point for the application.
"""

import asyncio
import logging
import os
import re

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from .mcp_client import fetch_googlecloud_doc
from .engine import (
    expand_query_for_search,
    format_query_for_search_documents,
    generate_answer,
    translate_to_english,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))


# Slack Block Kit: section の mrkdwn は 3000 文字まで / 1メッセージあたり最大 50 ブロック
_SECTION_TEXT_MAX = 3000
_SLACK_BLOCKS_MAX = 50


def _markdown_to_slack_mrkdwn(text: str) -> str:
    """
    Markdown 風のテキストを Slack の mrkdwn に変換する。
    - **bold** → *bold*（Slack は * の直後に空白があると太字にならないので詰める）
    - ## 見出し → *見出し*
    - 単独の *（太字になっていない）は削除またはビュレットに変換して表示崩れを防ぐ
    - バッククォート `...` および ``` ブロック内は変換しない（コードとしてそのまま表示）
    """
    if not text:
        return text
    # バッククォートで囲まれた部分は保護してから変換する（コード内の ** を太字にしない）
    placeholders: list[str] = []
    def save_chunk(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"\x00P{len(placeholders)-1}\x00"

    # ``` ブロックを保護
    text = re.sub(r"```[\s\S]*?```", save_chunk, text)
    # `...` インラインコードを保護
    text = re.sub(r"`[^`]+`", save_chunk, text)
    # **text** → *text*（Slack の太字）
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # * と語の間のスペースを除去（* Cloud Storage * → *Cloud Storage* でないと Slack で太字にならない）
    text = re.sub(r"\*\s+([^*]+?)\s+\*", r"*\1*", text)
    # ## 見出し → *見出し*
    text = re.sub(r"^##\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # 見出しや句点の直後の単独 "* " は太字ではないので削除し、改行で区切る（例: 強化する* 厳格な → 強化する\n\n厳格な）
    text = re.sub(r"(\S)\*\s+", r"\1\n\n", text)
    # 行頭の "*"（スペースあり/なし）を箇条書き "・ " に統一。ただし *...* の1行太字はそのまま
    lines = text.split("\n")
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("*"):
            # 同じ行で *...* と閉じている太字は触らない
            if re.match(r"^\*[^*]*\*\s*$", stripped):
                out.append(line)
            else:
                out.append(re.sub(r"^(\s*)\*\s*", r"\1・ ", line))
        else:
            out.append(line)
    text = "\n".join(out)
    # 既に • が含まれている箇所も ・ に統一（サイズ差をなくす）
    text = text.replace("• ", "・ ").replace("•", "・")
    # 保護したコードブロックを復元
    for i, ph in enumerate(placeholders):
        text = text.replace(f"\x00P{i}\x00", ph)
    return text


def _bold_heading_only(step_text: str) -> str:
    """
    番号付き見出しの先頭行だけ太字を残し、それ以外の *...* を外す（Slack で見出し以外の太字をやめる）。
    先頭行は「■ 1. 〇〇」または「1. 〇〇」の形式で、その〇〇部分だけ * で囲む。2行目以降の *...* は * を削除して通常テキストにする。
    """
    lines = step_text.split("\n")
    if not lines:
        return step_text
    # 先頭行: 既に *...* になっている部分はそのまま。それ以外の行の * を除去
    result = [lines[0]]
    for line in lines[1:]:
        # * で囲まれた部分を外す（*text* → text）。* が単数や不整合の場合は削除だけ
        line = re.sub(r"\*([^*]+)\*", r"\1", line)
        result.append(line)
    return "\n".join(result)


def _normalize_line_start(text: str) -> str:
    """各行の行頭の空白・タブを除去し、文頭を左揃えにする。"""
    if not text:
        return text
    return "\n".join(line.lstrip() for line in text.splitlines())


def _ensure_heading_marker(text: str) -> str:
    """番号付き見出しの先頭（1. 2. ...）に ■ がなければ付ける。"""
    if not text or not text.strip():
        return text
    # 先頭行が「数字. 」で始まり、まだ ■ がない場合のみ付与
    first_line = text.lstrip().split("\n")[0]
    if re.match(r"^\d+\.\s", first_line) and not first_line.strip().startswith("■"):
        return re.sub(r"^(\s*)(\d+\.\s)", r"\1■ \2", text, count=1)
    return text


def _answer_to_slack_blocks(answer: str) -> list[dict] | None:
    """
    回答テキストを Slack Block Kit の blocks に変換する。
    - 本文を「出典」前と「出典」以降に分割（mrkdwn 変換前に出典を分離すること）
    - 本文を番号付きステップ（1. 2. ...）で分割し、各ステップを section、区切りに divider を挿入
    - 出典はリンク付き section で末尾に追加
    - パースに失敗した場合や blocks が空の場合は None を返し、呼び出し側で通常の say(text=) にフォールバックする
    """
    if not answer or not answer.strip():
        return None

    # 出典セクションを「変換前」の本文から分離（**出典** は mrkdwn 変換で *出典* になるため、先に分ける）
    ref_marker = "**出典**"
    ref_marker_alt = "本回答は以下のドキュメントを参照しています。"
    body_raw = answer
    refs_text = ""
    if ref_marker in answer:
        i = answer.find(ref_marker)
        body_raw = answer[:i].rstrip()
        refs_text = answer[i:]
    elif ref_marker_alt in answer:
        i = answer.find(ref_marker_alt)
        body_raw = answer[:i].rstrip()
        refs_text = answer[i:]

    text = _markdown_to_slack_mrkdwn(body_raw)

    blocks: list[dict] = []

    # 本文をステップに分割。1改行以上 + 行頭の空白を許容して「(■)? 数字. 」の直前に区切る
    step_pat = re.compile(r"\n+(?=\s*(?:■\s*)?\d+\.\s)", re.MULTILINE)
    parts = step_pat.split(text)
    intro = ""
    steps: list[str] = []
    for i, p in enumerate(parts):
        s = p.strip()
        if not s:
            continue
        if re.match(r"^(■\s*)?\d+\.\s", s):
            steps.append(s)
        else:
            if not steps:
                intro = (intro + "\n\n" + s).strip() if intro else s
            else:
                steps[-1] = steps[-1] + "\n\n" + s

    def _add_section(t: str, is_step: bool = False) -> None:
        t = _normalize_line_start(t.strip())
        if not t:
            return
        if is_step:
            t = _ensure_heading_marker(t)
            t = _bold_heading_only(t)
        # 3000 文字を超える場合は複数 section に分割
        while len(t) > _SECTION_TEXT_MAX:
            chunk = t[:_SECTION_TEXT_MAX]
            # できるだけ改行で切る
            last_nl = chunk.rfind("\n")
            if last_nl > _SECTION_TEXT_MAX // 2:
                chunk = t[: last_nl + 1]
                t = t[last_nl + 1 :].lstrip()
            else:
                t = t[_SECTION_TEXT_MAX:].lstrip()
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _normalize_line_start(chunk)}})
        if t:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _normalize_line_start(t)}})

    if intro:
        _add_section(intro)
    for i, step in enumerate(steps):
        # 先頭ブロックの前に divider を置くと表示されないことがあるため、既にコンテンツがあるときだけ追加
        if i > 0 or intro:
            blocks.append({"type": "divider"})
        _add_section(step, is_step=True)

    # 出典: 「- https://...」を <url|短い表示> 形式の mrkdwn に
    if refs_text:
        ref_lines: list[str] = []
        for line in refs_text.splitlines():
            line = line.strip()
            if not line or line in ("**出典**", ref_marker_alt):
                continue
            if line.startswith("- "):
                url = line[2:].strip()
                if url.startswith("http"):
                    ref_lines.append(f"・ <{url}|{url}>")
                else:
                    ref_lines.append(f"・ {url}")
            else:
                ref_lines.append(line)
        if ref_lines:
            blocks.append({"type": "divider"})
            header = "本回答は以下のドキュメントを参照しています。"
            ref_body = "\n".join(ref_lines)
            _add_section(f"*出典*\n{header}\n\n{ref_body}")

    if not blocks:
        return None
    return blocks


def _say_blocks_chunked(say, blocks: list[dict], fallback_text: str, **kwargs) -> None:
    """
    blocks を最大 _SLACK_BLOCKS_MAX 個ずつに分けて、複数メッセージで送る。
    最初のメッセージには fallback_text、続きには「（続き）」を渡す。
    """
    for i in range(0, len(blocks), _SLACK_BLOCKS_MAX):
        chunk = blocks[i : i + _SLACK_BLOCKS_MAX]
        text = fallback_text if i == 0 else "（続き）"
        say(blocks=chunk, text=text, **kwargs)


# スレッドコンテキストとして使う過去メッセージの最大件数
_THREAD_CONTEXT_MAX_MESSAGES = 15


def _fetch_thread_context(client, channel: str, thread_ts: str, current_ts: str, bot_user_id: str | None) -> str:
    """
    スレッド内の会話を取得し、検索コンテキスト用の文字列にする。
    今回のメンション（current_ts）より前のメッセージを、直近 _THREAD_CONTEXT_MAX_MESSAGES 件まで取得する。
    """
    try:
        resp = client.conversations_replies(channel=channel, ts=thread_ts, limit=50)
        messages = resp.get("messages") or []
    except Exception:
        return ""
    lines = []
    for msg in messages:
        if msg.get("ts") == current_ts:
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        user_id = msg.get("user") or ""
        label = "Bot" if bot_user_id and user_id == bot_user_id else "ユーザー"
        lines.append(f"{label}: {text}")
    if not lines:
        return ""
    return "\n".join(lines[-_THREAD_CONTEXT_MAX_MESSAGES:])


def _build_answer(user_question: str, thread_context: str | None = None) -> str:
    """質問文から回答テキストを生成する。thread_context がある場合は検索クエリ拡張に利用する。エラー時はエラーメッセージを返す。"""
    try:
        expanded = expand_query_for_search(user_question, thread_context=thread_context)
        search_query = translate_to_english(expanded)
        search_query = format_query_for_search_documents(search_query)
        context_docs = asyncio.run(fetch_googlecloud_doc(search_query))
        logger.info("Successfully fetched documentation for user question")
        answer = generate_answer(user_question, context_docs)
        logger.info("Successfully generated answer for user question")
        return answer
    except (ValueError, OSError) as e:
        logger.error("Error generating answer: %s", e)
        return "エラーが発生しました。もう一度お試しください。"


@app.event("app_mention")
def handle_app_mention(event, say, client, context):
    """
    Bot へのメンション（@Bot 質問）に反応する。そのスレッドに返信する。
    同じスレッド内の過去のやり取りをコンテキストとして検索に利用する。
    """
    text = event.get("text", "").strip()
    # メンション部分 <@Uxxxxx> を除去
    user_question = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
    if not user_question:
        return
    channel = event.get("channel", "")
    thread_ts = event.get("thread_ts") or event.get("ts")
    current_ts = event.get("ts", "")
    bot_user_id = getattr(context, "bot_user_id", None)
    try:
        thread_context = _fetch_thread_context(client, channel, thread_ts, current_ts, bot_user_id) or None
        answer = _build_answer(user_question, thread_context=thread_context)
        blocks = _answer_to_slack_blocks(answer)
        if blocks:
            fallback = (answer.strip()[:200] + "…") if len(answer) > 200 else answer.strip()
            _say_blocks_chunked(
                say, blocks, fallback or "Google Cloud ドキュメントの回答",
                thread_ts=thread_ts, unfurl_links=False, unfurl_media=False,
            )
        else:
            say(text=_markdown_to_slack_mrkdwn(answer), mrkdwn=True, thread_ts=thread_ts, unfurl_links=False, unfurl_media=False)
    except Exception as e:
        logger.exception("Error in app_mention handler: %s", e)
        say(
            text=":warning: 回答の生成中にエラーが発生しました。しばらくしてからもう一度お試しください。",
            thread_ts=thread_ts,
        )


@app.message()
def handle_message(message, say):
    """
    通常のメッセージに反応する（Bot が参加しているチャンネル／DM）。
    """
    text = message.get("text", "").strip()
    if not text:
        return
    try:
        answer = _build_answer(text)
        blocks = _answer_to_slack_blocks(answer)
        if blocks:
            fallback = (answer.strip()[:200] + "…") if len(answer) > 200 else answer.strip()
            _say_blocks_chunked(say, blocks, fallback or "Google Cloud ドキュメントの回答")
        else:
            say(text=_markdown_to_slack_mrkdwn(answer), mrkdwn=True)
    except Exception as e:
        logger.exception("Error in message handler: %s", e)
        say(text=":warning: 回答の生成中にエラーが発生しました。しばらくしてからもう一度お試しください。")


def run():
    """
    Run the application with Socket Mode.
    """
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not app_token:
        raise ValueError("SLACK_APP_TOKEN is not set")
    handler = SocketModeHandler(app, app_token)
    handler.start()
    logger.info("Application started")


if __name__ == "__main__":
    run()
