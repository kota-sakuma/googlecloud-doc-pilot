"""Logic for the Developer Knowledge API (MCP)"""

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

MAX_DOCS_TO_FETCH = 20
MCP_ENDPOINT = "https://developerknowledge.googleapis.com/mcp"


def _get_api_key() -> str:
    """
    Get the API key for the Developer Knowledge API.
    """
    api_key = os.getenv("DEVELOPER_KNOWLEDGE_API_KEY")
    if not api_key:
        raise ValueError("DEVELOPER_KNOWLEDGE_API_KEY is not set")
    return api_key


async def _call_tool(
    tool_name: str,
    arguments: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    """
    Call a tool with the given name and arguments.
    """
    payload = {
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "jsonrpc": "2.0",
        "id": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Goog-Api-Key": api_key,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(MCP_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError("MCP error: %s" % data["error"])

    result = data.get("result", {})
    content = result.get("content", [])
    if content and isinstance(content[0].get("text"), str):
        return json.loads(content[0]["text"])

    return result


async def fetch_googlecloud_doc(query: str) -> str:
    """
    Fetch Google Cloud documentation for a given query.
    """
    api_key = _get_api_key()

    # 1. search_documents
    search_result = await _call_tool(
        "search_documents",
        arguments={"query": query},
        api_key=api_key,
    )
    results = search_result.get("results") or []
    if not results:
        return f"# 検索: {query}\n\n該当するドキュメントは見つかりませんでした｡"

    # 2. parent を重複除いて最大 20 件取得し、get_documents で全文を取得（ツール名は get_documents、引数は names 配列）
    parents = list(dict.fromkeys(r.get("parent") for r in results if r.get("parent")))[
        :MAX_DOCS_TO_FETCH
    ]
    parts = [f"# 検索: {query}\n"]

    if parents:
        try:
            get_result = await _call_tool(
                "get_documents",
                arguments={"names": parents},
                api_key=api_key,
            )
            documents = get_result.get("documents") or []
            for doc in documents:
                uri = doc.get("uri", "")
                content = doc.get("content", "")
                if uri:
                    parts.append(f"## 出典: {uri}\n\n{content}\n")
                else:
                    parts.append(f"## 出典: {doc.get('name', '')}\n\n{content}\n")
            if documents:
                return "\n".join(parts)
        except Exception:
            pass

    # get_documents が使えない、または結果が空の場合は検索スニペットで組み立てる
    for r in results[:MAX_DOCS_TO_FETCH]:
        parent = r.get("parent", "")
        content = r.get("content", "")
        if parent:
            parts.append(f"## 出典: {parent}\n\n{content}\n")
        else:
            parts.append(f"## スニペット\n\n{content}\n")
    return "\n".join(parts)
