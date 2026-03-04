from src.engine import translate_to_english, generate_answer
from src.mcp_client import fetch_googlecloud_doc
import asyncio


async def main():
    q = "Cloud Storage bucketの作り方は?"
    search_query = translate_to_english(q)
    context_docs = await fetch_googlecloud_doc(search_query)
    answer = generate_answer(q, context_docs)
    print(answer)


asyncio.run(main())
