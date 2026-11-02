# Runs inside Pyodide. Glue between the page and the gopniya package.
import json

from pyodide.http import pyfetch

from gopniya.agent import RedactionAgent


async def _cloud(messages):
    resp = await pyfetch(
        "/api/llm",
        method="POST",
        headers={"Content-Type": "application/json"},
        body=json.dumps({"messages": messages}),
    )
    data = await resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["answer"]


async def preview(text):
    agent = RedactionAgent(llm=None)
    res = await agent.run(text)
    return json.dumps(res.to_dict(), ensure_ascii=False)


async def run(text, task):
    agent = RedactionAgent(llm=_cloud)
    res = await agent.run(text, task)
    return json.dumps(res.to_dict(), ensure_ascii=False)
