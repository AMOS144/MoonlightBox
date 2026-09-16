"""在临时 workspace 副本上验证真实关键词/embedding 和 ASGI HTTP 契约。"""

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from moonlightbox_lightrag_sidecar.app import create_app
from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.core import CoreLightRAGRegistry
from moonlightbox_lightrag_sidecar.source_index import SourceIndex


async def run(workspace, index_dir, env_file):
    settings = SidecarSettings(_env_file=env_file)
    settings.require_model_keys()
    files = [p for p in workspace.iterdir() if p.suffix in {".json", ".graphml"}]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    temp = Path(tempfile.mkdtemp(prefix="moonlight-temporal-online-"))
    destination = temp / workspace.name
    shutil.copytree(workspace, destination)
    shutil.copy2(index_dir / "moonlight_source_index.sqlite3", destination)
    settings = settings.model_copy(update={"storage_dir": temp, "llm_timeout_seconds": 90})
    registry = CoreLightRAGRegistry(settings)
    app = create_app(settings=settings, registry=registry)
    docs = SourceIndex(destination).read()
    source_version = next(iter(docs.values())).source_version
    orders = {m.source_ordinal for d in docs.values() for m in d.message_spans}
    included = sorted(orders)[len(orders) // 2]
    results = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://sidecar",
        headers={"Authorization": "Bearer " + settings.api_token.get_secret_value()},
    ) as client:
        for period in ["before", "after"]:
            started = time.monotonic()
            response = await asyncio.wait_for(
                client.post(
                    f"/v1/workspaces/{workspace.name}/query",
                    json={
                        "query": "洪欣羽的工作安排与下班时间有什么线索？",
                        "top_k": 8,
                        "chunk_top_k": 8,
                        "temporal": {
                            "source_version": source_version,
                            "included_count": included,
                            "period": period,
                        },
                    },
                ),
                timeout=150,
            )
            if response.status_code != 200:
                return {
                    "passed": False,
                    "http_status": response.status_code,
                    "period": period,
                    "temporary_workspace": str(destination),
                }
            payload = response.json()
            assert payload["references"], "真实查询没有返回片段"
            assert payload["temporal_diagnostics"]["mapping_complete"]
            results.append(
                {
                    "period": period,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "returned_chunks": len(payload["references"]),
                    "diagnostics": payload["temporal_diagnostics"],
                }
            )
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {
        "passed": True,
        "online_models_called": True,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "original_assets_unchanged": True,
        "temporary_workspace": str(destination),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.workspace, args.index_dir, args.env_file)),
            ensure_ascii=False,
            indent=2,
        )
    )
