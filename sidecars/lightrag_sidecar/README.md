# MoonlightBox LightRAG Sidecar

This service keeps LightRAG and its storage dependencies outside the main
MoonlightBox Python environment. It exposes a deliberately small API:

- `POST /v1/workspaces/{workspace}/documents:batch`
- `POST /v1/workspaces/{workspace}/query`
- `GET /health`

Every MoonlightBox graph version receives a different validated LightRAG
workspace. Queries can therefore never see documents from another project or
from a later graph version.

The query endpoint always enables `only_need_context`; profile compilation is
performed by MoonlightBox, not by an untracked final answer from LightRAG.

For local WSL2 development:

```bash
cd sidecars/lightrag_sidecar
uv sync
cp .env.example .env
uv run moonlightbox-lightrag-sidecar
```

第一版推荐使用百炼华北 2（北京）的 `text-embedding-v4`，保持默认 1024 维。在 `.env`
中把 `YOUR_WORKSPACE_ID` 替换为百炼业务空间 ID，并把该地域的 API Key 填入
`LIGHTRAG_SIDECAR_EMBEDDING_API_KEY`。聊天 Bundle 的文本会发送给百炼生成向量。

这些密钥职责不同：

- `LIGHTRAG_SIDECAR_LLM_API_KEY`：LightRAG 实体和关系抽取模型的服务商密钥；
- `LIGHTRAG_SIDECAR_EMBEDDING_API_KEY`：百炼 Embedding 的服务商密钥；
- `LIGHTRAG_SIDECAR_API_TOKEN`：MoonlightBox 访问本地 Sidecar 的 bearer token，不是云模型密钥。

人物档案编译由 MoonlightBox 主服务执行，使用根目录 `.env` 中的
`MOONLIGHTBOX_NODE_ANALYSIS_API_KEY`。如果档案编译和实体抽取都使用 DeepSeek，两项可以
复用同一个 DeepSeek API Key；Embedding 仍使用单独的百炼 API Key。
