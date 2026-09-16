import uvicorn
import json
import os
from pathlib import Path

from moonlightbox_lightrag_sidecar.config import SidecarSettings


def main() -> None:
    # 后端设置页写入共享运行目录；启动时覆盖同名环境变量，避免必须手工复制密钥。
    path = Path(os.environ.get("MOONLIGHTBOX_DATA_DIR", "data")) / "lightrag-model-settings.json"
    if path.exists():
        for key, env_name in (("llm_model", "LIGHTRAG_SIDECAR_LLM_MODEL"), ("llm_endpoint", "LIGHTRAG_SIDECAR_LLM_BASE_URL"), ("llm_api_key", "LIGHTRAG_SIDECAR_LLM_API_KEY"), ("embedding_model", "LIGHTRAG_SIDECAR_EMBEDDING_MODEL"), ("embedding_endpoint", "LIGHTRAG_SIDECAR_EMBEDDING_BASE_URL"), ("embedding_api_key", "LIGHTRAG_SIDECAR_EMBEDDING_API_KEY"), ("embedding_dimension", "LIGHTRAG_SIDECAR_EMBEDDING_DIMENSION")):
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
            if value not in (None, ""):
                os.environ[env_name] = str(value)
    settings = SidecarSettings()
    uvicorn.run(
        "moonlightbox_lightrag_sidecar.app:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
