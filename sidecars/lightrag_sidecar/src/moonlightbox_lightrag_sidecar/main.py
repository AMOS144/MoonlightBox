import uvicorn

from moonlightbox_lightrag_sidecar.config import SidecarSettings


def main() -> None:
    settings = SidecarSettings()
    uvicorn.run(
        "moonlightbox_lightrag_sidecar.app:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
