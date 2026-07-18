from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOONLIGHTBOX_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/moonlightbox.db"
    chroma_dir: Path = Path("data/chroma")
    model_dir: Path = Path("models")
    auto_create_schema: bool = False
    cloud_evaluation_enabled: bool = True
    cloud_evaluation_endpoint: str = "https://api.openai.com/v1/chat/completions"
    cloud_evaluation_model: str = "gpt-4.1-mini"
    cloud_evaluation_api_key: str | None = None

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.chroma_dir, self.model_dir):
            directory.mkdir(parents=True, exist_ok=True)
