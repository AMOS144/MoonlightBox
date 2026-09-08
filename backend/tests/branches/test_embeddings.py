from pathlib import Path


def test_local_chinese_embedder_returns_normalized_vectors() -> None:
    from moonlightbox.branches.embeddings import LocalChineseEmbedder

    model_root = Path(__file__).parents[3] / "models" / "embeddings" / "fastembed-bge-small-zh-v1.5"
    vectors = LocalChineseEmbedder(model_root).embed(["想你了", "今天工作很累"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 512
    assert abs(sum(value * value for value in vectors[0]) - 1.0) < 0.001
