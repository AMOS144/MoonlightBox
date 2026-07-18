import random
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel


class BlindPairView(BaseModel):
    id: str
    prompt: str
    options: dict[str, str]
    model_version_id: str


@dataclass(frozen=True)
class BlindPair:
    id: str
    prompt: str
    options: dict[str, str]
    real_option: str
    model_option: str
    model_version_id: str

    def public_view(self) -> BlindPairView:
        return BlindPairView(
            id=self.id,
            prompt=self.prompt,
            options=self.options,
            model_version_id=self.model_version_id,
        )


def make_blind_pair(
    prompt: str,
    real_answer: str,
    model_answer: str,
    model_version_id: str,
    rng: random.Random | None = None,
) -> BlindPair:
    randomizer = rng or random.Random()
    real_option = randomizer.choice(["a", "b"])
    model_option = "b" if real_option == "a" else "a"
    return BlindPair(
        id=str(uuid4()),
        prompt=prompt,
        options={real_option: real_answer, model_option: model_answer},
        real_option=real_option,
        model_option=model_option,
        model_version_id=model_version_id,
    )
