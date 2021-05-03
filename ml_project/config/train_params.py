from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegression")
    seed: int = field(default=314159)
