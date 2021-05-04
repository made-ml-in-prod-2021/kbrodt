from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    test_size: float = field(default=0.1)
    seed: int = field(default=314159)
