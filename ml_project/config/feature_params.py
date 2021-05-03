from dataclasses import dataclass, field
from typing import List, Union


@dataclass()
class PipelineParams:
    name: str
    params: dict = field(default_factory=dict)


@dataclass()
class ColumnTransformerParams:
    name: str
    pipelines: Union[str, List[PipelineParams]]
    columns: List[str]


@dataclass()
class FeatureParams:
    feature_pipelines: List[ColumnTransformerParams]
    target_col: str
