import yaml

from clfit.config import build_config
from clfit.config.config import ConfigSchema


def test_build_config(tmp_path, config_dict):
    path = tmp_path / "config.yml"
    with open(path, "w") as fout:
        yaml.dump(config_dict, fout)

    config = build_config(path)
    config = ConfigSchema().dump(config)

    def is_subset(subset, superset):
        if isinstance(subset, dict):
            return all(
                key in superset and is_subset(val, superset[key])
                for key, val in subset.items()
            )

        if isinstance(subset, list) or isinstance(subset, set):
            return all(
                any(is_subset(subitem, superitem) for superitem in superset)
                for subitem in subset
            )

        # assume that subset is a plain value if none of the above match
        return subset == superset

    assert is_subset(config_dict, config)
