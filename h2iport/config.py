from dotmap import DotMap
from ruamel.yaml import YAML

from h2iport.paths import DATA_DIR


def read_yaml_dict(fp, typ="safe"):
    yaml = YAML(typ=typ)
    return yaml.load(fp)


def get_dotmap_from_yml(fp):
    return DotMap(read_yaml_dict(fp), _dynamic=False)


class Config:
    cf = get_dotmap_from_yml(DATA_DIR / "config.yml")
    cf.consumer_data = get_dotmap_from_yml(DATA_DIR / "consumer_data.yml")
