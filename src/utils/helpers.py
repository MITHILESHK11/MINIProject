def load_config(path="config.yaml"):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
