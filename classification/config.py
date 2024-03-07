import json

CONFIG_JSON_PATH = "./config.json"


class Config:
    def __init__(self) -> None:
        with open(CONFIG_JSON_PATH, "r") as f:
            conf = json.load(f)

        self.data = conf["data"]
        self.model = conf["model"]
