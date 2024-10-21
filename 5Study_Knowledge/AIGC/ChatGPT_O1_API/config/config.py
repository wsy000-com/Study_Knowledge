import torch, sys
import configparser

class Config:
    def __init__(self):
        settings = configparser.ConfigParser()
        settings.read('conf/config.ini')
        self.api_key = settings["openai"]["api_key"]
        self.base_url = settings["openai"]["base_url"]
        self.model_name = settings["openai"]["model_name"]

config = Config()