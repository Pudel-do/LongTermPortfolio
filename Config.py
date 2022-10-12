import os
import json

class Config:
    def __init__(self, config_name):
        self.config_name = config_name
        self.config = self.load()
        
    def load(self):
        with open (self.config_name) as conf:
            config = json.load(conf)
        return config
    



