from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, agent_number, agent_type):
        self.agent_number = agent_number
        self.agent_type = agent_type  # 'CLIENT' or 'SERVER'
        self.directory = None  # will be added after initialization

    @property
    def name(self):
        return str(self.agent_type) + str(self.agent_number)

    def set_directory(self, directory):
        self.directory = directory