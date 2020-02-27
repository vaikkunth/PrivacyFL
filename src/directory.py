class Directory:
    """
    Contains mappings for client and server names to instances in simulation
    """
    def __init__(self, clients, server_agents):
        self.clients = clients
        self.server_agents = server_agents
        self.all_agents = {**self.clients, **self.server_agents}