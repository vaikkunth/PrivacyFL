class Message:
    """
    Used for all client-client and client-server communications
    """
    def __init__(self, sender_name, recipient_name, body):
        """
        :param sender_name: name of sender
        :param recipient_name: name of recipient
        :param body: Content depends no the message being sent.
        """
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"