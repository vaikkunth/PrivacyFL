import datetime
from datetime import timedelta


class Latencies:
    client1 = {'client2': timedelta(seconds=1), 'client3': timedelta(seconds=4),
               'server1': timedelta(seconds=10)}
    client2 = {'client1': timedelta(seconds=1), 'client3': timedelta(seconds=3),
               'server1': timedelta(seconds=7)}
    client3 = {'client1': timedelta(seconds=4), 'client2': timedelta(seconds=3),
               'server1': timedelta(seconds=10)}

    
