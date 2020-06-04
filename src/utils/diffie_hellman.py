

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import pandas as pd
import numpy as np
import math
import nacl.bindings as xc
import random
# Generate some parameters. These can be reused.
#parameters = dh.generate_parameters(generator=2, key_size=2048,
                                   backend=default_backend())
#Generate a private key for use in the exchange.
#private_key = parameters.generate_private_key()
# In a real handshake the peer_public_key will be received from the
# other party. For this example we'll generate another private key and
 # get a public key from that. Note that in a DH handshake both peers
 # must agree on a common set of parameters.
# peer_public_key = parameters.generate_private_key().public_key()
# shared_key = private_key.exchange(peer_public_key)
# Perform key derivation.
# derived_key = HKDF(
#    algorithm=hashes.SHA256(),length=32,salt=None,info=b'handshake data',backend=default_backend()
# ).derive(shared_key)
 # For the next handshake we MUST generate another private key, but
 # we can reuse the parameters.
#private_key_2 = parameters.generate_private_key()
#peer_public_key_2 = parameters.generate_private_key().public_key()
#shared_key_2 = private_key_2.exchange(peer_public_key_2)
# derived_key_2 = HKDF(algorithm=hashes.SHA256(),length=32,salt=None,info=b'handshake data',backend=default_backend()
# ).derive(shared_key_2)
    
    

def keygeneration(n, ip): #ith party - ip
    assert(ip < n) ## Anton code
    publickey_list = []
    secretkey_list = []
    for i in range(n):
        if  i == ip:
            publickey_list.append(0)
            secretkey_list.append(0)
        else: 
            pubkey, secretkey = xc.crypto_kx_keypair()
            publickey_list.append(pubkey)
            secretkey_list.append(secretkey)
    return  publickey_list,secretkey_list 

def keyexchange(n, ip, publickey_list, secretkey_list, extra_list):
    exchangeKey = []
    for i in range(n):
        if i == ip:
            exchangeKey.append(0)
        else:
            if i > ip:
                comKeyint, _ = xc.crypto_kx_client_session_keys(publickey_list[i], secretkey_list[i], extra_list[i])
            else:  
                _, comKeyint = xc.crypto_kx_server_session_keys(publickey_list[i], secretkey_list[i], extra_list[i])
            exchangekey = int.from_bytes(xc.crypto_hash_sha256(comKeyint), byteorder='big')
            exchangeKey.append(exchangekey)
    return exchangeKey


def randomize( s, m, clientsign):
        random.seed(s)
        rand            = random.getrandbits(256*2)
        randBin      = bin(rand)
        zeros = 256 - (len(randBin) - 2)
        randR          = '0' * zeros + randBin[2:]
        first = int(randR[0:256], 2)
        sec = int(randR[256:] , 2)
        return first, sec 


def randomize_all(ip, comKey, m):
    
    for i in range(len(comKey)):
        if i == ip:
             continue
        clientsign = 1 if i > ip else -1
        comKey[i], client = randomize( comKey[i], m, clientsign)
        
    return comKey, client
