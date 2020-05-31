import pandas as pd
import numpy as np
import math
import nacl.bindings as nb
import random


def keygeneration(n, ip): #ith party - ip
    assert(ip < n) ## Anton code
    publickey_list = []
    secretkey_list = []
    for i in range(n):
        if  i == ip:
            publickey_list.append(0)
            secretkey_list.append(0)
        else: 
            pubkey, secretkey = nb.crypto_kx_keypair()
            publickey_list.append(pubkey)
            secretkey_list.append(secretkey)
    return  publickey_list,secretkey_list 

def keyexchange(n, ip, publickey_list, secretkey_list, extra_list):
    exchangeKey = []
    for i in range(n):
        #Generate DH keys 
        if i == ip:
            exchangeKey.append(0)
        else:
            if i > ip:
                comKeyint, _ = nb.crypto_kx_client_session_keys(publickey_list[i], secretkey_list[i], extra_list[i])
            else:  
                _, comKeyint = nb.crypto_kx_server_session_keys(publickey_list[i], secretkey_list[i], extra_list[i])
            #Hashing the common keys
            exchangekey = int.from_bytes(nb.crypto_hash_sha256(comKeyint), byteorder='big')
            exchangeKey.append(exchangekey)
    return exchangeKey


#PRG

def randomize( s, modulo, clientsign):
        random.seed(s)
        rand            = random.getrandbits(256*2)
        randBin      = bin(rand)
        zeros = 256 - (len(randBin) - 2)
        randR          = '0' * zeros + randBin[2:]
        first = int(randR[0:256], 2)
        sec = int(randR[256:] , 2)
        return first, sec 


def randomize_all(ip, comKey, modulo):
    
    for i in range(len(comKey)):
        if i == ip:
             continue
        clientsign = 1 if i > ip else -1
        comKey[i], client = randomize( comKey[i], modulo, clientsign)
        
    return comKey, client
