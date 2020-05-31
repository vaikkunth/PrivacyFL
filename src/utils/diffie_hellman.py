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
            pk, sk = nb.crypto_kx_keypair()
            publickey_list.append(pk)
            secretkey_list.append(sk)
    return  publickey_list,secretkey_list 

def keyexchange(n, ip, publickey_list, secretkey_list, extra_list):
    com_key_list = []
    for i in range(n):
        #Generate DH keys 
        if i == ip:
            com_key_list.append(0)
        else:
            if i > ip:
                com_key_raw, _ = nb.crypto_kx_client_session_keys(publickey_list[i], secretkey_list[i], extra_list[i])
            else:  
                _, com_key_raw = nb.crypto_kx_server_session_keys(publickey_list[i], secretkey_list[i], extra_list[i])
            #Hash the common keys
            com_key = int.from_bytes(nb.crypto_hash_sha256(com_key_raw), byteorder='big')
            com_key_list.append(com_key)
    return com_key_list


#PRG

def randomize( r, modulo, clientsign):
        random.seed(r)
        rand            = random.getrandbits(256*2)
        rand_b_raw      = bin(rand)
        nr_zeros_append = 256 - (len(rand_b_raw) - 2)
        rand_b          = '0' * nr_zeros_append + rand_b_raw[2:]
        # first half used to mask the inputs and second half as the next seed to the pseudorandom generator
        R = int(rand_b[0:256], 2)
        r = int(rand_b[256:] , 2)
        return r, R 


def randomize_all(ip, common_key_list, modulo):
    
    for i in range(len(common_key_list)):
        if i == ip:
             continue
        clientsign = 1 if i > ip else -1
        common_key_list[i], client = randomize( common_key_list[i], modulo, clientsign)
        
    return common_key_list, client
