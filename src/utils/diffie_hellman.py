import nacl.bindings as nb
import random
import pandas as pd
import numpy as np
import math

def keygeneration(n, party_i):
    assert(party_i < n) ## Anton code
    pkey_list = []
    skey_list = []
    for i in range(n):
        if  i == party_i:
            pkey_list.append(0)
            skey_list.append(0)
        else: 
            pk, sk = nb.crypto_kx_keypair()
            pkey_list.append(pk)
            skey_list.append(sk)
    return  pkey_list,skey_list 

def keyexchange(n, party_i, my_pkey_list, my_skey_list, other_pkey_list):
    common_key_list = []
    for i in range(n):
        #Generate DH (common) keys 
        if i == party_i:
            common_key_list.append(0)
        else:
            if i > party_i:
                common_key_raw, _ = nb.crypto_kx_client_session_keys(my_pkey_list[i], my_skey_list[i], other_pkey_list[i])
            else:  
                _, common_key_raw = nb.crypto_kx_server_session_keys(my_pkey_list[i], my_skey_list[i], other_pkey_list[i])
            #Hash the common keys
            common_key = int.from_bytes(nb.crypto_hash_sha256(common_key_raw), byteorder='big')
            common_key_list.append(common_key)
    return common_key_list


#PRG

def randomize( r, modulo, clientsign):
        # Call the double lenght pseudorsndom generator
        random.seed(r)
        rand            = random.getrandbits(256*2)
        rand_b_raw      = bin(rand)
        nr_zeros_append = 256 - (len(rand_b_raw) - 2)
        rand_b          = '0' * nr_zeros_append + rand_b_raw[2:]
        # Use first half to mask the inputs and second half as the next seed to the pseudorsndom generator
        R = int(rand_b[0:256], 2)
        r = int(rand_b[256:] , 2)
        return r, R 


def randomize_all(party_i, common_key_list, modulo):
    
    for i in range(len(common_key_list)):
        if i == party_i:
             continue
        clientsign = 1 if i > party_i else -1
        common_key_list[i], client = randomize( common_key_list[i], modulo, clientsign)
        
    return common_key_list, client