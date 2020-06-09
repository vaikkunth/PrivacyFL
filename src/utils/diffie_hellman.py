from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
import random

def keygeneration(n, party_i):
    pkey_list = []
    skey_list = []
    for i in range(n):
        if  i == party_i:
            pkey_list.append(0)
            skey_list.append(0)
        else: 
            sk = ec.generate_private_key(ec.SECP384R1(), default_backend())
            pk = sk.public_key()
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
            shared_key = my_skey_list[i].exchange(ec.ECDH(), other_pkey_list[i])
            #Hash the common keys
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'handshake data',
                backend=default_backend()
            ).derive(shared_key)
            common_key = int.from_bytes(derived_key, byteorder='big')
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