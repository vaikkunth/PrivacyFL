from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
import random

def keygeneration(num, ip): # num - number of parties ; ip - ith party
    publicKeyList = []
    secretKeyList = []
    for i in range(num):
        if  i == ip:
            publicKeyList.append(0)
            secretKeyList.append(0)
        else: 
            secretKey = ec.generate_private_key(ec.SECP384R1(), default_backend())
            publicKey = secretKey.public_key()
            publicKeyList.append(publicKey)
            secretKeyList.append(secretKey)
    return  publicKeyList,secretKeyList 

def keyexchange(num, ip, selfPublicKeys, selfSecretKeys, othersPublicKeys):
    exchangeKeys = []
    for i in range(num):
        if i == ip:
            exchangeKeys.append(0)
        else:
            shareKey = selfSecretKeys[i].exchange(ec.ECDH(), othersPublicKeys[i])
            #Hashing the keys
            newKey = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'handshake data',
                backend=default_backend()
            ).derive(shareKey)
            keyCom = int.from_bytes(newKey, byteorder='big')
            exchangeKeys.append(keyCom)
    return exchangeKeys


def randomize(sd):
        random.seed(sd)
        rand            = random.getrandbits(256*2)
        randBin      = bin(rand)
        appendZeros = 256 - (len(randBin) - 2)
        r          = '0' * appendZeros + randBin[2:]
        # first portion - mask the inputs ; second portion - seed for PRG
        second = int(r[0:256], 2)
        first = int(r[256:] , 2)
        return first, second 


def randomize_all(ip, exchangeKeys, div):
    
    for i in range(len(exchangeKeys)):
        if i == ip:
             continue
        exchangeKeys[i], party = randomize( exchangeKeys[i])
        
    return exchangeKeys, party
