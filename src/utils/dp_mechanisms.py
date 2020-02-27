from numbers import Real
import random
import numpy as np


"""
WARNING: only laplace will produce reproducible results since the other DP mechanisms 
included here using numpy's random module which is not thread-safe for randomness. 
"""
def laplace(mean, sensitivity, epsilon): # mean : value to be randomized (mean)
        scale = sensitivity / epsilon
        rand = random.uniform(0,1) - 0.5 # rand : uniform random variable
        return mean - scale * np.sign(rand) * np.log(1 - 2 * np.abs(rand))


def boundedLaplace(mean, sensitivity, epsilon, delta):
    #epsion>0 and delta >0 and <0.5
    
    scale = sensitivity / epsilon
    
    def cdf(mean):
        
        if scale == 0:
            if mean < 0:
                return 0 
            else:
                return 1

        if mean < 0:
            return 0.5 * np.exp(mean / scale)

        return 1 - 0.5 * np.exp(-mean / scale)
    
    if(scale==0):
        noiseBound = -1
    else: 
        noiseBound=scale * np.log(1 + (np.exp(epsilon) - 1) / 2 / delta)

        rand = random()
        rand *= cdf(noiseBound) - cdf(-noiseBound)
        rand += cdf(-noiseBound)
        rand -= 0.5

    return mean - scale * (np.sign(rand) * np.log(1 - 2 * np.abs(rand)))

def staircase(mean,sensitivity,epsilon,gamma):
    # 0 <= gamma <= 1 , delta=0, epsilon>0

    #gamma = 1 / (1 + np.exp(epsilon / 2))
            
    if random() < 0.5: 
        sign = -1 
    else:
        sign = 1
        
    geoRand = geometric(1 - np.exp(-epsilon)) - 1 # geoRand : geometric Random Variable
    rand = random() # Uniform random variable
        
    if random() < gamma / (gamma + (1 - gamma) * np.exp(-epsilon)):
        binRand = 0 # binary random variable
    else: 
        binRand = 1

    return mean + sign * ((1 - binRand) * ((geoRand + gamma * rand) * sensitivity) +
                               binRand * ((geoRand + gamma + (1 - gamma) * rand) *
                                            sensitivity))

def gaussian(mean, sensitivity, epsilon, delta ):
    
    scale = np.sqrt(2 * np.log(1.25 /delta)) * sensitivity / epsilon
    randUnif1 = random()
    randUnif2 = random()

    gaussian = np.sqrt(- 2 * np.log(randUnif1)) * np.sin(2 * np.pi * randUnif2)
    stdNormal = np.sqrt(- 2 * np.log(randUnif1)) * np.cos(2 * np.pi * randUnif2)
      
    return stdNormal * scale + mean


