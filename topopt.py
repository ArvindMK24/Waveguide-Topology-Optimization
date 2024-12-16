
from __future__ import division
from mmapy import mmasub, kktcheck
from util import setup_logger
from typing import Tuple
import numpy as np
import os


def main() -> None:
    
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "mma_opti.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    

    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    
   
    m, n = 1, 5 # 5 design variables (sigma in our case, which will modify the radius of disc) and one constraint
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xval = 5 * eeen
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = eeen.copy()
    xmax = 10 * eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    innerit = 0
    outeriter = 0
    maxoutit = 11
    kkttol = 0
    
    if outeriter == 0:
        f0val, df0dx, fval, dfdx = opti(xval)
        outvector1 = np.array([outeriter, innerit, f0val, fval])
        outvector2 = xval.flatten()
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}\n".format(outvector2))
    

    kktnorm = kkttol + 10 # iteration starts
    outit = 0

    while kktnorm > kkttol and outit < maxoutit:
        outit += 1
        outeriter += 1
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move)
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        f0val, df0dx, fval, dfdx = opti(xval)
        residu, kktnorm, residumax = kktcheck(
            m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
        
        outvector1 = np.array([outeriter, innerit, f0val, fval])
        outvector2 = xval.flatten()
        
        logger.info("outvector1 = {}".format(outvector1)) #Reminder for Arvind 14-12-2024 - Update util library in office PC and MIT Laptop
        logger.info("outvector2 = {}".format(outvector2)) #Reminder for Arvind 14-12-2024 - Update util library in office PC and MIT Laptop
        logger.info("kktnorm    = {}\n".format(kktnorm))  #Reminder for Arvind 14-12-2024 - Update util library in office PC and MIT Laptop
    
    logger.info("Finished")


def opti(xval: np.ndarray) -> Tuple[float, np.ndarray, float, np.ndarray]: #Defining the power transmission coefficient and constraints for given model
    nx = 5
    eeen = np.array([[5,5,5,5,5]]).T
    eeen1 = np.array([[10,10,10,10,10]]).T
    zin = 0.120
    zout = 0.576
    rmax = 15
    rmin = 10
    c1 = 12940848
    ze = c1 * (eeen.T* xval + eeen1.T)
    tau = 2/(ze*zout)
    f0val = 0
    df0dx = c1 * eeen
    fval = (np.dot(eeen1.T, eeen) - c1).item()
    dfdx = -3 * (eeen1.T * eeen).T
    return f0val, df0dx, fval, dfdx


if __name__ == "__main__":
    main()
