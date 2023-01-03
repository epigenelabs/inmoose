import numpy as np
from warnings import warn
from .aveLogCPM import aveLogCPM
from .dispCoxReid import dispCoxReid
from .validDGEList import validDGEList

def estimateGLMCommonDisp_DGEList(self, design=None, method="CoxReid", subset=10000, verbose=False):
    """
    Modifies self in place
    """
    y = validDGEList(self)
    AveLogCPM = y.aveLogCPM(dispersion=0.05)

    disp = estimateGLMCommonDisp(y=y.counts, design=design, offset=y.getOffset(), method=method, subset=subset, AveLogCPM=AveLogCPM, verbose=verbose, weights=y.weights)

    y.common_dispersion = disp
    y.AveLogCPM = y.aveLogCPM(dispersion=disp)
    return y

def estimateGLMCommonDisp(y, design=None, offset=None, method="CoxReid", subset=10000, AveLogCPM=None, verbose=False, weights=None):
    """
    Estimate common dispersion in a GLM
    """
    #Check design
    if design is None:
        design = np.ones((y.shape[1], 1))
    if design.shape[1] >= y.shape[1]:
        warn("No residual degree of freedom: setting dispersion to None")
        return None

    # Check method
    if method != "CoxReid" and weights is not None:
        warn("weights only supported by CoxReid method")

    # Check offset
    if offset is None:
        offset = np.log(y.sum(axis=0))

    # Check AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y, offset, weights)

    # Call lower-level function
    if method == "CoxReid":
        disp = dispCoxReid(y, design=design, offset=offset, subset=subset, AveLogCPM=AveLogCPM, weights=weights)
    elif method == "Pearson":
        disp = dispPearson(y, design=design, offset=offset, subset=subset, AveLogCPM=AveLogCPM)
    elif method == "deviance":
        disp = dispDeviance(y, design=design, offset=offset, subset=subset, AveLogCPM=AveLogCPM)
    else:
        raise ValueError(f"invalid method for dispersion evaluation: {method}")

    if verbose:
        print("Disp =", round(disp, 5), ", BCV =", round(sqrt(disp), 4))

    return disp
