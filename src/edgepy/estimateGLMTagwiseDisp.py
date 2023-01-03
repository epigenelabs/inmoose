import numpy as np

from .aveLogCPM import aveLogCPM
from .dispCoxReidInterpolateTagwise import dispCoxReidInterpolateTagwise

def estimateGLMTagwiseDisp_DGEList(self, design=None, prior_df=10, trend=None, span=None):
    """
    Modifies self in place
    """
    # Find appropriate dispersion
    if trend is None:
        trend = self.trended_dispersion is not None
    if trend:
        dispersion = self.trended_dispersion
        if dispersion is None:
            raise ValueError("No trended_dispersion in data object. Run estimateGLMTrendedDisp first.")
    else:
        dispersion = self.common_dispersion
        if dispersion is None:
            raise ValueError("No common_dispersion found in data object. Run estimateGLMCommonDisp first.")

    if self.AveLogCPM is None:
        self.AveLogCPM = self.aveLogCPM()
    ntags = self.counts.shape[0]

    if span is None:
        if ntags > 10:
            span = (10 / ntags)**0.23
        else:
            span = 1
    self.span = span

    d = estimateGLMTagwiseDisp(self.counts, design=design, offset=self.getOffset(), dispersion=dispersion, trend=trend, span=span, prior_df=prior_df, AveLogCPM=self.AveLogCPM, weights=self.weights)
    self.prior_df = prior_df
    self.tagwise_dispersion = d
    return self

def estimateGLMTagwiseDisp(y, dispersion, design=None, offset=None, prior_df=10, trend=True, span=None, AveLogCPM=None, weights=None):
    # Check y
    y = np.asarray(y, order='F')
    (ntags, nlibs) = y.shape
    if ntags == 0:
        return 0

    # Check design
    if design is None:
        design = np.ones((y.shape[1], 1), order='F')
    else:
        design = np.asarray(design, order='F')

    if design.shape[1] >= y.shape[1]:
        warn("No residual df: setting dispersion to NA")
        return np.full((ntags,), np.nan, order='F')

    # Check offset
    if offset is None:
        offset = np.log(y.sum(axis=0))

    # Check span
    # span can be chosen smaller when ntags is large
    if span is None:
        if ntags > 10:
            span = (10 / ntags)**0.23
        else:
            span = 1

    # Check AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y, offset=offset, weights=weights)

    # Call Cox-Reid grid method
    tagwise_dispersion = dispCoxReidInterpolateTagwise(y, design, offset=offset, dispersion=dispersion, trend=trend, prior_df=prior_df, span=span, AveLogCPM=AveLogCPM, weights=weights)
    return tagwise_dispersion
