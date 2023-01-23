from .core import DESeq
from .DESeqDataSet import DESeqDataSet, makeExampleDESeqDataSet
from .dispersions import (
    estimateDispersionsGeneEst,
    estimateDispersionsFit,
    estimateDispersionsMAP,
)
from .estimateSizeFactors import estimateSizeFactorsForMatrix
from .lrt import nbinomLRT
from .outliers import replaceOutliers
from .parallel import estimateMLEForBetaPriorVar
from .prior import estimateBetaPriorVar
from .replicates import collapseReplicates
from .wald import nbinomWaldTest
