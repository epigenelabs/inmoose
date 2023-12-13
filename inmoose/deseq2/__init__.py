from .core import DESeq
from .DESeqDataSet import DESeqDataSet, makeExampleDESeqDataSet
from .DESeqTransform import DESeqTransform
from .dispersions import (
    estimateDispersionsGeneEst,
    estimateDispersionsFit,
    estimateDispersionsMAP,
    estimateDispersionsPriorVar,
)
from .estimateSizeFactors import estimateSizeFactorsForMatrix
from .lrt import nbinomLRT
from .outliers import replaceOutliers
from .parallel import estimateMLEForBetaPriorVar
from .prior import estimateBetaPriorVar
from .replicates import collapseReplicates
from .vst import varianceStabilizingTransformation
from .wald import nbinomWaldTest
