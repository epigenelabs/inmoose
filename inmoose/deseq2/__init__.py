from .core import DESeq as DESeq
from .DESeqDataSet import DESeqDataSet as DESeqDataSet
from .DESeqDataSet import makeExampleDESeqDataSet as makeExampleDESeqDataSet
from .DESeqTransform import DESeqTransform as DESeqTransform
from .dispersions import (
    estimateDispersionsFit as estimateDispersionsFit,
)
from .dispersions import (
    estimateDispersionsGeneEst as estimateDispersionsGeneEst,
)
from .dispersions import (
    estimateDispersionsMAP as estimateDispersionsMAP,
)
from .dispersions import (
    estimateDispersionsPriorVar as estimateDispersionsPriorVar,
)
from .estimateSizeFactors import (
    estimateSizeFactorsForMatrix as estimateSizeFactorsForMatrix,
)
from .lrt import nbinomLRT as nbinomLRT
from .outliers import replaceOutliers as replaceOutliers
from .parallel import estimateMLEForBetaPriorVar as estimateMLEForBetaPriorVar
from .prior import estimateBetaPriorVar as estimateBetaPriorVar
from .replicates import collapseReplicates as collapseReplicates
from .vst import varianceStabilizingTransformation as varianceStabilizingTransformation
from .wald import nbinomWaldTest as nbinomWaldTest
