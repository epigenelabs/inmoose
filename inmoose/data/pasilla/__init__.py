def pasilla():
    """
    Retrieve the pasilla dataset as an :class:`AnnData` object

    The pasilla dataset is an RNA-Seq experiment on the effect of RNAi
    knockdown of Pasilla, the Drosophila melanogaster ortholog of mammalian
    NOVA1 and NOVA2, on the transcriptome [Brooks2011]_.
    """
    import importlib.resources

    import anndata as ad
    import pandas as pd

    data_dir = importlib.resources.files("inmoose.data.pasilla")
    cts = pd.read_csv(
        data_dir.joinpath("pasilla_gene_counts.tsv"), sep="\t", index_col=0
    )
    anno = pd.read_csv(data_dir.joinpath("pasilla_sample_annotation.csv"), index_col=0)

    # The columns of `cts` and the rows of `anno` use different labels and are
    # not in the same order. We first need to harmonize them before building the
    # AnnData object.

    # first get rid of the "fb" suffix
    anno.index = [i[:-2] for i in anno.index]

    # second reorder the index
    anno = anno.reindex(cts.columns)

    # we are now ready to build the AnnData object
    adata = ad.AnnData(cts.T, anno)

    return adata
