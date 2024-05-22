def airway():
    """
    Retrieve the airway dataset as an :class:`AnnData` object

    The airway dataset is an RNA-Seq experiment on four human airway smooth
    muscle cell lines treated with dexamethasone [Himes2014]_.
    """
    import importlib.resources

    import anndata

    data_dir = importlib.resources.files(__package__)
    return anndata.read_h5ad(data_dir.joinpath("airway.h5ad"))
