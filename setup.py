import sys
import numpy
from setuptools import Extension, setup

cxx_compile_flags = ["-std=c++17"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
profiling = False
linetrace = False
if "--profile" in sys.argv:
    profiling = True
    linetrace = True
    macros += [("CYTHON_TRACE_NOGIL", "1")]
    sys.argv.remove("--profile")

common_cpp = Extension(
    "inmoose.common_cpp",
    [
        "inmoose/common_cpp/common_cpp.pyx",
        "inmoose/common_cpp/matrix.cpp",
    ],
    include_dirs=[numpy.get_include()],
    extra_compile_args=cxx_compile_flags,
    define_macros=macros,
)

edgepy_cpp = Extension(
    "inmoose.edgepy.edgepy_cpp",
    [
        "inmoose/edgepy/edgepy_cpp/edgepy_cpp.pyx",
        "inmoose/edgepy/edgepy_cpp/add_prior.cpp",
        "inmoose/edgepy/edgepy_cpp/add_prior_count.cpp",
        "inmoose/edgepy/edgepy_cpp/adj_coxreid.cpp",
        "inmoose/edgepy/edgepy_cpp/fit_levenberg.cpp",
        "inmoose/edgepy/edgepy_cpp/glm_levenberg.cpp",
        "inmoose/edgepy/edgepy_cpp/glm_one_group.cpp",
        "inmoose/edgepy/edgepy_cpp/interpolator.cpp",
        "inmoose/edgepy/edgepy_cpp/nbdev.cpp",
        "inmoose/edgepy/edgepy_cpp/objects.cpp",
    ],
    include_dirs=[numpy.get_include(), "inmoose/common_cpp/"],
    extra_compile_args=cxx_compile_flags,
    define_macros=macros,
)


setup(
    packages=[
        "inmoose",
        "inmoose/consensus_clustering",
        "inmoose/data",
        "inmoose/data/airway",
        "inmoose/data/pasilla",
        "inmoose/pycombat",
        "inmoose/common_cpp",
        "inmoose/edgepy",
        "inmoose/edgepy/edgepy_cpp",
        "inmoose/sim",
        "inmoose/utils",
    ],
    package_data={
        "inmoose/edgepy/edgepy_cpp": [
            "edgepy_cpp.pyx",
            "__init__.pxd",
            "add_prior.h",
            "add_prior_count.h",
            "ave_log_cpm.cpp",
            "compute_apl.cpp",
            "compute_nbdev.cpp",
            "edgepy_cpp.h",
            "fit_levenberg.h",
            "fit_one_group.cpp",
            "get_one_way_fitted.cpp",
            "glm.h",
            "initialize_levenberg.cpp",
            "initialize_levenberg.h",
            "interpolator.h",
            "maximize_interpolant.cpp",
            "objects.h",
            "utils.h",
        ],
        "inmoose/common_cpp": [
            "matrix.h",
        ],
        "inmoose/data/airway": [
            "airway.h5ad",
        ],
        "inmoose/data/pasilla": [
            "Dmel.BDGP5.25.62.DEXSeq.chr.gff",
            "geneIDsinsubset.txt",
            "pasilla_gene_counts.tsv",
            "pasilla_sample_annotation.csv",
            "treated1fb.txt",
            "treated2fb.txt",
            "treated3fb.txt",
            "untreated1fb.txt",
            "untreated2fb.txt",
            "untreated3fb.txt",
            "untreated4fb.txt",
        ],
    },
    ext_modules=[common_cpp, edgepy_cpp],
)
