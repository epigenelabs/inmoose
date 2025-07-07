import sys

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class build_ext_cxx17(build_ext):
    def build_extensions(self):
        std_flag = (
            "-std:c++17" if self.compiler.compiler_type == "msvc" else "-std=c++17"
        )
        for e in self.extensions:
            e.extra_compile_args.append(std_flag)
        super().build_extensions()


macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
profiling = False
linetrace = False
if "--profile" in sys.argv:
    profiling = True
    linetrace = True
    macros += [("CYTHON_TRACE_NOGIL", "1")]
    sys.argv.remove("--profile")

stats_cpp = Extension(
    "inmoose.utils.stats_cpp",
    [
        "inmoose/utils/_stats.pyx",
    ],
    include_dirs=[numpy.get_include()],
    define_macros=macros,
)

edgepy_cpp = Extension(
    "inmoose.edgepy.edgepy_cpp",
    [
        "inmoose/edgepy/edgepy_cpp/edgepy_cpp.pyx",
    ],
    include_dirs=[numpy.get_include()],
    define_macros=macros,
)


setup(
    cmdclass={"build_ext": build_ext_cxx17},
    packages=[
        "inmoose",
        "inmoose/consensus_clustering",
        "inmoose/data",
        "inmoose/data/airway",
        "inmoose/data/pasilla",
        "inmoose/pycombat",
        "inmoose/limma",
        "inmoose/edgepy",
        "inmoose/edgepy/edgepy_cpp",
        "inmoose/sim",
        "inmoose/utils",
        "inmoose/deseq2",
        "inmoose/diffexp",
        "inmoose/cohort_qc",
    ],
    package_data={
        "inmoose/edgepy/edgepy_cpp": [
            "edgepy_cpp.pyx",
            "__init__.pxd",
            "edgepy_cpp.h",
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
        "inmoose/cohort_qc": ["qc_report.html"],
    },
    ext_modules=[edgepy_cpp, stats_cpp],
)
