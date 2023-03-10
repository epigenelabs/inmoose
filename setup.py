from codecs import open
import os
import sys
import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

here = os.path.abspath(os.path.dirname(__file__))

about: dict = dict()
with open(os.path.join(here, "inmoose", "__version__.py"), "r", "utf-8") as fp:
    exec(fp.read(), about)

macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
profiling = False
linetrace = False
if "--profile" in sys.argv:
    profiling = True
    linetrace = True
    macros += [("CYTHON_TRACE_NOGIL", "1")]
    sys.argv.remove("--profile")

edgepy_cpp = Extension(
    "inmoose.edgepy.edgepy_cpp",
    [
        'inmoose/edgepy/edgepy_cpp/edgepy_cpp.pyx',
        'inmoose/edgepy/edgepy_cpp/add_prior.cpp',
        'inmoose/edgepy/edgepy_cpp/add_prior_count.cpp',
        'inmoose/edgepy/edgepy_cpp/adj_coxreid.cpp',
        'inmoose/edgepy/edgepy_cpp/fit_levenberg.cpp',
        'inmoose/edgepy/edgepy_cpp/glm_levenberg.cpp',
        'inmoose/edgepy/edgepy_cpp/glm_one_group.cpp',
        'inmoose/edgepy/edgepy_cpp/interpolator.cpp',
        'inmoose/edgepy/edgepy_cpp/nbdev.cpp',
        'inmoose/edgepy/edgepy_cpp/objects.cpp',
    ],
    include_dirs = [numpy.get_include()],
    extra_compile_args = ["-std=c++17"],
    define_macros = macros,
    )


setup(
    install_requires=[
        "mpmath>=1.1.0",
        "numpy>=1.18.5",
        "pandas",
        "patsy",
        "scipy",
    ],
    name="inmoose",
    version=about["__version__"],
    author="Maximilien Colange",
    author_email="maximilien@epigenelabs.com",
    description="InMoose: the Integrated Multi Omic Open Source Environment",
    url = "https://github.com/epigenelabs/inmoose",
    packages = ["inmoose", "inmoose/batch", "inmoose/edgepy", "inmoose/edgepy/edgepy_cpp"],
    package_data = { "edgepy/edgepy_cpp": [
        'edgepy_cpp.pyx',
        'add_prior.h',
        'add_prior_count.h',
        'ave_log_cpm.cpp',
        'compute_apl.cpp',
        'compute_nbdev.cpp',
        'edgepy_cpp.h',
        'fit_levenberg.h',
        'fit_one_group.cpp',
        'get_one_way_fitted.cpp',
        'glm.h',
        'initialize_levenberg.cpp',
        'initialize_levenberg.h',
        'interpolator.h',
        'maximize_interpolant.cpp',
        'objects.h',
        'utils.h',
    ] },
    ext_modules = cythonize(
        [edgepy_cpp],
        annotate = True,
        gdb_debug = True,
        emit_linenums = True,
        compiler_directives = { 'language_level': "3", 'profile': profiling, 'linetrace': linetrace },
    )
)

