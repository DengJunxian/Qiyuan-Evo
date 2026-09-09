from setuptools import Extension, setup
import sys

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        "_civitas_lob",
        ["core/exchange/c_core/bindings.cpp", "core/exchange/c_core/lob.cpp"],
        include_dirs=[

            get_pybind_include(),
            "core/exchange/c_core"
        ],
        language="c++",
        extra_compile_args=["/std:c++17"] if sys.platform.startswith("win") else ["-std=c++17"],
    ),
]

setup(
    name="civitas_lob",
    version="0.1.0",
    description="C++ Limit Order Book for Civitas",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.10.0"],
    install_requires=["pybind11>=2.10.0"],
    zip_safe=False,
)
