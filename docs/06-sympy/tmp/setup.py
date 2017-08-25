from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('wrapper_module_5', ['wrapper_module_5.pyx', 'wrapped_code_5.c'], extra_compile_args=['-std=c99'])]
        )