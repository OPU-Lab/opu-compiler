from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("analysis",  ["analysis.py"]),
    Extension("debug",  ["debug.py"]),
    Extension("fp",  ["fp.py"]),
    Extension("global_var",  ["global_var.py"]),
    Extension("op",  ["op.py"]),
    Extension("outWriter",  ["outWriter.py"]),
    Extension("Quantizer",  ["Quantizer.py"]),
    Extension("sim",  ["sim.py"]),
    Extension("tvmFuncParser",  ["tvmFuncParser.py"]),
    Extension("suppress_stdout_stderr",  ["suppress_stdout_stderr.py"]),
#Extension("mymodule2",  ["mymodule2.py"]),
#   ... all your modules that need be compiled ...
]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3
setup(
    name = 'My Program Name',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)