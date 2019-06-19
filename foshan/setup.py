# -*- coding:utf-8 -*-
## compile .py to .so
## use: python setup.py [path] eg: python setup.py /home/nari/work/foshan/
import os, sys, shutil
from distutils.core import setup
from Cython.Build import cythonize

build_dir = "build"
build_tmp_dir = build_dir + "/temp"

py_list = [os.path.join(path, file_name)
			for path, _, file_list in os.walk(sys.argv[1])
			for file_name in file_list if file_name.endswith('.py')]
print(py_list)
try:
    setup(ext_modules = cythonize(py_list),script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir])
except Exception, ex:
    print("error!!!", ex.message)
else:
    if os.path.exists(build_tmp_dir):
        shutil.rmtree(build_tmp_dir)
