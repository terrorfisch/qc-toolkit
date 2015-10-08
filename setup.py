from setuptools import setup
from setuptools.command.test import test as TestCommand

from shutil import rmtree, move
from importlib import find_loader

import sys
import os.path
import subprocess

if sys.version_info < (3, 3):
    sys.stderr.write('ERROR: You need Python 3.3 or later '
                     'to install the qctoolkit package.\n')
    exit(1)

if sys.version_info < (3, 5):
    requires_typing = ['typing==3.5.0']
else:
    requires_typing = []


""" This tester requires that the package is already installed """
class GlobalTester(TestCommand):
    def run(self):
        if find_loader('qctoolkit') is None:
            sys.stderr.write('ERROR: Please install the package before testing or use the \'local_test\' command.')
            exit(1)
        TestCommand.run(self)


""" This tester builds the package and tests it locally, it may not be installed """
class LocalTester(TestCommand):
    install_dir = os.path.join(os.path.dirname(__file__),'qctoolkit')
    build_dir = os.path.join(os.path.dirname(__file__),'build','lib','qctoolkit')
    
    def run(self):
        if os.path.isdir(LocalTester.install_dir):
            sys.stderr.write('ERROR: No clean test enviroment.\nPlease delete the qctoolkit folder.')
            exit(1)
        
        if find_loader('qctoolkit') is not None:
            sys.stderr.write('ERROR: Found an installed version of qctoolkit.\nEither test the installed one with the \'test\' command or remove it and use \'local_test\'')
            exit(1)
        
        #build the package
        subprocess.call([sys.executable,__file__,'build'])
        
        # move qctoolkit to root
        move(LocalTester.build_dir,LocalTester.install_dir)
        rmtree(os.path.join(os.path.dirname(__file__),'build'),ignore_errors=True)
        
        #run test
        caught_exception = None
        try: 
            TestCommand.run(self)
        except BaseException as e:
            caught_exception = e
        
        #cleanup
        rmtree(LocalTester.install_dir,ignore_errors=True)
        rmtree(os.path.join(os.path.dirname(__file__),'qctoolkit.egg-info'),ignore_errors=True)

        if caught_exception:
            raise caught_exception

subpackages = ['pulses','utils']
packages = ['qctoolkit'] + ['qctoolkit.' + subpackage for subpackage in subpackages]

setup(name='qctoolkit',
    version='0.1',
    description='Quantum Computing Toolkit',
    author='qutech',
    package_dir = {'qctoolkit': 'src'},
    packages=packages,
    tests_require=['pytest'],
    install_requires= ['py_expression_eval'] + requires_typing,
    extras_require={
        'testing' : ['pytest'],
        'plotting' : ['matplotlib'],
        'faster expressions' : ['numexpr']
    },
    cmdclass = {'test': GlobalTester, 'local_test': LocalTester},
    test_suite="tests",
)
