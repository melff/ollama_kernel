from distutils.core import setup

with open('README.rst') as f:
    readme = f.read()

setup(
    name='ollama_kernel',
    version='1.0',
    packages=['ollama_kernel'],
    description='Simple Ollama kernel for Jupyter',
    long_description=readme,
    author='Martin Elff',
    author_email='martin@elff.eu',
    url='https://github.com/melff/ollama_kernel',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
)
