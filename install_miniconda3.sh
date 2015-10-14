#!/bin/sh

set -e

if [ ! -d $HOME/.cache/clean_miniconda3/$PYTHON_VERSION ]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/Miniconda3-latest-Linux-x86_64.sh
    chmod +x $HOME/Miniconda3-latest-Linux-x86_64.sh
    $HOME/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    
    
    $HOME/miniconda3/bin/conda install --yes python=$PYTHON_VERSION $CONDA_DEPENDENCIES
    $HOME/miniconda3/bin/pip install $PIP_DEPENDENCIES
    
    $HOME/miniconda3/bin/conda clean --yes --tarballs
    
    cp -r $HOME/miniconda3 $HOME/.cache/clean_miniconda3/$PYTHON_VERSION
else
    echo "Use chached miniconda enviroment."
    cp -r $HOME/.cache/clean_miniconda3/$PYTHON_VERSION $HOME/miniconda3
fi

