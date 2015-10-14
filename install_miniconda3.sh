#!/bin/sh

set -e

if [ ! -d $HOME/.cache/clean_miniconda3/bin ]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/Miniconda3-latest-Linux-x86_64.sh
    chmod +x $HOME/Miniconda3-latest-Linux-x86_64.sh
    $HOME/Miniconda3-latest-Linux-x86_64.sh -b
    
    $HOME/miniconda3/bin/conda install --yes $CONDA_PYTHON_DEPENDENCIES
    $HOME/miniconda3/bin/conda inslatt --yes $EXTERN_PAYTHON_DEPENDENCIES
    
    $HOME/miniconda3/bin/conda clean --yes --tarballs --packages
    
    cp -r $HOME/miniconda3 $HOME/.cache/clean_miniconda3
else
    echo "Use chached miniconda enviroment."
    cp -r $HOME/.cache/clean_miniconda3 $HOME/miniconda3
fi

