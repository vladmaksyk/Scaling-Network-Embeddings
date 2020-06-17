===============================
DeepWalk
===============================

Requirements
------------
* numpy
* scipy
(may have to be independently installed) 
or `pip install -r requirements.txt` to install all dependencies

Installation
------------
1. `cd Scaling-Network-Embeddings`
2. `pip install -r requirements.txt`
3. `python setup.py install`


DeepWalk uses short random walks to learn representations for vertices in graphs.

Usage
-----
``$deepwalk --type exact --input edgelists/BlogCatalog-edgelist.csv --undirected True --walk-length 40 --budget 1 --window-size 10 --workers 1 --output embeddings/BlogCatalog-exact.txt.embeddings
``


**Full Command List**
    The full list of command line options is available with ``$deepwalk --help``
