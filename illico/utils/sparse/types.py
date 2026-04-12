from collections import namedtuple

CSCMatrix = namedtuple("CSCMatrix", ["data", "indices", "indptr", "shape"])
CSRMatrix = namedtuple("CSRMatrix", ["data", "indices", "indptr", "shape"])
