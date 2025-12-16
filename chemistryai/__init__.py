from .chemlogic import *

def iupac(graph):
    return tree_to_iupac(build_tree_recursive(graph))
def smiles(string):
    return smiles_to_graphnode(string)
def draw(graph, filename="compound.png", size=(600, 400)):
    draw_graph_with_rdkit(graph, filename, size)
