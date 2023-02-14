import networkx as nx
import h5py
import matplotlib.pyplot as plt
import sys
from math import sqrt

filename = sys.argv[1]

def format_bytes(nbytes):
    if nbytes > 1e9:
        return f'{nbytes/1e9:0.1f}GB'
    if nbytes > 1e6:
        return f'{nbytes/1e6:0.1f}MB'
    if nbytes > 1e3:
        return f'{nbytes/1e3:0.1f}kB'
    return f'{nbytes}B'

nodes = []
edges = []
def add_node_or_edge(name, obj):
    global nodes
    global edges
    name = name.strip('/')
    if name[-4:] == '/ref':
        if isinstance(obj, h5py.Dataset):
            # path is a reference dataset
            edge = [n.strip('/') for n in name.split('/ref')]
            attr = dict(shape=obj.shape[:1], nbytes=obj.nbytes)
            print('\t', attr['shape'], '\t', format_bytes(attr['nbytes']), '\t', name)
            edges.append(((edge[1], edge[0]), attr))
    elif name[-5:] == '/data':
        if isinstance(obj, h5py.Dataset):
            # path is a dataset
            attr = dict(shape=obj.shape, nbytes=obj.nbytes)
            print('\t', attr['shape'], '\t', format_bytes(attr['nbytes']), '\t', name)
            nodes.append((name[:-5], attr))
    elif 'classname' in obj.attrs:
        print(name)
        nodes.append((name, dict(classname=obj.attrs['classname'] + f' {obj.attrs["class_version"]}')))
    elif not '/ref' in name:
        print(name)
        nodes.append((name, dict()))

with h5py.File(filename,'r') as f:
    f.visititems(add_node_or_edge)

graph = nx.DiGraph()
for node in nodes:
    graph.add_node(node[0], **node[1])
for edge in edges:
    graph.add_edge(edge[0][0], edge[0][1], **edge[1])

labels = dict([(node, node+'\n'+'\n'.join([
    str(attr[key]) if key != 'nbytes' else format_bytes(attr[key]) for key in attr]) + '\n' + '\n'*len(attr))
    for node,attr in graph.nodes.items()])
edge_labels = dict([(edge, f'{attr["shape"]}\n{format_bytes(attr["nbytes"])}')
    for edge,attr in graph.edges.items()])

# generate plots
root_paths = set()
for node in graph.nodes:
    root_paths.add(node.split('/')[0])

for root in root_paths:
    plt.figure()
    subgraph = nx.subgraph_view(graph, filter_node=(lambda n: root in n or any(root in v or root in u for u,v in graph.edges(n))))
    sublabels = dict([(n, labels[n]) for n in subgraph.nodes])
    subedgelabels = dict([(e, edge_labels[e]) for e in subgraph.edges])
    pos = nx.circular_layout(subgraph)
    #nx.draw_networkx(subgraph,
    #    pos=pos,
    #    with_labels=True,
    #    node_color=[(0,0,0,0)],
    #    labels=sublabels)
    nx.draw_networkx_nodes(subgraph,
        pos=pos,
        node_color=[(0,0,0,0)])
    nx.draw_networkx_labels(subgraph,
        pos=pos,
        labels=sublabels)
    nx.draw_networkx_edges(subgraph,
        pos=pos)
    nx.draw_networkx_edge_labels(subgraph,
                                 pos,
                                 subedgelabels)
    plt.title(filename+'/'+root)
    plt.tight_layout()
plt.show()
