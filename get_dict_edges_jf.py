import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle

G = nx.read_gml('/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/jf_social_directed_no_null_outdegree.gml')
dict_edges = dict()
for e in G.edges():
    dict_edges[e] = G.edges[e]


with open('/home/gustavo/Desktop/Mestrado/mestrado_dados/Arquivos_Pickle/dict_edges_jf.pkl', 'wb') as file:
	pickle.dump(dict_edges, file, protocol=pickle.HIGHEST_PROTOCOL)
	
