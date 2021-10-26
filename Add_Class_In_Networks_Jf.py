#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


def get_nodes_id(file):
    G = nx.read_gml(file)
    nodes_list = list(G.nodes(data=True))
    x = 0
    nodes_id = []
    while x < len(nodes_list):
        nodes_id.append(nodes_list[x][0])
        x += 1
    return nodes_id


# In[3]:


nodes_list = get_nodes_id('/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/jf_social_directed_no_null_outdegree.gml')


# In[4]:


df = pd.read_csv('/home/gustavo/Desktop/Mestrado/mestrado_dados/dados_18.11/df_main_jf.csv', sep=';')


# In[ ]:


df_merge = df[df['user_from'].isin(nodes_list)]
df_merge = df_merge[['user_from', 'classe']]
dict_user_class = df_merge.set_index('user_from').T.to_dict('records')
dict_user_class = dict_user_class[0]


# In[15]:


def generate_network_with_class(path_read, path_write):
    G = nx.read_gml(path_read)
    for i in G.nodes:
        for j in dict_user_class.keys():
            if i == j:
                G.nodes[i]['class'] = dict_user_class.get(j)
    for n in G.edges:
        G.edges[(n)]['numcalls'] = G.edges[(n)]['num_calls']
        del G.edges[n]['num_calls']
    nx.write_gml(G,path_write)


# In[16]:

generate_network_with_class('/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/jf_social_directed_no_null_outdegree.gml',
                            '/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/jf_social_directed_no_null_outdegree_with_class.gml')


generate_network_with_class('/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/jf_social_directed_no_null_outdegree.gml',
                            '/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/jf_social_directed_no_null_outdegree_with_class.gml')

