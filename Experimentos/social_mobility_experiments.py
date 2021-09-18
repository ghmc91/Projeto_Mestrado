#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
from datetime import datetime
import time

import numpy as np
from scipy import spatial

import networkx as nx

import skmob
from skmob.measures.collective import visits_per_location

import social_cdr as scdr
import mobility_cdr as mcdr

import pickle

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import random

from cdlib import algorithms, evaluation


# In[2]:


def calculate_correlation(social_undirected):
	num_calls_list = []
	duration_list = []
	remove_list = []
	for user_from, user_to, data in social_undirected.edges(data=True):
		num_calls = data['num_calls']
		duration = data['duration']
		if duration < 1000:
			num_calls_list.append(num_calls)
			duration_list.append(duration)
		else:
			remove_list.append((user_from,user_to))
	#end
	
	social_undirected.remove_edges_from(remove_list)
	
	corr = pearsonr(num_calls_list, duration_list)
	print(corr)
	
	#plt.scatter(num_calls_list, duration_list)
	#plt.xlabel('num_calls')
	#plt.ylabel('duration')
	#plt.show()
#end


# In[3]:


def calculate_similarity_social_hops(social_undirected,traj_dict):
	all_1hop = []
	all_2hop = []
	all_3hop = []
	all_random = []
	for node in social_undirected.nodes():
		traj_vector_a = traj_dict[node]
		
		#print("vai calcular similarity")
		start_time = time.time()
		
		# Neighbors 1 hop
		#neighbors1 = scdr.n_neighbor(social_undirected,node, 1)
		neighbors1 = social_undirected.neighbors(node)
		cos_similarity1 = []
		#print(len(neighbors1))
		for neighbor in neighbors1:
			#print(neighbor)
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
			cos_similarity1.append(cos_similarity)
		#print("terminou")
		#print(cos_similarity1)
		if len(cos_similarity1) > 0:
			all_1hop.append(np.mean(cos_similarity1))
		
		
		# Neighbors 2 hops
		neighbors2 = scdr.n_neighbor(social_undirected,node, 2)
		cos_similarity2 = []
		for neighbor in neighbors2:
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
			cos_similarity2.append(cos_similarity)
		#print(cos_similarity2)
		if len(cos_similarity2) > 0:
			all_2hop.append(np.mean(cos_similarity2))
		
		
		# Neighbors 3 hops
		neighbors3 = scdr.n_neighbor(social_undirected,node, 3)
		cos_similarity3 = []
		for neighbor in neighbors3:
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
			cos_similarity3.append(cos_similarity)
		#print(cos_similarity3)
		if len(cos_similarity3) > 0:
			all_3hop.append(np.mean(cos_similarity3))
			
			
		# Random nodes
		num_friends = len(cos_similarity1)
		random_nodes = scdr.get_random_nodes(social_undirected,num_friends)
		cos_similarity_random = []
		for neighbor in random_nodes:
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
			cos_similarity_random.append(cos_similarity)
		if len(cos_similarity_random) > 0:
			all_random.append(np.mean(cos_similarity_random))
		
		#elapsed_time = time.time() - start_time
		#print(elapsed_time,"seconds for similarity")
		
	#end for node in social_undirected.nodes()
	
	print("Mean 1 hop:", np.mean(all_1hop))
	print("Mean 2 hop:", np.mean(all_2hop))
	print("Mean 3 hop:", np.mean(all_3hop))
	print("Mean random:", np.mean(all_random))
#end


# In[4]:


def jaccard_similarity(list1, list2):
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	return float(intersection) / union


# In[5]:


def calculate_jaccard_similarity(social_undirected,traj_dict,parameters):
	
	nbins = 25
	all_jaccard = dict()
	for i in range(nbins):
		all_jaccard[i] = []
	
	
	for node in social_undirected.nodes():
		traj_vector_a = traj_dict[node]
		# Neighbors 1 hop
		#neighbors1 = scdr.n_neighbor(social_undirected,node, 1)
		neighbors1 = scdr.n_neighbor(social_undirected,node, 1)
		neighbors2 = scdr.n_neighbor(social_undirected,node, 2)
		
		neighbors_all = list(set(neighbors1+neighbors2))
		#print(neighbors_all)
		
		for neighbor in neighbors_all:
			friends_friends = scdr.n_neighbor(social_undirected,node, 1)
		
			#print(friends_friends)
			jaccard = jaccard_similarity(neighbors_all,friends_friends)
			
			
			node_bin = int(np.ceil(jaccard * (nbins - 1)))
		
			#print(jaccard)
		
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
		
			all_jaccard[node_bin].append(cos_similarity)
			
		#end
	#end
	
	jaccard_list = []
	similarity_list = []
	for jaccard in all_jaccard:
		print(jaccard,len(all_jaccard[jaccard]))
		if len(all_jaccard[jaccard]) > 350:
			jaccard_list.append(jaccard/nbins)
			similarity_list.append(np.mean(all_jaccard[jaccard]))
	
		
	plt.plot(jaccard_list,similarity_list, color = parameters["color"], linewidth=1)
	#plt.plot(jaccard_list,similarity_list, parameters["style"], color = parameters["color"], label = parameters["label"], linewidth=2)
	#plt.tick_params(labelsize=30,size=20)
	plt.legend(loc="upper right")
	plt.xlabel('Jaccard',fontsize=14)
	plt.ylabel('Similarity',fontsize=14)
	
	
	#plt.plot(jaccard_list,similarity_list, parameters["style"], color = parameters["color"], label = parameters["label"], linewidth=2)
	
	#plt.show()
			
		
		
		
		
#end


# In[6]:


def calculate_ranked_friends(social_undirected,traj_dict,parameters):
	
	rank_dict = dict()
	for node in social_undirected.nodes():
		traj_vector_a = traj_dict[node]
		neighbors = social_undirected.neighbors(node)
		neighbors_list = []
		num_calls_list = []
		for neighbor in neighbors:
			neighbors_list.append(neighbor)
			num_calls_list.append(social_undirected[node][neighbor]['num_calls'])#/social_undirected[node][neighbor]['num_calls'])
		#end
		
		#print(neighbors_list)
		#print("num_calls:",num_calls_list)
		
		arg_sorted_neighbors = np.argsort(np.array(num_calls_list))
		#print("arg_sort",arg_sorted_neighbors)
		
		for i in range(len(neighbors_list)-1,-1,-1):
			#for pos in arg_sorted_neighbors:
			#pos = arg_sorted_neighbors[i]
			neighbor = neighbors_list[arg_sorted_neighbors[i]]
			
			#print(pos,neighbor,neighbors_list[pos],num_calls_list[pos])
						
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
			
			rank = len(neighbors_list) - i
			#rank = pos
			if rank not in rank_dict:
				rank_dict[rank] = []
			#print(rank,neighbor,neighbors_list[arg_sorted_neighbors[i]],num_calls_list[arg_sorted_neighbors[i]])
			rank_dict[rank].append(cos_similarity)
			
		#end for
		#print(rank_dict)
		#exit()
	#end for node in social_undirected.nodes()
	
	print("foi")
	rank_list = []
	similarity_list = []
	for rank in rank_dict:
		#if rank <= 30:
		rank_list.append(rank)
		similarity_list.append(np.mean(rank_dict[rank]))
	#end
	
	print(rank_list)
	print(similarity_list)
	#print(rank_dict)
	plt.plot(rank_list,similarity_list, color = parameters["color"], linewidth=1)
	plt.plot(rank_list,similarity_list, parameters["style"], color = parameters["color"], label = parameters["label"], linewidth=2)
	#plt.tick_params(labelsize=30,size=20)
	plt.legend(loc="upper right")
	plt.xlabel('Rank (# calls)',fontsize=14)
	plt.ylabel('Similarity',fontsize=14)
	
#end


# In[7]:


def calculate_ranked_reciprocity(social_directed,traj_dict,parameters):
	
	rank_dict = dict()
	for node in social_directed.nodes():
		traj_vector_a = traj_dict[node]
		neighbors = social_directed.neighbors(node)
		neighbors_list = []
		reciprocity_list = []
		for neighbor in neighbors:
			neighbors_list.append(neighbor)
			
			reciprocity = scdr.reciprocity(social_directed,node,neighbor)
			reciprocity_list.append(reciprocity)
		#end
		
		#print(neighbors_list)
		#print("num_calls:",num_calls_list)
		
		arg_sorted_neighbors = np.argsort(np.array(reciprocity_list))
		#print("arg_sort",arg_sorted_neighbors)
		
		for i in range(len(neighbors_list)-1,-1,-1):
			#for pos in arg_sorted_neighbors:
			#pos = arg_sorted_neighbors[i]
			neighbor = neighbors_list[arg_sorted_neighbors[i]]
			
			#print(pos,neighbor,neighbors_list[pos],num_calls_list[pos])
						
			traj_vector_b = traj_dict[neighbor]
			cos_similarity = 1 - spatial.distance.cosine(traj_vector_a, traj_vector_b)
			
			rank = i + 1
			#rank = pos
			if rank not in rank_dict:
				rank_dict[rank] = []
			#print(rank,neighbor,neighbors_list[arg_sorted_neighbors[i]],num_calls_list[arg_sorted_neighbors[i]])
			rank_dict[rank].append(cos_similarity)
			
		#end for
		#print(rank_dict)
		#exit()
	#end for node in social_undirected.nodes()
	
	print("foi")
	rank_list = []
	similarity_list = []
	for rank in rank_dict:
		if rank <= 30:
			rank_list.append(rank)
			similarity_list.append(np.mean(rank_dict[rank]))
	#end
	
	print(rank_list)
	print(similarity_list)
	#print(rank_dict)
	
	rank_list, similarity_list = zip(*sorted(zip(rank_list, similarity_list)))
	
	plt.plot(rank_list,similarity_list, color = parameters["color"], linewidth=1)
	plt.plot(rank_list,similarity_list, parameters["style"], color = parameters["color"], label = parameters["label"], linewidth=2)
	#plt.tick_params(labelsize=30,size=20)
	plt.legend(loc="upper right")
	plt.xlabel('Rank (reciprocity)',fontsize=14)
	plt.ylabel('Similarity',fontsize=14)
	
	
#end


# In[8]:


def calculate_reciprocity_list(social_directed,traj_dict,parameters):
	
	rank_dict = dict()
	reciprocity_list = []
	for node in social_directed.nodes():
		neighbors = social_directed.neighbors(node)
		for neighbor in neighbors:
			reciprocity = scdr.reciprocity(social_directed,node,neighbor)
			reciprocity_list.append(reciprocity)
		#end
		
		#print(neighbors_list)
		#print("num_calls:",num_calls_list)
		
	#end for node in social_undirected.nodes()
	
	
	return reciprocity_list

#end


# In[9]:


def calculate_friends_unique_places(social_undirected,traj_dict,parameters):
	num_friends_dict = dict()
	for (node, val) in social_undirected.degree():
		num_friends = val
		traj_vec = traj_dict[node]
		num_places = sum(map(lambda x : x != 0, traj_vec))
		
		if num_friends not in num_friends_dict:
			num_friends_dict[num_friends] = []
		num_friends_dict[num_friends].append(num_places)
	#end
	
	
	num_friends_list = []
	num_places_list = []
	for num_friends in num_friends_dict:
		if num_friends <= 30:
			num_friends_list.append(num_friends)
			num_places_list.append(np.mean(num_friends_dict[num_friends]))
	#end
	
	#print(num_friends_list)
	#print(num_places_list)
	
	num_friends_list, num_places_list = zip(*sorted(zip(num_friends_list, num_places_list)))
	
	plt.plot(num_friends_list,num_places_list, color = parameters["color"], linewidth=1)
	plt.plot(num_friends_list,num_places_list, parameters["style"], color = parameters["color"], label = parameters["label"], linewidth=2)
	#plt.tick_params(labelsize=30,size=20)
	plt.legend(loc="upper right")
	plt.xlabel('# of friends',fontsize=14)
	plt.ylabel('# of unique places',fontsize=14)
#end


# In[10]:


def calculate_reciprocity(region):
	social_directed = nx.read_gml('social_networks/%s_social_directed_filtered.gml' % region)
	print(nx.reciprocity(social_directed))
	
	social_directed_no_null_outdegree = nx.read_gml('social_networks/%s_social_directed_no_null_outdegree.gml' % region)
	print(nx.reciprocity(social_directed_no_null_outdegree))


# In[11]:


def cdf(x, plot=True, *args, **kwargs):
	x, y = sorted(x), np.arange(len(x)) / len(x)
	
	print(np.arange(len(x)))
	plt.plot(x, y, *args, **kwargs)
	plt.show()


# In[12]:


def ccdf(x, xlabel, ylabel, parameters):
	x, y = sorted(x,reverse=True), np.arange(len(x)) / len(x)
	
	plt.plot(x, y,color = parameters["color"],label = parameters["label"], linewidth=2)
	plt.legend(loc="upper right")
	plt.xlabel(xlabel,fontsize=14)
	plt.ylabel(ylabel,fontsize=14)
	plt.yscale('log')
	plt.xscale('log')


# In[13]:


def plot_similarity_hops():
	
	labels = ['1 hop','2 hops','3 hops','random']
	
	similarity_sjdr = [0.6877017325862492,0.5412426001930376,0.44191102936537185,0.06496870032335639]
	similarity_jf = [0.4270352122806851,0.24532627467520204,0.15570176482093037,0.04735678935674234]
	
	x = np.arange(len(labels))  # the label locations
	width = 0.35  # the width of the bars
	
	fig, ax = plt.subplots(figsize=(10,10))
	rects1 = ax.bar(x - width/2, similarity_sjdr, width, label='SJDR', color = 'tab:red')
	rects2 = ax.bar(x + width/2, similarity_jf, width, label='JF', color = 'tab:blue')
	
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_xlabel('Social distance')
	ax.set_ylabel('Similarity')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()

	#fig.tight_layout()

	#plt.show()


# In[14]:


def link_overlap(social_undirected,traj_dict):
	degree_list = {}
	for (node, val) in social_undirected.degree():
		degree_list[node] = val
	
	edge_duration_list = []
	edge_num_calls_list = []
	link_overlap_list = []
	for user_from, user_to, data in social_undirected.edges(data=True):
		edge_duration_list.append(data['duration'])
		edge_num_calls_list.append(data['num_calls'])
		
		neighbors_user_from = scdr.n_neighbor(social_undirected,user_from, 1)
		neighbors_user_to = scdr.n_neighbor(social_undirected,user_to, 1)
		
		number_common_neighbors = len(list(set(neighbors_user_from).intersection(neighbors_user_to)))
		degree_user_from = degree_list[user_from]
		degree_user_to = degree_list[user_to]
		
		if number_common_neighbors > 0:
			link_overlap_edge = number_common_neighbors / ((degree_user_from - 1) + (degree_user_to - 1) - number_common_neighbors)
			
		else:
			link_overlap_edge = 0
			
		
		if link_overlap_edge != 0 and link_overlap_edge != 1:
			link_overlap_list.append(link_overlap_edge)
		
		
	return [link_overlap_list,edge_duration_list]


# In[15]:


def plot_link_overlap(link_overlap_list,edge_duration_list, xlabel, ylabel, parameters):
	
	
	max_duration = max(edge_duration_list)
	print(max_duration)
	
	nbins = 25
	all_duration = dict()
	for i in range(nbins):
		all_duration[i] = []
	
	for duration,link_overlap in zip(edge_duration_list,link_overlap_list):
				
		plot_bin = int(np.ceil(duration/max_duration * (nbins - 1)))
		
		all_duration[plot_bin].append(link_overlap)
		
	#end
	
	duration_list = []
	link_list = []
	for duration in all_duration:
		#print(jaccard,len(all_jaccard[jaccard]))
		#if len(all_duration[duration]) > 350:
		duration_list.append(duration/max_duration/nbins)
		link_list.append(np.mean(all_duration[duration]))
	
	
	
	#edge_list_duration, y = sorted(edge_list_duration,reverse=True), np.arange(len(edge_list_duration)) / len(edge_list_duration)
	
	plt.plot(duration_list,link_list,color = parameters["color"],label = parameters["label"], linewidth=2)
	plt.legend(loc="upper right")
	plt.xlabel(xlabel,fontsize=14)
	plt.ylabel(ylabel,fontsize=14)
	#plt.yscale('log')
	#plt.xscale('log')


# In[ ]:


# Os comentários do código são voltados às análise realizadas para o artigo
# do Brasnam ("Caracterização da relação entre redes sociais e mobilidade de
# indivíduos em contextos urbanos")

# Cada gráfico e análise do paper é resultado de uma função deste código.
# Então a cada vez que preciso de um gráfico, eu descomento a linha correspondente 
# à análise e comento as outras.


# Nessa linha eu abro uma figura única, que vai guardar os gráficos com 
# as análises de todas as cidades
# (no caso paper do Branam, foram apenas SJDR e JF)
fig= plt.figure(figsize=(10,10))


# Essa função plota o gráfico da Fig. 5(a).
# Eu não passo nada pra ela (os valores são todos previamente calculados).
# (por isso essa função ficou um pouco mais distante das outras)
# O cálculo desses valores é feito na função calculate_similarity_social_hops e levam bastante tempo
# (especialmente pra JF).
#plot_similarity_hops()
#plt.savefig("results/similarity_hops.svg")

# Aqui eu defino algumas propriedades da execução, para tornar mais clara a visualização
# Se quiser rodar outra cidade, tem que definir os seus parâmetros em outro trecho,
region = 'sjdr' # Nome da região, para referência no código
parameters = {}
parameters["color"] = "tab:red" # Esquema de cores a ser usado nos gráficos
parameters["label"] = "SJDR" # Label a ser usado nos gráficos
parameters["style"] = "x" # Estilo de marcação nos gráficos

# Temos duas versõs das redes sociais, uma direcionada e uma não direcionada,
# cada tipo de análise vai demandar um tipo de rede.
# As redes já foram previamente computadas pelo código do arquivo build_social.py,
# que demora bastante para ser executado.
# É bom dar preferência para o uso das redes com "_no_null_outdegree" porque todos
# os nós delas realizaram chamadas e, por isso, têm marcação de geolocalização.
social_undirected = nx.read_gml('social_networks/%s_social_undirected_no_null_outdegree.gml' % region)
#social_directed = nx.read_gml('social_networks/%s_social_directed_no_null_outdegree.gml' % region)

# Esse trecho lê a estrutura de dados que guarda as trajetórias dos indivíduos,
# que é um dicionário. Isso já foi computado previamente no arquivo mobility_cdr.py,
# que demora bastante para ser executado.
traj_filename = 'traj_vectors/%s_traj_vectors.picke' % region
traj_file = open(traj_filename,'rb')
traj_dict = pickle.load(traj_file)
traj_file.close()

# Com a rede social e o dicionário de trajetórias calculados, as análises podem ser realizadas.

# Aqui, é importante dizer que na rede social os nós são as pessoas
# e uma aresta (i,j) existe se um indivíduo i liga para um indivíduo j.
# Uma aresta (i,j) guarda as informações agregadas de todas chamadas de i para j.
# Ou seja, não importa quantas ligações existam entre i e j, vai haver no máximo
# uma aresta (i,j). A aresta (i,j) tem duas propriedades:
# - "num_calls", que guarda o total de ligações de i para j.
# - "duration", que guarda o tempo total de ligação de i para j.
# Na versão não-direcionada, as propriedades "num_calls" e "duration" guardam o
# total de ligações e duração de i para j e de j para i.

# Essa função faz uma análise do link overlap como descrito por 
# Onnela, Jukka-Pekka & Saramäki, Jari & Hyvönen, J & Szabó, G & Lazer, David & Kaski, Kimmo & Kertész, János & Barabási, A.-L. (2007). Structure and Tie Strengths in Mobile Communication Networks. Proceedings of the National Academy of Sciences of the United States of America. 104. 7332-6. 10.1073/pnas.0610245104. 
link_overlap_list,edge_duration_list = link_overlap(social_undirected,traj_dict)
#plot_link_overlap(link_overlap_list,edge_duration_list,'link overlap','P',parameters)
ccdf(link_overlap_list,'link weight','P(> link weight)',parameters)



# Essa função faz uma análise da correlação entre as propriedades "num_calls" e "duration".
# Isso não entrou no paper, mas serviu para ver que meio que tanto faz usar uma ou outra.
#calculate_correlation(social_undirected)

# Essa função faz os cálculos da variação similaridade x hops (Fig 5(a)),
# que retorna os resultados que foram anotados e usados na função plot_similarity_hops
# lá em cima.
#calculate_similarity_social_hops(social_undirected,traj_dict)

# Essa função faz o cálculo que foi usado para as Fig. 4(a)
#calculate_ranked_friends(social_undirected,traj_dict,parameters)

# Essa função faz um cálculo de número de amigos X número de locais visitados,
# mas não entrou no paper.
# Eu fiquei com medo de que o número de amigos significasse apenas que um indivíduo
# fez mais chamadas e por isso ele visita mais lugares.
# Isso representaria um viés que eu preferi não expor no paper, por isso
# não coloquei esse gráfico.
#calculate_friends_unique_places(social_undirected,traj_dict,parameters)


# Nesse trecho eu identifico comunidades usando a biblioteca cdlib, usando o método
# de Louvain. Eu cheguei a considerar outros métodos, mas alguns testes que eu fiz
# mostraram que não ia ser muito diferente, então acabei usando só o Louvain mesmo.
# (no paper, usei o resultado desse trecho combinado com a Fig. 3(a) para discutir)
#partition = algorithms.louvain(social_undirected, resolution=1., randomize=False)
#mod = evaluation.newman_girvan_modularity(social_undirected,partition)
#print(mod)
#print(len(partition.communities))


# Nessa função eu calculo a reciprocidade das relações direcionadas, seguindo a
# metodologia de Chawla. O retorno é uma lista com as reciprocidades de todas
# as arestas da rede.
#reciprocity_list = calculate_reciprocity_list(social_directed,traj_dict,parameters)

# Essa função é um plor CCDF da reciprocidade (Fig. 3(b))
#ccdf(reciprocity_list,'reciprocity','P(> reciprocity)',parameters)

# Essa função é semelhante à calculate_ranked_friends, mas considerando a 
# reciprocidade (Fig. 4(b)).
#calculate_ranked_reciprocity(social_directed,traj_dict,parameters)

# O trecho abaixo pega os graus para a rede não-direcionada e para a rede direcionada
# e depois plota o CCDF (Fig. 1 (a)(b)(c))
#num_calls_list = []
#for user_from, user_to, data in social_undirected.edges(data=True):
#	num_calls_list.append(data['num_calls'])
#ccdf(num_calls_list,'# calls','P(> # calls)',parameters)
#degree_list = []
#for (node, val) in social_directed.out_degree():
#	degree_list.append(val)
#ccdf(degree_list,'k_out','P(> k_out)',parameters)


# Essa função calcula o coeficiente de Jaccard e faz a análise da Fig. 5(b).
#calculate_jaccard_similarity(social_undirected,traj_dict,parameters)


# A partir daqui, todo o código é repetido, para fazer a mesma análise para outra cidade.



#"""
region = 'jf'
parameters = {}
parameters["color"] = "tab:blue"
parameters["label"] = "JF"
parameters["style"] = "s"
social_undirected = nx.read_gml('social_networks/%s_social_undirected_no_null_outdegree.gml' % region)
#social_directed = nx.read_gml('social_networks/%s_social_directed_no_null_outdegree.gml' % region)

traj_filename = 'traj_vectors/%s_traj_vectors.picke' % region
traj_file = open(traj_filename,'rb')
traj_dict = pickle.load(traj_file)
traj_file.close()

#calculate_correlation(social_undirected)
#calculate_similarity_social_hops(social_undirected,traj_dict)
#calculate_ranked_friends(social_undirected,traj_dict,parameters)
#calculate_friends_unique_places(social_undirected,traj_dict,parameters)

#partition = algorithms.louvain(social_undirected, resolution=1., randomize=False)
#mod = evaluation.newman_girvan_modularity(social_undirected,partition)
#print(mod)
#print(len(partition.communities))
#calculate_reciprocity(region)

#calculate_ranked_reciprocity(social_directed,traj_dict,parameters)

#num_calls_list = []
#for user_from, user_to, data in social_undirected.edges(data=True):
#	num_calls_list.append(data['num_calls'])
#ccdf(num_calls_list,'# calls','P(> # calls)',parameters)

#degree_list = []
#for (node, val) in social_directed.out_degree():
#	degree_list.append(val)
#ccdf(degree_list,'k_out','P(> k_out)',parameters)

#calculate_jaccard_similarity(social_undirected,traj_dict,parameters)

#reciprocity_list = calculate_reciprocity_list(social_directed,traj_dict,parameters)
#print("foi 3")
#ccdf(reciprocity_list,'Reciprocity','P(> Reciprocity)',parameters)
#print("foi 4")

link_overlap_list,edge_duration_list = link_overlap(social_undirected,traj_dict)
#plot_link_overlap(link_overlap_list,edge_duration_list,'link overlap','P',parameters)
ccdf(link_overlap_list,'link weight','P(> link weight)',parameters)


# Aqui termina o código da outra cidade e eu já posso salvar a figura.

plt.savefig("results/ccdf_link_weight.svg")

