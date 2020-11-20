#!/usr/bin/env python
# -*- coding: utf8 -*-

Run # %%
import csv
import sys
import pandas as pd
from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx
import datetime
from math import sin, cos, sqrt, atan2, radians
from geopy.distance import geodesic
import numpy as np
from sklearn.metrics import r2_score
import scipy
Run # %%

def fill_geolocation_from_antenna():
	# Essa função é mais geralzona... ela lê o arquivo de antenas e preenche a cidade...
	# Eu usei para fazer o primeiro preenchimento do arquivo de antenas, mas agora não uso mais...
	
	geolocator = Nominatim(user_agent="GoogleV3333")
	df_antenas = pd.read_csv("antenas.csv", sep=";", header=0)

	print(df_antenas.head(5))

	id_list = []
	antenas_id_list = []
	lat_list = []
	long_list = []
	municipality_list = []
	
	outfilename = "antenas_municipality.txt"
	outfile = open(outfilename,"w")


	for row in df_antenas.itertuples():
		coordinates = "%f, %f" % (row[4],row[5])
		
		ok = False
		while ok == False:
			try:
				location = geolocator.reverse(coordinates)
				try:
					city = location.raw["address"]["city"]
				except KeyError:
					city = "Unknown"
				try:
					municipality = location.raw["address"]["municipality"]
				except KeyError:
					municipality = "Unknown"
				print(row[0],":",city,"-",municipality)
				outfile.write("%d;%d;%s;%d;%.6f;%.6f;%s;%s\n" % (row[0],row[1],row[2],row[3],row[4],row[5],city,municipality))
				ok = True
			except KeyError:
				print("\tKeyError...",sys.exc_info()[0])
				outfile.write("%d;%d;%s;%d;%.6f;%.6f;%s,%s\n" % (row[0],row[1],row[2],row[3],row[4],row[5],"Unkown","Unknown"))
				ok = True
			except:
				print("\tWaiting...",sys.exc_info()[0])
	outfile.close()
	#end fill_geolocation_from_antenna()
	
def get_antennas_from_microrregion(microrregion,antenna_filename):
	# Essa função lê o arquivo de antenas e devolve as antenas de uma microrregião de interesse.
	# A microrregião é um atributo do 'location', então é bem fácil de filtrar.
	# Porém, essa não é a melhor forma, pq a denominação de microrregião caiu em desuso em 2017 e
	#   o IBGE passou a usar região imediata. Então é melhor usar região imediata.
	
	antenna_file = open(antenna_filename,"r")
	
	df_antenas = pd.read_csv(antenna_filename, sep=";", header=0)
	
	geolocator = Nominatim(user_agent="GoogleV3333")
	
	print("Ready...")
	
	#print(df_antenas.head(5))
	antennas_microrregion = []
	city_list = []
	index_list = []
	for row in df_antenas.itertuples():
		if row[8] == microrregion:
			print("Found",microrregion,"(",row[1],")")
			antennas_microrregion.append(row[2])
			index_list.append(row[1])
			
			coordinates = "%f, %f" % (row[5],row[6])
			location = geolocator.reverse(coordinates)
			
			try:
				city = location.raw["address"]["city"]
			except KeyError:
				try:
					city = location.raw["address"]["village"]
				except KeyError:
					try:
						city = location.raw["address"]["town"]
					except:
						city = "unknown"
			city_list.append(city)			
		
	
	# Se precisar retornar outros atributos, como as coordenadas, e preciso usar os outros índices da linha
	return [antennas_microrregion,city_list,index_list]
	#end get_antennas_from_microrregion

#end


def get_antennas_from_state(state,antenna_filename):
	# Essa função recebe uma lista de antenas com lat e long e descobre a cidade.
	# Para deixar o processamento mais rápido, eu filtro pelo estado e pelos DDDs que me interessam.
	# Aí eu leio uma lista e retorno apenas as antenas que  interessam na região (no caso, foi a região imediata de SJDR com os DDDs 32 e 35
	# Apesar de o objeto 'location' já ter um campo 'microrregion', essa denominação está defasada.
	# Por isso, optei por pensar nas cidades (que podem ser chamadas de 'city', 'village' ou 'town')
	
	antenna_file = open(antenna_filename,"r")
	cities_ibge = ['São João del-Rei', 'Tiradentes', 'São Vicente de Minas', 'São Tiago', 'São João del Rei', 'Santa Cruz de Minas', 'Ritápolis', 'Resende Costa', 'Prados', 'Piedade do Rio Grande', 'Nazareno', 'Coronel Xavier Chaves', 'Conceição da Barra de Minas', 'Madre de Deus de Minas', 'Lagoa Dourada', '---']
	ddd3235 = [35,32]
	
		
	df_antenas = pd.read_csv(antenna_filename, sep=";", header=0)
	geolocator = Nominatim(user_agent="GoogleV3333")
	
	print("Ready...")
	print(df_antenas.head(5))
	
	antennas_region = []
	city_list = []
	index_list = []
	lat_list = []
	long_list = []
	
	for row in df_antenas.itertuples():
		ok = 0
		city = ""
		
		if (row[3] == state) and (row[4] in ddd3235):
			print("Found MG (ddd 32 or 35)", row[7])
			
				
			if row[7] != "Unknown": # Já sabe a cidade
				city = row[7]
				print("City in CSV:",city)
			else: # Não sabe a cidade... vai ter que pegar pelas coordenadas
				coordinates = "%f, %f" % (row[5],row[6])
				while ok == 0: #Vai ficar tentando obter a cidade pela coordenada
					try:
						location = geolocator.reverse(coordinates)
						ok = 1 #OK, conseguiu
					except:
						print("\tERROR!!") #Não deu, mas vai continuar tentando...
						
				# Nesse ponto já tem o location...
				# Agora precisa obter a cidade do location
				try: # Tenta ver se está com a tag city
					city = location.raw["address"]["city"]
				except KeyError: # Se não for
					try: # Tenta ver se está com a tag village
						city = location.raw["address"]["village"]
					except KeyError: # Se não for
						try: # Tenta ver se está com a tag town
							city = location.raw["address"]["town"]
						except: # Se não for, desiste e imprime city
							city = "---"
			# OK, nesse ponto eu espero que a cidade já esteja preenchida com alguma coisa
			if city == '':
				print('\t\t\t****Cidade vazia!!!')
				exit()
			
			# Cerificar se a cidade está na região imediata...
			if city in cities_ibge:
				print("\t+++Found city in list",city)
				city_list.append(city)
				antennas_region.append(row[2])
				index_list.append(row[1])
				lat_list.append(row[5])
				long_list.append(row[6])
			elif city == "---":
				print("\t\t****continua unkown...")
			else:
				print("cidade não me interessa(", city,")")
	
	
	# Se precisar retornar outros atributos, como as coordenadas, e preciso usar os outros índices da linha
	return [antennas_region,city_list,index_list,lat_list,long_list]
	#end get_antennas_from_state
#end



def get_antenna_from_file(antenna_microrregion_filename):
	with open(antenna_microrregion_filename,'r') as f:
		content = f.readlines()
	antennas_microrregion = [int(x.strip().split()[0]) for x in content]
	
	return antennas_microrregion

#end get_antenna_from_file

def count_city(city_list):
	city_list_unique = set(city_list)
	city_count = []
	for city in city_list_unique:
		city_count.append(city_list.count(city))
		
		print(city,city_list.count(city))
	
#end count_city

"""
def filter_cdr_antennas(cdr_filename,antennas_microrregion):
	#print(cdr_filename,antennas_microrregion)
	
	#cdr_filename_antennas = "%s_sjdr" % cdr_filename
	cdr_filename_antennas = "cdr_sjdr.csv"
	
	#df_all_antennas = pd.read_csv(cdr_filename, sep=";", header=None)
	#print(df_all_antennas.shape)
	#count = 0
	#for row in df_all_antennas.itertuples():
	#	#print(row)
	#	if row[8] in antennas_microrregion:
	#		count+=1
	#		print("achou",count)
	#		#guarda em um novo df
	#		#print(row)
	#pd.set_option('display.max_columns', None)
	#print(df_all_antennas.head(5))
	#df_microrregion_antennas = df_all_antennas[df_all_antennas[7].isin(antennas_microrregion)]
	#print("antennas microrregion = ",df_microrregion_antennas.shape)
	
	
	#count = 0
	#with open(cdr_filename,'r') as fin:
	#	for row in csv.reader(fin, delimiter=';'):
	#		if int(row[7]) in antennas_microrregion:
	#			count+=1
	#print("achou",count)
	
	count = 0
	with open(cdr_filename,'r') as fin, open(cdr_filename_antennas,'a') as fout:
		writer = csv.writer(fout, delimiter=';')            
		for row in csv.reader(fin, delimiter=';'):
			#print(row[7])
			if int(row[7]) in antennas_microrregion:
				#print("achou")
				count+=1
				writer.writerow(row)
	print(count,"records found!")
	fin.close()
	fout.close()
	
#end
"""


def filter_cdr_antennas(cdr_filename):
	# Essa função toma como base um arquivo com antenas e guarda um CSV apenas com os registros
	#  das ligações ocorridas nessas antenas.
	# Como ele lê um CDR grande, eu nem uso o pandas nessa função.
	# A função não tem retorno. O resultado é guadado num CDR.
	# Normalmente essa função é o primeiro pré-processamento e é chamada através de um script que percorre
	#  todos os arquivos de CDR de uma pasta.
	
	#print(cdr_filename,antennas_microrregion)
	
	#cdr_filename_antennas = "%s_sjdr" % cdr_filename
	df_antennas = pd.read_csv("antenas_regiao_imediata_sjdr.csv", sep=";")
	
	antennas = []
	for row in df_antennas.itertuples():
		antennas.append(row.antenna)
	
	#print(antennas)
	#exit()
	
	cdr_filename_antennas = "cdr_regiao_imediata_sjdr.csv"
	
	"""
	df_all_antennas = pd.read_csv(cdr_filename, sep=";", header=None)
	print(df_all_antennas.shape)
	count = 0
	for row in df_all_antennas.itertuples():
		#print(row)
		if row[8] in antennas_microrregion:
			count+=1
			print("achou",count)
			#guarda em um novo df
			#print(row)
	pd.set_option('display.max_columns', None)
	print(df_all_antennas.head(5))
	df_microrregion_antennas = df_all_antennas[df_all_antennas[7].isin(antennas_microrregion)]
	print("antennas microrregion = ",df_microrregion_antennas.shape)
	"""
	"""count = 0
	with open(cdr_filename,'r') as fin:
		for row in csv.reader(fin, delimiter=';'):
			if int(row[7]) in antennas_microrregion:
				count+=1
	print("achou",count)
	"""
	count = 0
	with open(cdr_filename,'r') as fin, open(cdr_filename_antennas,'a') as fout:
		writer = csv.writer(fout, delimiter=';')            
		for row in csv.reader(fin, delimiter=';'):
			#print(row[7])
			if int(row[7]) in antennas:
				#print(row[7])
				#print("achou")
				count+=1
				writer.writerow(row)
				#exit()
	print(count,"records found!")
	fin.close()
	fout.close()
	
#end

def read_cdr_pandas(cdr_filename):
	
	# Essa função lê o CDR e exibe algumas características descritivas da base.
	# A função retorna o DF do CDR
	
	pd.set_option("display.max_columns", 15, "display.max_rows", None)
	
	df_cdr = pd.read_csv(cdr_filename, sep=";", encoding='utf-8')

	#print(df_cdr.head(5))
	
	#print(df_cdr.count)
	
	#df_cdr = df_cdr.head(10)
	#print(df_cdr)
	
	index = df_cdr.index
	
	print(len(index),"linhas (CDR completo)...")
	
	print(df_cdr.head(10))
	
	users_from = df_cdr.user_from
	users_from_unique = set(users_from)
	print(len(users_from_unique),"user_from (CDR completo)...")
	users_to = df_cdr.user_to
	users_to_unique = set(users_to)
	print(len(users_to_unique),"user_to (CDR completo)...")
	users_all_unique = users_from_unique.union(users_to_unique)
	
	print(len(users_all_unique),"union users (CDR completo)")
	duration_max = df_cdr.duration.max()
	duration_min = df_cdr.duration.min()
	duration_mean = df_cdr.duration.mean()
	duration_median = df_cdr.duration.median()
		
	print("CDR completo:")
	print("max:",duration_max,"| min:",duration_min,"| mean:",duration_mean,"| median:",duration_median)
	#print(duration_max,duration_min,duration_mean,duration_median)
	
	
	return [df_cdr]
	
#end read_cdr_pandas
	
	
def filter_cdr_duration_calls(df_cdr):
	
	# Essa função pega o CDR e filtra de acordo com as durações das ligações.
	# A idéia é tirar ligações muito rápidas e muito demoradas, que são consideradas outliers.
	# A função devolve o CDR filtrado.
	# A função também imprime algumas coisas básicas sobre a base.
	
	df_cdr_filter = df_cdr[df_cdr.duration <= 10]
	df_cdr_filter = df_cdr_filter[df_cdr_filter.duration >= 1]
	
	# Eliminando ligações "estranhas"
	print("dentro do intervalo [0.07,120]:", len(df_cdr_filter.index), "(",len(df_cdr_filter.index)/len(df_cdr.index),")")
	
	duration_max = df_cdr_filter.duration.max()
	duration_min = df_cdr_filter.duration.min()
	duration_mean = df_cdr_filter.duration.mean()
	duration_median = df_cdr_filter.duration.median()
	
	
	print("max:",duration_max,"| min:",duration_min,"| mean:",duration_mean,"| median:",duration_median)
	print("=====")
	
	
	"""
	# Aqui eu poderia fazer um filtro baseado em número de ligações
	groupby_user_from_duration_max = df_cdr_groupby_user_from.duration.sum().max()
	groupby_user_from_duration_min = df_cdr_groupby_user_from.duration.sum().min()
	groupby_user_from_duration_mean = df_cdr_groupby_user_from.duration.sum().mean()
	groupby_user_from_duration_median = df_cdr_groupby_user_from.duration.sum().median()
	
	print("GROUPBY max:",groupby_user_from_duration_max,"| min:",groupby_user_from_duration_min,"| mean:",groupby_user_from_duration_mean,"| median:",groupby_user_from_duration_median)
	"""
	
	return [df_cdr_filter]
	
# end filter_cdr_duration_calls

def calculate_residence_antenna(df_cdr_filter):

	# Essa função faz um filtro dos usuários que podem ter residência presumida.
	# A função também faz um filtro dos usuários que têm stay_locations ok para terem a residência calculada.
	# A função retorna os usuários com as residências presumidas.
	
	
	
	# Agrupando os user_from. Com base nos grupos de usuários vou aplicar os filtros.		
	df_cdr_groupby_user_from = df_cdr_filter.groupby('user_from')
	
		
	
	valid_users = [] # Aqui vão ficar guardados os usuários com residência calculada
	residence_antenna = [] # Aqui vão ficar as residências
	
	all_valid_users = [] # Aqui vão todos os usuários que poderiam ter residência calculada (mas pode ser que tenham 0, 1 ou 2 stay locations)
	all_count_stay_locations = [] # Aqui vão as contagens dos stay locations
	
	for user in df_cdr_groupby_user_from:
		
		num_calls_ok = False
		#print("[0]",user[0])
		user_from = user[0]
		####print("user_from",user_from)
		df_user = user[1]
		num_calls = len(df_user.index)
		###print("num_calls:",num_calls)
		if num_calls >= 5: # Atendeu ao critério de número mínimo de ligações
			if num_calls < 500: # Atendeu ao critério de número máximo de ligações
			 	num_calls_ok = True
		#	else:
		#		invalid_more_than_max_calls.append(user_from)
		#else:
		#	invalid_less_than_min_calls.append(user_from)
		
		if num_calls_ok == True: # Antendeu ao número de ligações. Vamos verificar o número de dias.
			####print("\tentrou no critério num_calls")
			num_days = len(set(df_user.date))
			###print("num_days:",num_days)
			
			if num_days >= 3: # Atende ao critério de número de dias distintos
				####print("\tentrou no critério num_days")
			
				stay_locations = []
			
				###print(df_user.date)
				
				for date,time,antenna in zip(df_user.date, df_user.time, df_user.antenna):
					###print(date,time)
					#Monday is 0 and Sunday is 6.
					weekday = datetime.date.fromisoformat(date).weekday()
					if 0 <= weekday <= 4: # dia da semana
						###print("dia da semana")
						time_call = datetime.time.fromisoformat(time)
						
						time19h = datetime.time(19,0,0)
						time6h = datetime.time(6,0,0)
						
						if time_call >= time19h or time_call <= time6h:
							#print("---noite")
							stay_locations.append(antenna)
						#else:
						#	#print("---diaa")
					elif weekday == 6: # domingo
						stay_locations.append(antenna)
						###print("domingo")
						
				#end for date,time,antenna
				# Vamos processar o stay_locations
				
				num_stay_locations = len(stay_locations)
				all_valid_users.append(user_from)
				all_count_stay_locations.append(len(set(stay_locations)))
				
				if num_stay_locations > 0:
					####print("\tentrou no critério de stay locations > 0")
					###print(stay_locations)
					
					
					for location in set(stay_locations):
						count_location = stay_locations.count(location)
						###print(location,count_location)
						if count_location > (num_stay_locations * 0.5):
							#####print("\tachou uma location com frequência maior que a metade")
							###print("count_location",count_location)
							###print("num_stay_locations",num_stay_locations)
							###print("(num_stay_locations / 2)",(num_stay_locations / 2))
							residence = location
							####print("\t\t\t*** residência:",residence)
							
							valid_users.append(user_from)
							residence_antenna.append(residence)
						# end if count_location > (num_stay_locations / 2)
					# end for location in set(stay_locations):
						
				# end if num_stay_locations > 0		
			# end if numdays >= 3
			#else: #num_days
			#	invalid_less_than_min_days.append(user)
		#end num_calls_ok == True
	# end user in df_cdr_groupby_user_from		
		
	
	return [valid_users,residence_antenna,all_valid_users,all_count_stay_locations]
		

def plot_antenas():
	# Essa função vai plotar as antenas, mas ainda não está feita...		
	gdf = gpd.GeoDataFrame(df_antennas,geometry=gpd.points_from_xy(df_antennas.long, df_antennas.lat))
	
	print(gdf)
	
	
	cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

	# We restrict to South America.
	ax = cities.plot(color='white', edgecolor='black')
	
	#gdf = gdf.to_crs(epsg=3857)
	ctx.add_basemap(ax, zoom=12)
	
	ax.set_xlim(-44.5, -44.0)
	ax.set_ylim(-21.5, -20.8)

	# We can now plot our ``GeoDataFrame``.
	gdf.plot(ax=ax, color='red')

	plt.show()
	
	gdf = geopandas.GeoDataFrame(cdr_df,geometry=geopandas.points_from_xy(cdr_df.Longitude, cdr_df.Latitude))
	
	
	
	exit()
	
#end

def read_cdr():
	# Função qque só lê o CDR... não tem mais muita serventia...
	
	user_from = []
	user_to = []
	with open("cdr_sjdr.csv",'r') as fin:
		count = 0
		for row in csv.reader(fin, delimiter=';'):
			if row[4] not in user_from:
				user_from.append(row[4])
			if row[6] not in user_to:
				user_to.append(row[6])
			#print(row)
			count+=1
	print(count,"linhas...")
	print(len(user_from),"user_from")
	print(len(user_to),"user_to")

#end

def read_shapefile_region(shapefile_name):

	
	city_list = ['SÃO JOÃO DEL REI', 'TIRADENTES', 'SÃO VICENTE DE MINAS', 'SÃO TIAGO', 'SANTA CRUZ DE MINAS', 'RITÁPOLIS', 'RESENDE COSTA', 'PRADOS', 'PIEDADE DO RIO GRANDE', 'NAZARENO', 'CORONEL XAVIER CHAVES', 'CONCEIÇÃO DA BARRA DE MINAS', 'MADRE DE DEUS DE MINAS', 'LAGOA DOURADA']
	
	gdf_all = gpd.read_file(shapefile_name)#, layer='countries')
	
	gdf_sjdr = gdf_all[gdf_all.NM_MUNICIP.isin(city_list)]
	
	#print(gdf_all.head(10))
	
	#print(gdf_sjdr)
	
	#print(len(gdf_sjdr.index))
	
	return [gdf_sjdr]
	
#end


def read_residence_antennas_from_file(residence_antennas_filename):
	df_residence_antennas = pd.read_csv(residence_antennas_filename, sep=";")
	
	#print(df_residence_antennas.head(10))
	
	return [df_residence_antennas]
#end read_residence_antennas_from_file


def calculate_closest_antenna(gdf_sjdr,df_antennas):
	# Aqui eu vou:
	#	+ criar uma lista de setores (com cada linha do gdf_sjdr)
	#	+ obter lat e long de cada setor
	#	+ percorrer o df_antennas e ver qual a antena mais próxima do setor
	#	+ cada linha do setor/closest_antenna vai ter id do setor/id da antena mais próxima
	
	setores = []
	closest_antennas = []
	distance_antennas = []
	
	
	
	
	for setor in gdf_sjdr.itertuples():
		
		
		setor_long = setor.geometry.centroid.x
		setor_lat = setor.geometry.centroid.y
		#print(setor.ID, setor.CD_GEOCODI, setor_lat, setor_long)
		least_distance = np.Inf
		least_distance_same_city = np.Inf
		
		
		for antenna in df_antennas.itertuples():

			antenna_lat = antenna.lat
			antenna_long = antenna.long
			
			#[distance] = calculate_distance_two_coordinates(antenna_lat,antenna_long, setor_lat, setor_long)
			#print(antenna_lat, antenna_long, setor_lat, setor_long, distance)
			
			coords_1 = (antenna_lat, antenna_long)
			coords_2 = (setor_lat, setor_long)
			distance = geodesic(coords_1, coords_2).kilometers
			#print("distância para a antena", antenna.antenna, " = ",distance)
			
			if antenna.city == setor.NM_MUNICIP:
				if distance < least_distance_same_city:
					least_distance_same_city = distance
					closest_antenna_same_city = antenna.antenna
			
			if distance < least_distance:
				#print("atualizou a distância")
				least_distance = distance
				closest_antenna = antenna.antenna
				
		#end for antenna
			
		#print("\n\n\t\tantena mais próxima do setor",setor.CD_GEOCODI,":", closest_antenna, "(dist = ",least_distance,")")
		
		setores.append(setor.CD_GEOCODI)
		if least_distance_same_city != np.Inf:
			closest_antennas.append(closest_antenna_same_city)
			distance_antennas.append(least_distance_same_city)
		else:
			closest_antennas.append(closest_antenna)
			distance_antennas.append(least_distance)
	#end for setor
		
	return [setores,closest_antennas,distance_antennas]
#end

def calculate_distance_two_coordinates(lat1,lon1,lat2,lon2):

	# approximate radius of earth in km
	R = 6373.0

	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)
	
	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c
	
	return [distance]

#end

def calculate_antenna_aggregated_information(df_antennas,df_residence_antennas,setores,closest_antennas,df_basico):

	#for setor,closest_antenna in zip(setores,closest_antennas):
	#	print(setor,closest_antenna)
		
	#print("\n\n")
	
	ids_antennas = df_antennas.antenna
	#print(ids_antennas)
	
	residence_antenna = df_residence_antennas['residence_antenna'].tolist()
	
	
	domicilios_setores_antenna = []
	residentes_setores_antenna = []
	presumidos_antenna = []
	
	for id_antenna in ids_antennas:
		"""
		- descobrir uma lista com as linhas do closest_antennas que têm valor igual a id_antenna (setores_id_antenna)
		- usar a lista de setores_filtrados para filtrar o df_basico (df_basico_antenna)
		- fazer os cálculos em cima do df_basico_antenna
		"""
		
		setores_id_antenna = [setores[i] for i, value in enumerate(closest_antennas) if value == id_antenna]
		
		#print(id_antenna)
		#print(setores_id_antenna)
		
		df_basico_antenna = df_basico[df_basico.Cod_setor.isin(setores_id_antenna)]
		
		domicilios_setor = df_basico_antenna.V001.sum()
		residencias_setor = df_basico_antenna.V002.sum()
		
		domicilios_setores_antenna.append(domicilios_setor)
		residentes_setores_antenna.append(residencias_setor)
			
		presumidos_antenna.append(residence_antenna.count(id_antenna))
		
		
	#for id_antenna in ids_antennas:
	
	
	
	print("\n\n\n")
	
	print("antenna\tpresumidos\tdomicilios\tresidentes")
	for antenna,presumidos,domicilios,residentes in zip(ids_antennas,presumidos_antenna,domicilios_setores_antenna,residentes_setores_antenna):
		print(antenna,"\t",presumidos,"\t",domicilios,"\t",residentes)
		
		
		
	plt.clf()
	plt.cla()
	plt.close()
	
	#r2 = r2_score(residentes_setores_antenna, domicilios_setores_antenna)
	correlation = scipy.stats.pearsonr(presumidos_antenna, domicilios_setores_antenna)
	print(correlation)
	#axes = plt.gca()
	fig= plt.figure(figsize=(12,12))
	textlabel = "Pearson = %.4f" % correlation[0]
	plt.plot(presumidos_antenna,domicilios_setores_antenna,'o',label = textlabel, color='tab:blue')
	plt.xlabel('Presumidos')
	plt.ylabel('Domicilios')
	leg = plt.legend(loc=1,fontsize='large')
	plt.show()
	#plt.savefig("legenda.png")
	plt.clf()
	plt.cla()
	plt.close()
	
	
	correlation = scipy.stats.pearsonr(presumidos_antenna, residentes_setores_antenna)
	print(correlation)
	#axes = plt.gca()
	fig= plt.figure(figsize=(12,12))
	textlabel = "Pearson = %.4f" % correlation[0]
	plt.plot(presumidos_antenna,residentes_setores_antenna,'o',label = textlabel, color='tab:red')
	plt.xlabel('Presumidos')
	plt.ylabel('Residentes')
	leg = plt.legend(loc=1,fontsize='large')
	plt.show()
	plt.clf()
	plt.cla()
	plt.close()
	
	exit()
		
		

#calculate_antenna_aggregated_information


			
if __name__ == "__main__":

	pd.set_option("display.max_columns", 50, "display.max_rows", None)
	
	"""
	# Usei esse trecho para filtrar apenas os registros CDR da região que eu quero...
	microrregion = "Microrregião São João del-Rei"
	antenna_filename = "antenas_municipality_with_unkown.txt"
	
	antenna_microrregion_filename = "antenas_regiao_imediata_sjdr.csv"
	
	cdr_filename = sys.argv[1]
	filter_cdr_antennas(cdr_filename)
	"""
	
	
	"""
	# Usei esse trecho para filtrar as antenas tomando com base as cidades (melhor opção), mas já está feito.
	
	#microrregion = "Microrregião São João del-Rei"
	#antenna_filename = "antenas_municipality_with_unkown.txt"
	#state = "MG"
	#[antennas_region,city_list,index_list,lat_list,long_list] = get_antennas_from_state(state,antenna_filename)
	#print(antennas_region)
	#print(city_list)
	#print(index_list)
	#print(lat_list)
	#print(long_list)
	
	#exit()
	"""
	
	
	"""
	# Usei esse trecho para filtrar as antenas tomando como base uma microregião.
	
	#antenna_microrregion_filename = "antenas_sjdr.txt"
	#cdr_filename = sys.argv[1]
	#[antennas_microrregion,city_list,index_list] = get_antennas_from_microrregion(microrregion,antenna_filename)
	#antennas_microrregion = get_antenna_from_file(antenna_microrregion_filename)
	#print(antennas_microrregion)
	
	#filter_cdr_antennas(cdr_filename,antennas_microrregion)
	
	#read_cdr()
	"""
	
	#"""
	# Nesse trecho eu leio o CDR, faço os filtros que preciso e calculo a antena da residência de cada usuário
	
	cdr_filename = "/home/gustavo/Desktop/mestrado_source/Projeto_Mestrado/processamento/cdr/cdr_regiao_imediata_sjdr.csv"
	[df_cdr] = read_cdr_pandas(cdr_filename)
	
	#df_cdr = df_cdr.head(1000)
	
	[df_cdr_filter] = filter_cdr_duration_calls(df_cdr) # <==== MEXER AQUI PARA SETAR FILTROS!!
	
	[valid_users,residence_antenna,all_valid_users,all_count_stay_locations] = calculate_residence_antenna(df_cdr_filter) # <==== MEXER AQUI PARA SETAR FILTROS!!
	

	print("valid_users:",len(valid_users))
	
	df_cdr_self_valid = df_cdr[df_cdr.user_to.isin(valid_users)]
	print("linhas com from e to no users_from (valid_users)",len(df_cdr_self_valid.index))
		
	
	d = {'user': valid_users, 'residence_antenna': residence_antenna}
	df_residence_antenna = pd.DataFrame(data=d)
	
	df_residence_antenna.to_csv (r'residence_antenna.csv', index = False, header=True, sep=";")
	
	#plt.hist(all_count_stay_locations,100)
	#plt.show()
	#"""
	
	
	#exit()
	
	[gdf_sjdr] = read_shapefile_region("../mg_setores_censitarios/31SEE250GC_SIR.shp")
	df_residence_antennas = pd.read_csv("residence_antenna.csv", sep=";")
	df_antennas = pd.read_csv("antenas_unique_regiao_imediata_sjdr.csv", sep=";")
	
	[setores,closest_antennas,distance_antennas] = calculate_closest_antenna(gdf_sjdr,df_antennas)
	
	df_basico = pd.read_csv("Basico_SJDR.csv", sep=";")
	
	
	calculate_antenna_aggregated_information(df_antennas,df_residence_antennas,setores,closest_antennas,df_basico)
	
	exit()
		
	
	
	
	
	
	#plt.hist(all_count_stay_locations,100)
	#plt.show()
	

#end main


"""



pd.set_option("display.max_columns", 15, "display.max_rows", None)

infile = sys.argv[1]

antennas_region = [23733,51492,23723,23741]

df_all = pd.read_csv(infile, sep=";", header=None)

print(df_all.head(5))

# Notes:
# - the `subset=None` means that every column is used 
#    to determine if two rows are different; to change that specify
#    the columns as an array
# - the `inplace=True` means that the data structure is changed and
#   the duplicate rows are gone  
df_all.drop_duplicates(subset=None, inplace=True)
print(df_all)
print("------")

#df_region = df_all[df_all[7] == 23733]
df_region = df_all[df_all[7].isin(antennas_region)]

print(df_region)

print("\n\n\n")
for row in df_region:
	print(row)

print("------")
exit()

for row in df_all.itertuples():
	if str(row[8]) in antennas_immediate_region:
		print("valido \n")
	#print(row[8])
	#exit()

exit()



print(infile)
with open(infile,'r') as fin:
	seen = set()
	for row in csv.reader(fin, delimiter=';'):
		print(row)
		if row not in seen:
			seen.add(row)
			print(row)
		exit()
	
	for row in csv.reader(fin, delimiter=';'):
		if row[3] == '32' or row[5] == '32':
			writer.writerow(row)
"""
