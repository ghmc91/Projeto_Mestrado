import pandas as pd
import networkx as nx
G = nx.read_gml('/home/gustavo/Desktop/Mestrado/mestrado_dados/Redes/Teste.gml')

dict_users = {"5B5F2C071D12AF13219DF5EBE05132AF": 3,
              "9FB3B96B6D5E16C9DD564AA3E84F1954": 2,
              "B3299B0E587D7275E3E4D530E9EECF50": 3,
              "6432F1DF21BA38368D9A165C739EEBB3": 2,
              "85D5C50A6D882CA8E4BB00BCA3574417": 3,
              "0D8583F810B9720A8032BB939F12B3FF": 2,
              "6C10A9E9F325CAA3CCB7F9A0D6983D2A": 3,
              "B0C50ED1DEC9E06E4C64E7419DDC4B09": 3}

for i in G.nodes:
    for j in dict_users.keys():
        if i == j:
            G.nodes[i]['class'] = j