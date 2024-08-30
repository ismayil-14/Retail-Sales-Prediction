import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from streamlit_option_menu import option_menu
import re
from PIL import Image
import warnings
import pickle
import tensorflow as tf
warnings.filterwarnings("ignore")
from datetime import datetime

default_date = datetime(2015, 1, 1)
df =  pd.read_csv("Datasets/preprocessed_table.csv")
df.dropna(inplace= True)
store_df = pd.read_csv("Datasets/stores_data_set.csv")
with_markdown = pd.read_csv("Datasets/Cleaned_Dataset_with_Markdown.csv")
without_markdown = pd.read_csv("Datasets/Cleaned_Dataset_without_Markdown.csv")

ws_with_m = {1: 19808.243600620546, 2: 24203.06781449404, 3: 5820.287235244318, 4: 26147.91537646442, 5: 4783.873451841685, 6: 20123.33879785023,
 7: 7993.853332471463, 8: 12536.384078139301, 9: 8232.890087152582, 10: 23449.754096311222, 11: 17942.119810800432, 12: 13914.361180393811,
 13: 24337.64858231718, 14: 25435.364990800997, 15: 8563.087908877907, 16: 7409.681487340213, 17: 12132.921212912388, 18: 14184.52786286945,
 19: 18623.166696098087, 20: 25975.905043230992, 21: 10680.60685973687, 22: 13824.124134030531, 23: 17941.37534270249, 24: 17624.505621499993,
 25: 9721.549560418247, 26: 13525.02699333773, 27: 22189.0453054442, 28: 17367.57410842991,29: 7842.389304021081, 30: 7823.424731797525,
 31: 17805.734078885318, 32: 15145.727021642359, 33: 5258.486398900579, 34: 12782.866072555891, 35: 12729.374395801564, 36: 7530.019149453384,
 37: 9115.538675634318, 38: 6885.939101134989, 39: 19009.014270246786, 40: 12955.999970273298, 41: 16547.341818912253, 42: 9963.157038112586,
 43: 11464.734018083502, 44: 5572.953425290008, 45: 10934.050584071598}

ws_without_m = {1: 19870.35551558018, 2: 24198.08641197711, 3: 5812.558129662731, 4: 26102.588100070676, 5: 4782.2403657541945, 6: 20217.13089094456,
 7: 8045.494559683181, 8: 12560.319165182547, 9: 8343.044246947646, 10: 23371.467996758453,11: 17989.13996499917, 12: 13956.013211924186,
 13: 24387.288428035456, 14: 25609.079408209807, 15: 8647.272644994433, 16: 7488.494262866116, 17: 12202.22030939327, 18: 14221.556689913077,
 19: 18696.5512864033, 20: 26231.71597847101, 21: 10799.653819586529, 22: 13902.481575442136, 23: 18010.270433349273, 24: 17676.065938521824,
 25: 9882.212232002239, 26: 13587.765500507803, 27: 22301.20068676227, 28: 17409.4834119775, 29: 7832.162393207192, 30: 7883.891380611554,
 31: 17896.229667178613, 32: 15264.162567445026, 33: 5190.421840822657, 34: 12831.86154842414, 35: 12872.444477009345, 36: 7486.939226271415,
 37: 9140.853500522535, 38: 6965.251394639921, 39: 19173.84008204988, 40: 12978.003619825156, 41: 16639.041778009585, 42: 10002.823286655293,
 43: 11469.444307645754, 44: 5596.341891040143, 45: 10915.342515767985}

dept_dict= {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 78, 96, 99, 77], 
2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 51, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 39, 60, 47, 99, 77], 
3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 47, 51, 52, 54, 55, 56, 59, 60, 67, 71, 72, 74, 79, 81, 82, 85, 87, 90, 91, 92, 95, 96, 97, 94, 78, 80, 98, 77, 49, 83], 
4: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 78, 99, 39, 77], 
5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 81, 82, 85, 87, 90, 91, 92, 95, 96, 47, 51, 94, 98, 45, 78, 97, 80, 49, 77], 
6: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 78, 60, 99, 77], 
7: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 47, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 78, 48, 77, 99], 
8: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 49, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 45, 47, 78, 51, 99, 77, 96], 
9: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 49, 52, 54, 55, 56, 59, 67, 71, 72, 74, 79, 81, 82, 85, 87, 90, 91, 92, 95, 96, 97, 45, 94, 47, 51, 80, 93, 60, 48, 98, 77, 78], 
10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 77], 
11: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 45, 50, 99, 48, 77], 
12: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 47, 78, 77, 96, 99], 
13: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 78, 99, 77, 43], 
14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 43, 47, 78, 99, 77, 96], 
15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 49, 50, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 48, 51, 47, 78, 45, 60, 43, 77, 37, 99], 
16: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 79, 81, 82, 85, 87, 90, 91, 92, 94, 95, 96, 97, 98, 51, 60, 80, 47, 49, 93, 83, 78, 48, 77, 99], 
17: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 48, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 45, 47, 49, 77, 96, 99], 
18: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 49, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 95, 97, 94, 98, 48, 51, 60, 50, 99, 39, 47, 77, 96], 
19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 78, 19, 99, 48, 47, 77, 39], 
20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 78, 47, 99, 77, 96], 
21: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 49, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 78, 79, 81, 82, 83, 85, 87, 90, 91, 92, 93, 95, 97, 98, 51, 94, 45, 47, 80, 60, 48, 77, 96, 50, 99], 
22: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 95, 97, 98, 19, 78, 94, 47, 48, 49, 99, 77, 96], 
23: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 78, 79, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 80, 47, 60, 77, 99], 
24: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 49, 50, 51, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 45, 60, 47, 99, 77], 
25: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 49, 50, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 95, 97, 98, 47, 51, 94, 19, 45, 48, 60, 78, 77, 96], 
26: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 49, 51, 52, 54, 55, 56, 59, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 48, 78, 60, 99, 50, 77], 
27: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 49, 50, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 51, 78, 45, 60, 99, 77, 39], 
28: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 78, 47, 99, 43, 77], 
29: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 80, 19, 47, 48, 78, 45, 49, 77, 96, 43, 99], 
30: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 25, 28, 31, 32, 38, 40, 42, 44, 46, 52, 55, 56, 59, 60, 67, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 41, 24, 23, 26, 22, 20, 49, 34, 27, 29, 33, 19], 
31: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 47, 49, 51, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 45, 60, 77], 
32: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 78, 47, 48, 77], 
33: [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 21, 25, 26, 38, 40, 46, 55, 59, 60, 67, 74, 79, 80, 81, 82, 83, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 12, 9, 36, 41, 20, 71, 42, 27, 44, 6, 52, 72, 22, 56, 34, 31, 24, 33, 35, 23, 32, 49, 99], 
34: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 51, 52, 54, 55, 56, 58, 59, 65, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 99, 60, 77], 
35: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 78, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 95, 97, 98, 94, 47, 49, 19, 45, 77, 96], 
36: [1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 16, 17, 21, 26, 33, 38, 40, 46, 55, 59, 60, 67, 74, 79, 80, 81, 82, 83, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 11, 20, 87, 25, 18, 44, 9, 42, 12, 52, 41, 24, 23, 31, 32, 6, 29, 34, 56, 72, 22, 36, 99, 71, 49], 
37: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 25, 28, 31, 38, 40, 42, 46, 52, 55, 56, 59, 60, 67, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 26, 32, 24, 44, 23, 20, 71, 27, 33, 41, 22, 49, 99], 
38: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 23, 25, 26, 28, 31, 32, 34, 38, 40, 42, 46, 52, 59, 60, 67, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 20, 56, 29, 33, 44, 55, 24, 35, 27, 22, 49, 99], 
39: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 49, 52, 54, 55, 56, 58, 59, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 51, 78, 99, 60, 19, 77], 
40: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 52, 54, 55, 56, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 51, 49, 78, 19, 58, 99, 77], 
41: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 45, 46, 48, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 47, 19, 78, 99, 37, 77], 
42: [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 25, 28, 31, 32, 38, 40, 42, 46, 52, 59, 60, 67, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 44, 55, 56, 33, 20, 72, 71, 26, 41, 23, 34, 27, 22, 6, 49, 24], 
43: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 25, 26, 28, 31, 32, 38, 40, 42, 46, 52, 59, 60, 67, 72, 74, 79, 80, 81, 82, 83, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 87, 56, 33, 44, 20, 24, 49, 22, 71, 55, 23, 27, 99], 
44: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 23, 25, 28, 32, 38, 40, 42, 46, 52, 59, 60, 67, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 31, 55, 56, 99, 44, 20, 27, 71, 24, 34, 22, 49, 33, 26], 
45: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 95, 97, 98, 78, 94, 47, 45, 49, 77, 96]}



icon = Image.open("icon.jpeg")
st.set_page_config(page_title= "Retail Sales Price Prediction| By Mohamed Ismayil",
                layout= "wide",
                initial_sidebar_state= "expanded",
                menu_items={'About': """# This dashboard app is created by *Mohamed Ismayil*!"""}
                )

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Retail Sales Price Prediction</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu("Menu", ["HOME","PREDICT RETAIL PRICE","PREDICTED TABLE","INSIGHTS"], 
                        icons=["house","graph-up-arrow","table","info"],
                        menu_icon= "menu-button-wide",
                        default_index=0,
                        styles={"nav-link": {"font-size": "25px", "text-align": "left", "margin": "-2px", "--hover-color": "#FF5A5F"},
                                "nav-link-selected": {"background-color": "#FF5A5F"}}
                        )

if selected == "HOME":
    st.markdown("## :blue[Domain] : Retail Analytics")
    st.markdown("## :blue[Technologies used] : Time Series Analysis, Feature Engineering, Predictive Modeling, Data Cleaning and Preprocessing, Exploratory Data Analysis (EDA), Deep Learning Algorithms, AWS Deployment, Model Evaluation and Validation, Data Visualization, Tensorflow")
    st.markdown("## :blue[Overview] : This project involves developing a predictive ANN model to forecast department-wide sales for each store over the next year. It analyzes the impact of markdowns on sales during holiday weeks, providing actionable insights to optimize markdown strategies.")



if selected == "PREDICT RETAIL PRICE":
    st.header("Predict Retail Price")
    st.markdown("### Enter the details below to predict the Retail price.")

    with st.form("predict_price_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_date = st.date_input(
                "Select a date",
                default_date  
            )
            store = st.text_input("Enter store number: (Min: 1,Max : 45)")
            if store:
                dept = st.selectbox("Select a Department:", options=sorted(dept_dict[int(store)]))
            temperature = st.text_input("Enter temperature: (Min: -2 ,Max : 100)")
            fuel = st.text_input("Enter Fuel Price: (Min: 2.47 ,Max : 4.46)")
            cpi = st.text_input("Enter Customer Price Index: (Min: 126.06,Max : 227.23)")
            unemp = st.text_input("Enter Unemployment Rate: (Min: 3.87,Max : 14.31)")
            

        with col2:
            holiday = st.selectbox("Holiday Week", options= {"Yes","No"})
            m1 = st.text_input("Enter Markdown 1 price: ")
            m2 = st.text_input("Enter Markdown 2 price: ")
            m3 = st.text_input("Enter Markdown 3 price: ")
            m4 = st.text_input("Enter Markdown 4 price: ")
            m5 = st.text_input("Enter Markdown 5 price: ")
            lag  = st.text_input("Enter Previous Weekly Sales(If unknown leave it blank): ")

        submit_button = st.form_submit_button(label="Predict Resale Price")


    holiday = 1 if holiday == "Yes" else 0
    dummy = store_df[store_df["Store"]==store]
    store_type = store_df.loc[0, 'Type']
    store_type = 1 if store_type== "A" else 2 if store_type == "B" else 3
    store_size = store_df.loc[0, 'Size']
    day = selected_date.day
    month = selected_date.month
    year = selected_date.year
    if store:
        lag = ws_with_m[int(store)] if lag == "" else lag


    if submit_button:
        flag = 0 
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [temperature,fuel,cpi,unemp,m1,m2,m3,m4,m5]:             
            if not re.match(pattern, i):
                flag = 1
                break
        if flag == 1:
            st.error(f"You have entered an invalid value: {i}. Please enter a valid number without spaces.")
        else:
            with open("deep_learning_with_markdown.pkl", 'rb') as file:
                model_with_markdown = pickle.load(file)
            with open('scalar_with_markdown.pkl', 'rb') as f:
                scaler_with_markdown = pickle.load(f)
            with open("deep_learning.pkl", 'rb') as file:
                model = pickle.load(file)
            with open("scalar.pkl", 'rb') as f:
                scaler = pickle.load(f)
            
            
            testing = np.array([[store,store_type,store_size,day,month,year,dept,temperature,fuel,cpi,unemp,holiday,m1,m2,m3,m4,m5,lag]])
            tester = scaler_with_markdown.transform(testing)
            new_pred = model_with_markdown.predict(tester)[0]
            new_pred = new_pred[0] * new_pred[0] 
            st.success("Predicted Retail Weekly Sales Price with markdown impact: ${:.2f}".format(new_pred))
            
            testing = np.array([[store,store_type,store_size,day,month,year,dept,temperature,fuel,cpi,unemp,holiday,lag]])
            tester = scaler.transform(testing)
            new_pred = model.predict(tester)[0]
            new_pred = new_pred[0] * new_pred[0] 
            st.success("Predicted Retail Weekly Sales Price without markdown impact: ${:.2f}".format(new_pred))

if selected == "PREDICTED TABLE":
    st.header("Predicted vs. Actual Sales Data")
    
    store_options = ['All'] + sorted(with_markdown['Store'].unique())
    store_filter = st.sidebar.selectbox("Select Store", options=store_options)
    
    dept_filter = st.sidebar.selectbox("Select Department", options=['All'] + sorted(with_markdown['Dept'].unique()))
    is_holiday_filter = st.sidebar.selectbox("Select Holiday Status", options=['All', 'Yes', 'No'])
    
    if store_filter == 'All':
        filtered_with_markdown = with_markdown
        filtered_without_markdown = without_markdown
    else:
        filtered_with_markdown = with_markdown[with_markdown['Store'] == store_filter]
        filtered_without_markdown = without_markdown[without_markdown['Store'] == store_filter]
    
    if dept_filter != 'All':
        filtered_with_markdown = filtered_with_markdown[filtered_with_markdown['Dept'] == dept_filter]
        filtered_without_markdown = filtered_without_markdown[filtered_without_markdown['Dept'] == dept_filter]
        
    if is_holiday_filter != 'All':
        filtered_with_markdown = filtered_with_markdown[filtered_with_markdown['IsHoliday'] == is_holiday_filter]
        filtered_without_markdown = filtered_without_markdown[filtered_without_markdown['IsHoliday'] == is_holiday_filter]
    
    st.subheader("With Markdown Effects")
    st.dataframe(filtered_with_markdown)
    
    st.subheader("Without Markdown Effects")
    st.dataframe(filtered_without_markdown)

if selected == "INSIGHTS":
    st.header("Insights")

    st.title("Sales Analysis and Insights")
    st.write("This section provides insights based on the analysis of sales data during holiday and non-holiday weeks, as well as markdown effectiveness across different store types.")

    st.header("Insights")

    holiday_sales_max = df.groupby('IsHoliday')['Weekly_Sales'].max().reset_index()
    holiday_sales_mean = df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()

    st.title('Holiday vs Non-Holiday Weekly Sales Analysis')

    st.subheader('Max Weekly Sales on Holiday vs Non-Holiday Weeks')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='IsHoliday', y='Weekly_Sales', data=holiday_sales_max, ax=ax)
    ax.set_title('Max Weekly Sales on Holiday vs. Non-Holiday Weeks')
    ax.set_xlabel('Is Holiday')
    ax.set_ylabel('Max Weekly Sales')
    for index, row in holiday_sales_max.iterrows():
        ax.text(index, row['Weekly_Sales'], f'{row["Weekly_Sales"]:.2f}', color='black', ha="center")
    st.pyplot(fig)

    st.subheader('Mean Weekly Sales on Holiday vs Non-Holiday Weeks')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='IsHoliday', y='Weekly_Sales', data=holiday_sales_mean, ax=ax)
    ax.set_title('Mean Weekly Sales on Holiday vs. Non-Holiday Weeks')
    ax.set_xlabel('Is Holiday')
    ax.set_ylabel('Mean Weekly Sales')
    for index, row in holiday_sales_mean.iterrows():
        ax.text(index, row['Weekly_Sales'], f'{row["Weekly_Sales"]:.2f}', color='black', ha="center")
    st.pyplot(fig)
    st.markdown("""
    ### Higher Maximum Weekly Sales During Holidays
    - The maximum sum of weekly sales during holiday weeks across all stores is significantly higher than during non-holiday weeks, with a difference of approximately 300,000. This suggests that holidays drive substantial spikes in sales, indicating a strong impact of holiday events on customer spending behavior.

    ### Higher Average Weekly Sales in Holiday Weeks
    - The average weekly sales during holiday weeks are around 18,000, while for non-holiday weeks, it is about 15,000. This indicates that holiday weeks have higher peaks and consistently higher sales on average compared to non-holiday weeks.
    """)
    df['Date'] = pd.to_datetime(df['Date'])

    combined_data = df.groupby(['Date', 'IsHoliday'])['Weekly_Sales'].sum().reset_index()

    combined_data.set_index('Date', inplace=True)

    st.title('Combined Weekly Sales Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['Weekly_Sales'], hue=combined_data['IsHoliday'], palette={0: 'blue', 1: 'red'}, ax=ax)
    ax.set_title('Combined Weekly Sales Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Weekly Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("""
    ### Spillover Effect After Holiday Weeks
    - There are instances where combined weekly sales in non-holiday weeks surpass those in holiday weeks, typically occurring in the weeks immediately following a holiday week. This suggests a potential "spillover effect," where increased shopping momentum from a holiday week carries over into the following week, driven by factors such as post-holiday sales or delayed shopping.

    ### Sales Patterns Indicate Extended Shopping Periods
    - The higher sales in non-holiday weeks right after holiday weeks suggest that customers may continue shopping after the main holiday period due to leftover promotions or restocking. Retailers could capitalize on this by extending promotions or offering special post-holiday discounts to retain customer interest.
    """)

    st.title('Weekly Sales Over Time with Store Types Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x='Date', y='Weekly_Sales', data=df, hue='Type', palette={1: 'red', 2: 'green', 3: 'blue'}, ax=ax)
    ax.set_title('Weekly Sales Over Time with Store Types Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Sales')
    ax.legend(title='Store Type', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("""
    ### Clear Hierarchical Difference in Store Types
    - There is a distinct hierarchy in sales performance among store types, with Type A having the highest sales, followed by Type B, and Type C. This hierarchy remains consistent across the timeline, indicating significant and consistent differences in sales due to factors such as store size, location, or target market.
    """)
    df["Total_MarkDown"] = df["MarkDown1"] + df["MarkDown2"] + df["MarkDown3"] + df["MarkDown4"] + df["MarkDown5"]
    combined_data = df.groupby(['Date', 'IsHoliday'])['Total_MarkDown'].sum().reset_index()
    combined_data.set_index('Date', inplace=True)

    st.title('Combined Total MarkDown Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['Total_MarkDown'], hue=combined_data['IsHoliday'], palette={0: 'blue', 1: 'red'}, ax=ax)
    ax.set_title('Combined Total MarkDown Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total MarkDown Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("""
    ### Higher Total Markdown Sales During Non-Holiday Weeks
    - Total markdown sales are generally higher during non-holiday weeks compared to holiday weeks. This suggests that retailers may be more aggressive with markdowns in non-holiday periods to attract customers and drive sales during slower periods.

    ### Markdown Strategies Differ for Holiday and Non-Holiday Periods
    - During holiday weeks, markdowns may not need to be as extensive because customer traffic is naturally higher. In contrast, non-holiday weeks may require more significant markdowns to incentivize purchases, leading to higher total markdown values.
    """)

    combined_data = df.groupby(['Date', 'IsHoliday'])['MarkDown1'].sum().reset_index()
    combined_data.set_index('Date', inplace=True)

    st.title('Combined MarkDown1 Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['MarkDown1'], hue=combined_data['IsHoliday'].astype(str), 
                palette={'0': 'blue', '1': 'red'}, style=combined_data['IsHoliday'].astype(str), ax=ax)
    ax.set_title('Combined MarkDown1 Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total MarkDown1 Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)


    st.markdown("""
    ### Higher Sales During Non-Holiday Weeks
    - Non-holiday weeks dominate in total sales, indicating that markdowns represented by MarkDown1 are more effective or heavily used during non-holiday periods to drive baseline sales.
    """)

    combined_data = df.groupby(['Date', 'IsHoliday'])['MarkDown2'].sum().reset_index()
    combined_data.set_index('Date', inplace=True)

    st.title('Combined MarkDown2 Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['MarkDown2'], hue=combined_data['IsHoliday'].astype(str), 
                palette={'0': 'blue', '1': 'red'}, style=combined_data['IsHoliday'].astype(str), ax=ax)
    ax.set_title('Combined MarkDown2 Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total MarkDown2 Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("""
    ### Higher Sales During Holiday Weeks
    - MarkDown2 shows higher markdown sales during holiday weeks, suggesting that it is linked to holiday promotions or seasonal goods that are more popular during these periods.
    """)

    combined_data = df.groupby(['Date', 'IsHoliday'])['MarkDown3'].sum().reset_index()
    combined_data.set_index('Date', inplace=True)

    st.title('Combined MarkDown3 Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['MarkDown3'], hue=combined_data['IsHoliday'].astype(str), 
                palette={'0': 'blue', '1': 'red'}, style=combined_data['IsHoliday'].astype(str), ax=ax)
    ax.set_title('Combined MarkDown3 Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total MarkDown3 Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("""
    ### Significant Spike During Holiday Weeks
    - MarkDown3 shows a substantial difference between holiday and non-holiday weeks, with a mean markdown sales during holiday weeks being remarkably high. This suggests that MarkDown3 is highly focused on holiday sales strategies, possibly involving major discounts on high-value or bulk items.
    """)

    combined_data = df.groupby(['Date', 'IsHoliday'])['MarkDown4'].sum().reset_index()
    combined_data.set_index('Date', inplace=True)

    st.title('Combined MarkDown4 Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['MarkDown4'], hue=combined_data['IsHoliday'].astype(str), 
                palette={'0': 'blue', '1': 'red'}, style=combined_data['IsHoliday'].astype(str), ax=ax)
    ax.set_title('Combined MarkDown4 Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total MarkDown4 Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)


    st.markdown("""
    ### Peak Sales Before Holiday Weeks
    - Non-holiday weeks often have higher markdown sales, with a noticeable trend of peak markdown activity just before holiday weeks. This indicates strategic use of markdowns to attract early holiday shoppers or clear inventory.
    """)

    combined_data = df.groupby(['Date', 'IsHoliday'])['MarkDown5'].sum().reset_index()
    combined_data.set_index('Date', inplace=True)

    st.title('Combined MarkDown5 Over Time with Holiday Weeks Highlighted')

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(x=combined_data.index, y=combined_data['MarkDown5'], hue=combined_data['IsHoliday'].astype(str), 
                palette={'0': 'blue', '1': 'red'}, style=combined_data['IsHoliday'].astype(str), ax=ax)
    ax.set_title('Combined MarkDown5 Over Time with Holiday Weeks Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total MarkDown5 Sales')
    ax.legend(title='Is Holiday', loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("""
    ### Higher Sales During Non-Holiday Weeks
    - Similar to MarkDown1 and MarkDown4, MarkDown5 also sees higher sales during non-holiday weeks, suggesting it is more effective outside of holiday periods.
    """)

    st.header("Comparative Analysis")
    st.markdown("""
    ### Holiday vs. Non-Holiday Markdown Effectiveness
    - **Markdown1 and Markdown5**: More effective during non-holiday weeks.
    - **Markdown2, Markdown3, and Markdown4**: Show higher effectiveness during holiday weeks or leading up to holidays.

    ### Sales Patterns
    - Holiday weeks generally show higher total sales due to increased consumer spending.
    - Non-holiday weeks require more aggressive markdowns to drive sales.""")
    st.header("Strategic Recommendations")
    st.markdown("""
    #### Holiday Week Strategies
    - **Tailored Promotions**: Align Markdown2, Markdown3, and Markdown4 with holiday periods for maximum impact.
    - **High-Value Items**: Use Markdown3 for high-value or bulk items to capitalize on holiday shopping behavior.
    - **Pre-Holiday Planning**: Implement pre-holiday promotions with Markdown4 to boost early sales.

    #### Non-Holiday Week Strategies
    - **Aggressive Markdown Strategies**: Use Markdown1 and Markdown5 for regular promotions to maintain steady sales during slower periods.
    - **Inventory Management**: Leverage non-holiday periods for inventory clearance and promotional activities.

    #### Store Type Considerations
    - **Customized Approaches**: Develop store-type-specific markdown strategies based on performance data.
    - **Optimization**: Analyze store type data to optimize markdown effectiveness and tailor promotional strategies.
    """)