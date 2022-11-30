import pickle as pk
import numpy as np
import pandas as pd

usdCop = pd.read_csv('/sf2/DH_USD_COP.csv').loc[::-1]
dfPetr = pd.read_csv('/sf2/DH_Futuros_petroleo_Brent.csv').loc[::-1]

def values(x,mode):
  if x[-1] == 'M': return (float(x[:-2]))*1000000
  elif x[-1] == 'K': return (float(x[:-2]))*1000

def floatM(x:str):
  if '%' in x: x = float(x.replace("%","").replace(",","."))
  elif 'M' in x or 'K' in x : x = float(values(x.replace(",","."),'vol'))
  else : x = float(x.replace(".","").replace(",","."))
  return x

y = usdCop['Último'].apply(floatM)
yv = usdCop['% var.'].apply(floatM)
x = dfPetr['Último'].apply(floatM)*y
xv = dfPetr['% var.'].apply(floatM)
xl = dfPetr['Vol.'].apply(floatM)

test = input("Valor del petroleo \n")
test = float(test)

values = []

values.append(x)
values.append(xl)
values.append(y)
values.append(test)

pk.dump(values,open('/sf3/values.txt','wb'))
