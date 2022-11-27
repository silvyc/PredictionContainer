import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


usdCop = pd.read_csv('DH_USD_COP.csv').loc[::-1]
dfPetr = pd.read_csv('DH_Futuros_petroleo_Brent.csv').loc[::-1]


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


test = 500000 #Barriles de petroleo
demo = False


cap1 = tf.keras.layers.Dense(units=4,input_shape=[1])
cap2 = tf.keras.layers.Dense(units=3)
capOut = tf.keras.layers.Dense(units=1)

petr_cop = tf.keras.Sequential([cap1,cap2,capOut])
vol_petr = tf.keras.Sequential([cap1,cap2,capOut])

if demo: 
  vol_petr.compile(
      optimizer=tf.keras.optimizers.Adam(0.1),
      loss = 'mean_squared_error'
  )
petr_cop.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print('iniciando entrenamiento')
if demo: 
  regist1 = vol_petr.fit(xl,x,epochs=800,verbose=False)
regist2 = petr_cop.fit(x,y,epochs=800,verbose=False)
print('Completado')


if demo:
  plt.xlabel('#Epoca')
  plt.ylabel('Magnitud perdida')
  plt.plot(regist1.history['loss'])

if demo:
  print('Probemos el modelo:')
  result  = vol_petr.predict([test])
  print(f'Si hay {test} barriles de petroleo en el mercado, el precio por barril es de ${result[0][0]} USD')

if demo: 
  vol_petr.get_weights()

plt.xlabel('#Epoca')
plt.ylabel('Magnitud perdida')
plt.plot(regist2.history['loss'])

print('Probemos el modelo:')
price = test
if demo:
  price = result
result  = petr_cop.predict([price])
print(f'Si el valor del barril de petroleo es ${price} COP, se estima que el valor de 1 USD seria ${round(result[0][0])} COP en Colombia')
