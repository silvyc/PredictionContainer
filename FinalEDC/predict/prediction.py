import pickle as pk
import tensorflow as tf
import matplotlib.pyplot as plt


demo = False

values = pk.load(open('/sf0/values.txt', 'rb'))

cap1 = tf.keras.layers.Dense(units=4, input_shape=[1])
cap2 = tf.keras.layers.Dense(units=3)
capOut = tf.keras.layers.Dense(units=1)

model = petr_cop = tf.keras.Sequential([cap1, cap2, capOut])
vol_petr = tf.keras.Sequential([cap1, cap2, capOut])

if demo:
    vol_petr.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )
petr_cop.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

if demo:
    regist1 = vol_petr.fit(values[1], values[0], epochs=800, verbose=False)
regist2 = petr_cop.fit(values[0], values[2], epochs=800, verbose=False)

if demo:
    plt.xlabel('#Epoca')
    plt.ylabel('Magnitud perdida')
    plt.plot(regist1.history['loss'])
if demo:
    plt.xlabel('#Epoca')
    plt.ylabel('Magnitud perdida')
    plt.plot(regist1.history['loss'])

if demo:
    result = vol_petr.predict(values[3])
    print(
        f'Si hay {values[3]} barriles de petroleo en el mercado, el precio por barril es de ${result[0][0]} USD')

if demo:
    vol_petr.get_weights()

plt.xlabel('#Epoca')
plt.ylabel('Magnitud perdida')
plt.plot(regist2.history['loss'])

price = values[3]
if demo:
    price = result
result = petr_cop.predict([price])
print(
    f'Si el valor del barril de petroleo es ${price} COP, se estima que el valor de 1 USD seria ${round(result[0][0])} COP en Colombia')
