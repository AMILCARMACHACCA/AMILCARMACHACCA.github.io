<center><span style="font-size: 40px; color: #000080;"><b>PORTAFOLIO</b></span></center>

<center><span style="font-size: 20px;"><b>PROYECTOS DE INGENIERÍA ESTRUCTURAL</b></span></center>

---

# **1. MÉTODOS NUMÉRICOS**

## **1.1. METODO DE RECTANGULO**

[![Run in Google Colab](https://img.shields.io/badge/Colab-Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1x_ca2-5u-1tdmCVc-W1JuR15ZftuTgDx?usp=sharing)

Método numérico para la integración, conocida como el Método del Rectángulo, utiliza una aproximación basada en rectángulos para calcular la integral definida de una función en un intervalo dado. En términos generales, la fórmula se expresa como:

<p align="center">
  <img src="assets/img/rectangulo.svg" alt="Ecuación">
</p>



```Python
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# ingreso de datos
n= 30                         # numero de intervalos
a= 0                         # limite inferior
b= 5                          # limite superior
f = 'np.cos(x) + 0.1 * np.cos(2*x)'            # funcion a integrar

# calculos iniciales
h = (b - a) / n                                                   # ancho de los intervalos
xi = np.linspace(a, b, n + 1)                                     # nodos

# calculo de los puntos medios
x = []                                                            # lista vacia para los puntos medios
for i in range(1, n + 1):
    punto_medio = (xi[i - 1] + xi[i]) / 2                         # punto medio
    x.append(punto_medio)                                         # adiciona el punto medio a la lista x

x = np.array(x)                                                   # convierte la lista x en un arreglo
y = eval(f)                                                       # evaluacion de la funcion

# calculo de la integral
integral  = h*np.sum(y)
print("el resultado aprox. de la integral es::", integral)

#comprobacion con scipy
Ireal=spi.quad(lambda x: eval(f),a,b)                              # calcular la integral real
print("el resultado real de la integral es: ",Ireal[0])            # imprimir resultado

# Gráfico del metodo de integración
plt.bar(x, y, width=h, alpha=0.7, align='center', color=plt.cm.viridis(np.linspace(0, 1, len(x))))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Histograma y Curva de la Función')

# Gráfico de la curva f(x)
x_c = np.linspace(a, b, 1000)
y_c = eval(f.replace('x','x_c'))                                     # Evaluar f en 
plt.plot(x_c, y_c, color='red')
plt.show()
```

## **1.2. METODO DE SIMPSON 1/3**

[![Run in Google Colab](https://img.shields.io/badge/Colab-Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/12sg77KJNU2TiDso-KcWyi_m74l0gMmHy?usp=sharing)

Método numérico de integración conocida como Método de Simpson 1/3 emplea una estimación basada en parábolas para determinar la integral definida de una función dentro de un rango específico. La expresión general de este método se describe mediante la siguiente fórmula:

<p align="center">
  <img src="assets/img/simpson.svg" alt="Ecuación">
</p>


```Python
# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# ingreso de datos
n= 30                         # numero de intervalos
a= 0                         # limite inferior
b= 4                          # limite superior
f = 'np.sin(x**2)'            # funcion a integrar

# metodo de simpson
h=(b-a)/n                      # ancho de los intervalos
x=np.arange(a,b+h,h)           # vector de nodos
y=eval(f)                      # evaluar la funcion en los nodos

#calculo de la integral
s=y[0]+y[n]                    # sumar los nodos 0 y n
for i in range(1,n):           # sumar los nodos intermedios
    if i%2==0:                 # si es par
        s+=2*y[i]
    else:                      # si es impar
        s+=4*y[i]
s=s*h/3                                                             # multiplicar por h/3
print("el resultado aprox. de la integral es: ",s)                  # imprimir resultado

#comprobacion con scipy
Ireal=spi.quad(lambda x: eval(f),a,b)                               # calcular la integral real
print("el resultado real de la integral es: ",Ireal[0])             # imprimir resultado

# ploteo de la funcion y metodo de simpson
x1 = np.linspace(a, b, 10000)
y1 = eval(f.replace('x', 'x1'))                                      # Calcula la función en los puntos x1

plt.figure(figsize=(15, 5))
plt.plot(x1, y1, color='r', lw=2)
plt.plot(x, y, color='k', lw=0.5, alpha=1)
plt.fill_between(x, y, alpha=0.1, color='b')
plt.title("Metodo de Simpson")
plt.show()

```