# Artificial Intelligence master.
### International University of Valencia.


Hi there!
I attended this AI master's degree while working (2020-2022). Here are uploaded **some** assessments that I particulary liked.

Master: https://www.universidadviu.com/es/master-inteligencia-artificial?var=no&c=I90502M7004&&gclid=EAIaIQobChMI9oP9ktro6wIVNxoGAB2JyQ7LEAAYASAAEgJ_G_D_BwE&gclsrc=aw.ds

------------------------------------------------------

# Courses:

### 1. Data Cleaning.

<details>
    <summary> Click to expand. </summary>

1. Introducción
2. Python 101 y Jupyter Notebook.
3. Colecciones: Numpy.
4. Estructuras de Datos: Pandas.
5. Visualización de Datos: Matplotlib y Seaborn.
6. Python para Ciencia de Datos

</details>

----------------

### 2. Mathematics for AI.

----------------


### 3. Introduction to algorithms for optimization.

<details>
    <summary> Click to expand. </summary>


Foundations of heuristic and exact algorithms.

- Exact algorithms:
    - General approach => greedy algorithms, gradient descent, divide and conquer...
    - Sorting algorithms.
    - Searching algorithms => Branch and Bound...

- Heuristic algorithms:
    - Ant Colony Optimization, genetic algorithms...



</details>

----------------


### 4. Supervised Learning.

<details>
    <summary> Click to expand. </summary>

- Tipos de aprendizaje.

- Estructura de datos.

- Limpieza de datos.
    - Introducción a la limpieza de datos.
    - Normalización y estandarización.
    - Detección de outliers.
    - Imputación de valores ausentes.
    - Selección de atributos.

- Validación y evaluación.
    - Validación hold-out.
    - Validación cruzada.
    - Ajuste de parámetros y validación anidada.
    - Evaluación en regresión.
    - Evaluación en clasificación.

- Regresión.
    - Regresión lineal múltiple.
    - Vecinos más cercanos.

- Clasificación.
    - Regresión logística.
    - Árboles de decisión.

</details>

----------------

### 5. Unsupervised Learning.

<details>
    <summary> Click to expand. </summary>

- Introducción.
    - Minería de datos.
    - Aprendizaje supervisado y no supervisado.
    - Medidas de distancia.

-  Análisis de agrupamientos o clustering.
    - Basado en centroides: k-means, k-medoids.
    - Jerárquico.
    - Espectral.
    - Basado en densidades: Mean-shift, DBSCAN.
    - Basado en distribuciones: Mixtura de Gaussianas.

-  Aprendizaje semi-supervisado:
    - EM.
    - Basado en grafos.
    - Co-training.

- Reducción de dimensionalidad.
    - PCA.
    - ICA.

- Otras técnicas no supervisadas:
    - Análisis de grafos – Algoritmo PageRank.
    - Reglas de asociación – AlgoritmoApriori.

-------------------

### Clustering. 

**Métodos de clustering divisivo - n1.1.**

- KMeans.

- KMedioid. La diferencia es que, en K-medoids, cada cluster está representado por una observación presente en el cluster (medoid), mientras que en K-means cada cluster está representado por su centroide, que se corresponde con el promedio de todas las observaciones del cluster pero con ninguna en particular.

- Elegir el mejor valor de k con técnica del codo.

- K-means ++ 

**Métodos de clustering jerárquico - n1.2.**

- Aglomerativo.
    - Disimilitud intercluster mínima.
    - Disimilitud máxima.
    - Medidas ancho de silueta, calinski harabaz.

- Divisivo.
    - Disimilitud diámetro.
    - Disimilitud media.
    - Separación mcnaughton smith.

**Métodos de clustering espectral - n1.3.**

**Métodos de clustering basados en densidad -n1.4.**

- DBSCAN.

- Mean shift.

- Affinity propagation.

**Métodos de clustering basados en modeos probabilísticos -n1.5.**

- Algoritmo EM.

- sk-learn GaussianMixture.

**Métodos de aprendizaje semi-supervisado -n2.**

- Para cuando no todos los datos están etiquetados.

- Dos ejemplos: el primero, cómo funciona Naive Bayes con datos sin etiquetar, y el segundo, cómo podemos mejorar la versión básica del Naive Bayes aprendiendo un mejor modelo iterativamente mediante el algoritmo de EM.

- Multinomianl Naive Bayes
- Semi-supervised Multinomial Naive Bayes EM (expectation maximization )

**Métodos de análisis de componentes PCA -n3.**

- PCA.

- PCA en imágenes (VC08)

- Vamos a usar la función de scikit-learn para realizar el PCA. Esta función acepta dos parámetros complementarios: le podemos decir el número de componentes que queremos conservar (como en el ejemplo anterior con el dataset Iris) o el porcentaje de varianza explicada que queremos conservar (el número de componentes necesarias se calcula en base a ello). En este caso, le decimos que queremos un número de componentes que nos garanticen al menos un cierto umbral de varianza (mínima) explicada. Vamos a probar varios umbrales para observar el efecto que esto tiene en el número de componentes obtenidas y en el rendimiento de un clasificador aprendido con los datos transformados resultantes.


**Otros usos de no supervisado -n4.**

- VC09.
    - Segmentación mediante grafos.
    - Corte normalizado en imágenes.
    - GrabCut => eliminar fondo de la imagen.
    - GeneraR contenido con GANS: Generative Adversarial Networks para obtener imágenes de números.

- PageRank. Matriz de adyacencias, matriz de transiciones.

- Algoritmo HITS.

- PPR (personalized PageRank).

- Apriori. El algoritmo Apriori es un procedimiento para encontrar subsets frecuentes de ítems. En el caso de la cesta de la compra serían conjuntos de productos que suelen comprarse simultáneamente.

</details>

----------------

### 6. Approximate Reasoning

<details>
    <summary> Click to expand. </summary>

- Diseño de una solución con FuzzyCLIPS.

</details>

----------------


### 7. Neural Networks foundations.

<details>
    <summary> Click to expand. </summary>

- Fundamentos de las Redes Neuronales.
    -  Perceptrón simple y perceptrón multicapa.
    - Algoritmo de backpropagation.
    - Hiperparámetros de una red neuronal. 
    
- Deep learning.
    - Descripción de tipos de capas y su aplicabilidad.
    - Ejemplos de arquitecturas de red.
    - Optimización de hiperparámetros.

- Aplicación de las Redes Neuronales y Deep Learning a la resolución de tareas de IA.
    - Clasificación de imágenes.
    - Tratamiento de secuencias lógicas: Análisis y generación de textos con LSTM.
    - Introducción a Keras y TensorFlow
    
- Aprendizaje por refuerzo.
    - Introducción a aprendizaje por refuerzo. Estado del arte y retos futuros
    - Conceptos básicos y avanzados. 
    - Terminología. 
    - Clasificación de algoritmos: por Modelo, por Estrategia/Política, por Proceso de aprendizaje.    
    - Aprendizaje por refuerzo y Deep Learning 1. Algoritmo DQN.
    - Aprendizaje por refuerzo y Deep Learning 2. Algoritmo Policy Gradient.



</details>

----------------

### 8. Seminars.

----------------

# 9. Master's thesis

**https://github.com/albertovpd/viu_tfm-deep_vision_classification**