Aplicación Web para la predicción de la etiqueta **delitos_seguimiento** usando el modelo entrenado de *Machine Learning*, denominado **ROBOS_IA**. 

El modelo obtuvo un 90% de acc y f1_score sobre un dataset de prueba. Los resultados de matriz de confusión también indican un desempeño adecuado en cuanto a la predicción de las clases.

Este desarrollo puede ser usado para implementar un API para al predicción de la etiqueta desagregación que es la que usa el funcionario USAI en la digitación del evento

## CATEGORÍA SIN INFORMACION

Existen casos que son interesantes en el sentido de que la Comisión los etiqueta como **SIN INFORMACION** y el Modelo es capaz de derivar una posible categorización suponiendo que las otras etiquetas
(i.e. las categorías que no son SIN INFORMACION) son las única del espacio de soluciones. Es decir si consideramos el espacio de soluciones $S = {Robo de motos, Robo a unidades económicas, Robo a personas, Robo a domicilio, Robo de bienes, accesorios y autopartes de vehículos, Robo de carros }$, la IA tratará de mapear el texto $\mathcal{X}$ una de estas categorías. En este caso se debe hacer la observarción a la comisión
y consultar si:

1. Se asume que el espacio de soluciones es S
2. Se asume que el espacio de soluciones debe ser ampliado a $S \cup {SIN INFORMACION}$. En este caso se tendrá que re-entrenar el modelo a fin de que reconozca la categoría SIN INFORMACIÓN

Ejemplos de esta situación son (revisar el archivo de PATH_TEST_SET = '/home/falconiel/CodePrograms/clasificaion_robos_fge/data/processed/validacionJunio2022.csv'):

* 180201822050019
* 170101822050315
* 090101822054985
* 170101822053955
* 170101822053826
* 090101822052720
* 091901822050025
* 170101822061466 Para mí son ambas etiquetas apropiadas