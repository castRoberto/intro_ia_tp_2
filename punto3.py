import sys
import os

import data_manager

import matplotlib.pyplot as plt
import numpy as np


def punto3 () -> None:

    """
    
    3. Calcular la regresión lineal usando todos los atributos. Con el set de 
       entrenamiento, calcular la varianza total del modelo y la que es explicada 
       con el modelo. ¿El modelo está capturando el comportamiento del target? 
       Expanda su respuesta.
    
    """

    X_train_scaled, _ = data_manager.get_X_train_test_scaled ()
    y_train, _ = data_manager.get_y_train_test ()

    print ("\n\n-> Punto 3: \n")

    # 3.
    # 3.1. Calcular la regresión lineal usando todos los atributos
    regresion = data_manager.get_model (X_train_scaled, y_train)

    # 3.2. Con el set de entrenamiento, calcular la varianza total del modelo y la que es explicada con el modelo
    y_pred = regresion.predict (X_train_scaled)

    # Varianza total del modelo
    total_variance = np.sum ((y_train - np.mean (y_train)) ** 2) / (y_train.size - 1)

    # Varianza no explicada por el modelo
    unexplained_variance = np.sum ((y_train - y_pred) ** 2) / (y_train.size - 1) # Varianza residual

    # Varianza explicada por el modelo
    explained_variance = total_variance - unexplained_variance

    print (f"\n- Varianza total del modelo: {total_variance}")
    print (f"\n- Varianza explicada por el modelo: {explained_variance}")
    print (f"\n- Varianza no explicada por el modelo: {unexplained_variance}")

    # 3.3. ¿El modelo está capturando el comportamiento del target?
    print ("\n- El modelo captura una parte considerable del comportamiento del target,\
           \n  pero aún hay mucha variabilidad que no es explicada por el modelo ...\n")



if __name__ == "__main__":
    """Application entry point"""
    try:

        punto3 ()

    except KeyboardInterrupt:
        print ("Finish...")
        try:
            sys.exit (0)
        except SystemExit:
            os._exit (0)