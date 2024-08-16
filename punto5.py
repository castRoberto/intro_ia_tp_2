import sys
import os

import data_manager

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def punto5 () -> None:

    """
    
    5. Crear una regresión de Ridge. Usando una validación cruzada de 5-folds y usando 
       como métrica el MSE, calcular el mejor valor de $\alpha$, buscando entre [0, 12.5]. 
       Graficar el valor de MSE versus $\alpha$.
    
    """

    ALPHA_NUM = 1000
    MAX_ALPHA = 12.5

    X_train_scaled, _ = data_manager.get_X_train_test_scaled ()
    y_train, _ = data_manager.get_y_train_test ()

    print ("\n\n-> Punto 5: \n")

    alpha_array = np.linspace(0, MAX_ALPHA, ALPHA_NUM)

    # Almacenar los MSE para cada alpha
    mse_array = []

    # Calculamos los coeficientes para diferentes valores de lambda
    for index, alpha in enumerate(alpha_array):

        # Creamos el modelo de Ridge
        ridge_model = Ridge (alpha=alpha)

        # Usar cross_val_score para calcular el MSE
        mse = -cross_val_score (ridge_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        mse_array.append(mse)


    # Convertir mse_array a numpy array
    mse_array = np.array(mse_array)

    # Encontrar el mejor valor de alpha
    best_alpha = alpha_array[np.argmin(mse_array)]
    best_mse = np.min(mse_array)

    print (f"\n- Mejor valor de alpha: {best_alpha}")
    print (f"\n- Mejor valor de MSE: {best_mse}")

    # Graficar el MSE versus alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_array, mse_array, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs Alpha en Ridge Regression')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    """Application entry point"""
    try:

        punto5 ()

    except KeyboardInterrupt:
        print ("Finish...")
        try:
            sys.exit (0)
        except SystemExit:
            os._exit (0)