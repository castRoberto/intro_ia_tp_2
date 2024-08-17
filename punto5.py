import sys
import os

import common

import matplotlib.pyplot as plt

def punto5 () -> None:

    """
    
    5. Crear una regresión de Ridge. Usando una validación cruzada de 5-folds y usando 
       como métrica el MSE, calcular el mejor valor de $\alpha$, buscando entre [0, 12.5]. 
       Graficar el valor de MSE versus $\alpha$.
    
    """

    X_train_scaled, _ = common.get_X_train_test_scaled ()
    y_train, _ = common.get_y_train_test ()

    print ("\n\n-> Punto 5: \n")

    # 5.
    alpha_array, mse_array = common.explore_alphas_with_cross_val (X_train_scaled, y_train)

    best_alpha, best_mse = common.get_best_alpha (alpha_array, mse_array)

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