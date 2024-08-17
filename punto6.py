


import sys
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error

import common

def punto6 () -> None:

    """

    6. Comparar, entre la regresión lineal y la mejor regresión de Ridge, los resultados 
       obtenidos en el set de evaluación. ¿Cuál da mejores resultados (usando MSE y MAE)? 
       Conjeturar por qué el mejor modelo mejora. ¿Qué error puede haberse reducido?
    
    """

    # 6.

    X_train_scaled, X_test_scaled = common.get_X_train_test_scaled ()
    y_train, y_test = common.get_y_train_test ()

    print ("\n\n-> Punto 6: \n")

    # Obtenemos el modelo de regresión lineal
    linear_model = common.get_linear_model (X_train_scaled, y_train)

    # Obtenemos el menjor modelo de Ridge
    alpha_array, mse_array = common.explore_alphas_with_cross_val (X_train_scaled, y_train)

    best_alpha, _ = common.get_best_alpha (alpha_array, mse_array)

    best_ridge_model = common.get_best_ridge_model (best_alpha, X_train_scaled, y_train)

    # Predecir los valores de y para el set de evaluación
    y_pred_linear = linear_model.predict (X_test_scaled)
    y_pred_ridge = best_ridge_model.predict (X_test_scaled)


    # Calcular las métricas de MSE y MAE para la regresión lineal
    mse_linear = mean_squared_error (y_test, y_pred_linear)
    mae_linear = mean_absolute_error (y_test, y_pred_linear)

    # Calcular las métricas de MSE y MAE para la regresión de Ridge
    mse_ridge = mean_squared_error (y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error (y_test, y_pred_ridge)

    # Comparar los resultados
    print(f"Linear Regression - MSE: {mse_linear}, MAE: {mae_linear}")
    print(f"Ridge Regression - MSE: {mse_ridge}, MAE: {mae_ridge}")

    print ("\n- El modelo de Ridge da mejores resultados en el set de evaluación.\
           \n  El modelo de Ridge mejora el error cuadrático medio y el error absoluto medio son un poco menores que el modelo lineal.\
           \n  El error que se puede haber reducido es el error de generalización, ya que el modelo de Ridge\
           \n  penaliza los coeficientes de los atributos, evitando el sobreajuste del modelo de regresión lineal.\n")







if __name__ == "__main__":
    """Application entry point"""
    try:

        punto6 ()

    except KeyboardInterrupt:
        print ("Finish...")
        try:
            sys.exit (0)
        except SystemExit:
            os._exit (0)