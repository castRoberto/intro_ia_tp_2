import sys
import os

import common

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def punto4 () -> None:

    """
    
    4. Calcular las métricas de MSE, MAE y $R^2$ del set de evaluación.
    
    """

    X_train_scaled, X_test_scaled = common.get_X_train_test_scaled ()
    y_train, y_test = common.get_y_train_test ()

    print ("\n\n-> Punto 4: \n")

    regresion = common.get_linear_model (X_train_scaled, y_train)
    y_pred = regresion.predict (X_test_scaled)

    # 4.

    # 4.1. Calcular las métricas de MSE, MAE y R^2 del set de evaluación
    mse = mean_squared_error (y_test, y_pred)
    mae = mean_absolute_error (y_test, y_pred)
    r2 = r2_score (y_test, y_pred)

    print (f"\n- MSE: {mse}")
    print (f"\n- MAE: {mae}")
    print (f"\n- R^2: {r2}\n")


if __name__ == "__main__":
    """Application entry point"""
    try:

        punto4 ()

    except KeyboardInterrupt:
        print ("Finish...")
        try:
            sys.exit (0)
        except SystemExit:
            os._exit (0)