import sys
import os

import common

import matplotlib.pyplot as plt
import seaborn as sns

def punto1 () -> None:

    """

    1. Obtener la correlación entre los atributos y los atributos con el target. 
       ¿Cuál atributo tiene mayor correlación lineal con el target y cuáles atributos 
       parecen estar más correlacionados entre sí? Se puede obtener los valores o 
       directamente graficar usando un mapa de calor.
    
    """

    X_train_scaled, _ = common.get_X_train_test_scaled ()
    df_california = common.get_data_frame ()
    correlation_matrix = None

    sns.pairplot(data=df_california, diag_kind="kde");

    print ("\n\n-> Punto 1: \n")

    # 1. 
    # 1.1. Obtener la correlación entre los atributos
    correlation_matrix = X_train_scaled.corr()

    plt.figure (figsize=(10, 10))
    sns.heatmap (correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title ("Correlacion entre atributos")
    plt.show ()

    # 1.2. Obtener la correlación entre los atributos y el target
    correlation_matrix = df_california.corr ()['MedHouseVal']             # Obtenemos la correlación con el target
    correlation_matrix = correlation_matrix.drop ('MedHouseVal')          # Eliminamos el target
    correlation_matrix = correlation_matrix.sort_values (ascending=False) # Ordenamos de mayor a menor

    plt.figure (figsize=(10, 5))
    correlation_matrix.plot (kind='bar', color='black')
    plt.title ("Correlacion entre atributos y target")
    plt.ylabel ("Corr MedHouseVal")
    plt.show ()

    # 1.3. ¿Cuál atributo tiene mayor correlación lineal con el target?
    print (f"\n- El atributo con mayor correlación lineal con el target es: {correlation_matrix.idxmax ()} = {correlation_matrix.max ()}")

    # 1.4. ¿Cuáles atributos parecen estar más correlacionados entre sí?
    print ("\n- Los atributos que parecen estar más correlacionados entre sí son:\n")
    print ("    1. El AveRooms con AveBedrms. Correlación = 0.85\n")
    print ("    2. El AveRooms con MedInc. Correlación = 0.32\n")




if __name__ == "__main__":
    """Application entry point"""
    try:

        punto1 ()

    except KeyboardInterrupt:
        print ("Finish...")
        try:
            sys.exit (0)
        except SystemExit:
            os._exit (0)