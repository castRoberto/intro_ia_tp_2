import sys
import os

import data_manager

import matplotlib.pyplot as plt


def punto2 () -> None:

    """
    
    2. Graficar los histogramas de los diferentes atributos y el target. ¿Qué tipo de 
       forma de histograma se observa? ¿Se observa alguna forma de campana que nos 
       indique que los datos pueden provenir de una distribución gaussiana, sin entrar 
       en pruebas de hipótesis?

    """

    df_california = data_manager.get_data_frame ()

    print ("\n\n-> Punto 2: \n")

    # 2.

    # 2.1. Graficar los histogramas de los diferentes atributos y el target
    df_california.hist (bins=50, color='black')
    plt.suptitle ("Histograma de los atributos y el target")
    plt.show ()

    # 2.2. ¿Qué tipo de forma de histograma se observa?
    print ("\n- Se observa una forma de histograma asimétrica positiva en el target.")



if __name__ == "__main__":
    """Application entry point"""
    try:

        punto2 ()

    except KeyboardInterrupt:
        print ("Finish...")
        try:
            sys.exit (0)
        except SystemExit:
            os._exit (0)