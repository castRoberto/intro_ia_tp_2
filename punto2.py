import sys
import os

import common

import matplotlib.pyplot as plt


def punto2 () -> None:

    """
    
    2. Graficar los histogramas de los diferentes atributos y el target. ¿Qué tipo de 
       forma de histograma se observa? ¿Se observa alguna forma de campana que nos 
       indique que los datos pueden provenir de una distribución gaussiana, sin entrar 
       en pruebas de hipótesis?

    """

    df_california = common.get_data_frame ()

    print ("\n\n-> Punto 2: \n")

    # 2.

    # 2.1. Graficar los histogramas de los diferentes atributos y el target
    df_california.hist(bins=150)
    plt.suptitle ("Histograma de los atributos y el target")
    plt.show ()

    # 2.2. ¿Qué tipo de forma de histograma se observa?
    print ("\n- Se observa una forma de histograma asimétrica positiva en el target. \
             Se puede considerar que los datos provienen mas o menos de una distribución gaussiana")



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