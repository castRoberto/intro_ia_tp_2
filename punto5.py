import sys
import os

import data_manager

def punto5 () -> None:

    """
    
    5. Crear una regresión de Ridge. Usando una validación cruzada de 5-folds y usando 
       como métrica el MSE, calcular el mejor valor de $\alpha$, buscando entre [0, 12.5]. 
       Graficar el valor de MSE versus $\alpha$.
    
    """

    pass



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