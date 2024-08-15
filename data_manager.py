
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Cargamos el dataset de viviendas de California
__california_housing = None
__df_california = None
__fearure_names = None

__X, __y = None, None

__X_train, __X_test, __y_train, __y_test = None, None, None, None

__X_train_scaled, __X_test_scaled = None, None


__model = None


def __load_data () -> None:
    
    global __california_housing, __df_california, __fearure_names
    global __X, __y, __X_train, __X_test, __y_train, __y_test
    global __X_train_scaled, __X_test_scaled

    if __california_housing is None:

        # Cargamos el dataset de viviendas de California
        __california_housing = fetch_california_housing ()

        __fearure_names = __california_housing['feature_names']

        # Obtenemos los atributos y target transformados en Pandas
        __X = pd.DataFrame (__california_housing.data, columns=__california_housing['feature_names'])
        __y = pd.Series (__california_housing.target, name=__california_housing['target_names'][0])

        # Unimos a X e y, esto ayuda a la parte de la gráfica del mapa de calor de correlación
        __df_california = pd.concat ([__X, __y], axis=1)

        # Mostramos los primeros registros
        print (f"\nData: \n{__X.head ()}")
        print (f"\nTargets: \n{__y.head ()}")

        # Se separa el dataset en entrenamiento y evaluación
        __X_train, __X_test, __y_train, __y_test = train_test_split (__X, __y, test_size=0.3, random_state=42)

        # Se escalan los datos
        scaler = StandardScaler ()
        __X_train_scaled = pd.DataFrame (scaler.fit_transform (__X_train), columns=__fearure_names)
        __X_test_scaled = pd.DataFrame (scaler.transform (__X_test), columns=__fearure_names)



def get_X_train_test_scaled () -> list:

    if __X_train_scaled is None or __X_test_scaled is None:

        __load_data ()

    return [__X_train_scaled, __X_test_scaled]


def get_y_train_test () -> list:

    if __y_train is None or __y_test is None:

        __load_data ()

    return [__y_train, __y_test]


def get_data_frame () -> pd.DataFrame:

    if __df_california is None:

        __load_data ()

    return __df_california


def get_model (X_test, y_test) -> object:

    global __model

    if __model is None:

        try:
            
            with open ('modelo_regresion_lineal.pkl', 'rb') as file:
                __model = pickle.load (file)

        except FileNotFoundError:

            __model = LinearRegression ()
            __model.fit (X_test, y_test)

            with open ('modelo_regresion_lineal.pkl', 'wb') as file:
                pickle.dump(__model, file)

    return __model