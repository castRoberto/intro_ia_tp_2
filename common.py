
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
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


# Modelos
__linear_model = None
__best_ridge_model = None

# Ridge data
ALPHA_NUM = 1000
MAX_ALPHA = 12.5

# Array de alphas
__alpha_array = None #np.linspace (0, MAX_ALPHA, ALPHA_NUM)

# MSE para cada alpha
__mse_array = None


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


def get_linear_model (X_train, y_train) -> object:

    global __linear_model

    if __linear_model is None:

        try:
            
            with open ('modelo_regresion_lineal.pkl', 'rb') as file:
                __linear_model = pickle.load (file)

        except FileNotFoundError:

            __linear_model = LinearRegression ()
            __linear_model.fit (X_train, y_train)

            with open ('modelo_regresion_lineal.pkl', 'wb') as file:
                pickle.dump(__linear_model, file)

    return __linear_model



def explore_alphas_with_cross_val (X_train, y_train) -> None:

    global __alpha_array, __mse_array

    if __alpha_array is None or __mse_array is None:

        __alpha_array = np.linspace (0, MAX_ALPHA, ALPHA_NUM)
        __mse_array = []

        # Calculamos los coeficientes para diferentes valores de alpha
        for index, alpha in enumerate(__alpha_array):

            # Creamos el modelo de Ridge
            ridge_model = Ridge (alpha=alpha)

            # Usar cross_val_score para calcular el MSE
            mse = -cross_val_score (ridge_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
            __mse_array.append(mse)

        # Convertir mse_array a numpy array
        __mse_array = np.array(__mse_array)

    return [__alpha_array, __mse_array]



def get_best_alpha (alpha_array, mse_array) -> list:

    # Encontrar el mejor valor de alpha
    best_alpha = alpha_array[np.argmin(mse_array)]
    best_mse = np.min(mse_array)

    return [best_alpha, best_mse]



def get_best_ridge_model (best_alpha, X_train, y_train) -> object:

    global __best_ridge_model

    if __best_ridge_model is None:

        __best_ridge_model = Ridge (alpha=best_alpha)
        __best_ridge_model.fit (X_train, y_train)

    return __best_ridge_model