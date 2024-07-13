#%% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
#import pymrmr

def process_file(file):
    #Lectura df
    df = pd.read_csv(file)
    columns = ['id', 'edad_', 'sexo_', 'comuna', 'tipo_ss_', 'fec_con_', 'tip_cas_', 'pac_hos_', 'fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'dolor_abdo', 'vomito', 'somnolenci', 'hipotensio', 'hem_mucosa', 'caida_plaq']
    df = df[columns]

    #Renombrar columnas
    names = {'edad_':'edad', 'sexo_':'sexo', 'tipo_ss_':'seguridad_social', 'fec_con_':'fecha_consulta', 'tip_cas_':'tipo_caso', 'pac_hos_':'hospitalizacion', 'dolrretroo': 'dolor_retro_ocular', 'malgias':'mialgias', 'dolor_abdo':'dolor_abdominal', 'somnolenci':'somnolencia', 'hipotensio':'hipotension', 'hem_mucosa':'sangrado_mucosas', 'caida_plaq':'caida_plaquetas'}
    df.rename(columns=names, inplace=True)

    #Validar tipos de columnas
    symptoms = ['cefalea', 'dolor_retro_ocular', 'mialgias', 'artralgia', 'dolor_abdominal', 'vomito', 'hipotension']
    df[symptoms] = df[symptoms].astype(str)
    df['hospitalizacion'] = df['hospitalizacion'].astype(str)

    #Reemplazar valores nulos
    df.fillna(0, inplace=True)
    df.replace({'SD': 0, pd.NA: 0, 'nan':0}, inplace=True)
    empty_symptoms=['caida_plaquetas', 'somnolencia', 'sangrado_mucosas']

    #Eliminar sintomas vacios
    df = df.drop(columns=empty_symptoms)
    df = df[df.hospitalizacion != 0]
    return df, symptoms
