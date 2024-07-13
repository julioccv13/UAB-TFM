import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess.preprocess import process_file
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
#import pymrmr

df, symptoms = process_file("sivigila_dengue.csv")

#Conteo de datos faltantes
rows_zero = df[symptoms].eq(0).all(axis=1).sum()
total_df = len(df)
missing_symptoms_df = total_df - rows_zero
print("full df " + str(total_df))
print("missing symptoms df " + str(missing_symptoms_df))
#Pacientes hospitalizados
df.hospitalizacion.value_counts()