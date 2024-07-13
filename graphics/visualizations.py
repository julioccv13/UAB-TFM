import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
#import pymrmr

#Visualizacion proporcion de sintomas presentes vs ausentes
def proportions(symptoms, df):
    for symptom in symptoms:
        plt.figure(figsize=(3,3))
        df[symptom].value_counts().plot(kind='pie', rot=0, autopct='%1.1f%%')
        plt.xlabel(symptom)
        plt.ylabel('')
        plt.title('Proporción de pacientes con {}'.format(symptom), fontsize=5)
        plt.text(1.2, 0.5, f'1 = presente \n2 = ausente \n0 = sin dato', transform=plt.gca().transAxes, fontsize=5, verticalalignment='center')
        plt.show()

def hospitalization(df):
    symptoms_1 = df.columns[8:]
    df[symptoms_1] = df[symptoms_1].apply(pd.to_numeric)
    symptoms_df = df[(df[symptoms_1] == 1).any(axis=1)]
    custom_labels = {1: 'hospitalizado', 2: 'no hospitalizado'}

    symptoms_df['hospitalizacion'] = pd.to_numeric(symptoms_df['hospitalizacion'])

    for symptom in symptoms_1:
        positive_symptoms_df = symptoms_df[symptoms_df[symptom] == 1]
        if not positive_symptoms_df.empty:
            plt.figure(figsize=(3,3))
            value_counts = positive_symptoms_df['hospitalizacion'].value_counts().sort_index()
            ax = value_counts.plot(kind='bar')
            ax.set_xticklabels([custom_labels.get(x, str(x)) for x in value_counts.index], rotation=0, fontsize=5)
            plt.title(f'Hospitalizacion vs {symptom} ({symptom} positivo', fontsize=5)
            plt.xlabel(f'{symptom}', fontsize=5)
            plt.ylabel('Count', fontsize=5)
            plt.xticks(rotation=0)
            plt.show()

def distribution(df):
    symptoms_1 = df.columns[8:]
    plt.figure(figsize=(7,10))
    for i, column in enumerate(symptoms_1, 1):
        plt.subplot(6, 3, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribucion de {column}', fontsize=5)
    plt.tight_layout()
    plt.show()

def correlation(df):
    symptoms_1 = df.columns[8:]
    plt.figure(figsize=(7,10))
    custom_labels = {1: 'hospitalizado', 2: 'no hospitalizado'}
    for i, column in enumerate(symptoms_1, 1):
        plt.subplot(6, 3, i)
        sns.countplot(x=column, hue='hospitalizacion', data=df)
        plt.title(f'{column} vs hospitalizacion', fontsize=5)
        plt.legend(title='hospitalizacion', labels=[custom_labels[2], custom_labels[1]], fontsize=5)
    plt.tight_layout()
    plt.show()

def heatmap(df):
    symptoms_1 = df.columns[8:]
    plt.figure(figsize=(6,4))
    correlation_matrix = df[symptoms_1].apply(pd.to_numeric).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Dispersion', fontsize=5)
    plt.show()