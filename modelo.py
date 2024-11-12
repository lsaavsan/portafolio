import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# cargar dataset de pinguinos
df = sns.load_dataset("penguins")

# codificar variable objetivo
encoder1=LabelEncoder()
df['species'] = encoder1.fit_transform(df['species'])

# división en variables características (X) y variable objetivo (y)
X = df.drop(columns=['species'])
y = df['species']

# división del en X_train, y_train, X_test e y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# imputación de columnas categóricas
categorical_columns = ['island','sex']
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train.loc[:, categorical_columns] = cat_imputer.fit_transform(X_train[categorical_columns])
X_test.loc[:, categorical_columns] = cat_imputer.transform(X_test[categorical_columns])

# imputación de columnas numéricas
numerical_columns = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
num_imputer = SimpleImputer(strategy='mean')
X_train.loc[:, numerical_columns] = num_imputer.fit_transform(X_train[numerical_columns])
X_test.loc[:, numerical_columns] = num_imputer.transform(X_test[numerical_columns])


variables_numericas = X_train.select_dtypes(include=['float64','int64']).columns
scaler = StandardScaler()
X_train[variables_numericas] = scaler.fit_transform(X_train[variables_numericas])
X_test[variables_numericas] = scaler.transform(X_test[variables_numericas])


variables_categoricas = X_train.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, drop = 'first')
X_train_encoded = encoder.fit_transform(X_train[variables_categoricas])
X_test_encoded = encoder.transform(X_test[variables_categoricas])


X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(variables_categoricas))
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(variables_categoricas))

X_train_prepared = pd.concat([X_train[variables_numericas].reset_index(drop=True), X_train_encoded_df.reset_index(drop=True)], axis=1)
X_test_prepared =  pd.concat([X_test[variables_numericas].reset_index(drop=True), X_test_encoded_df.reset_index(drop=True)], axis=1)


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
        n_estimators = 100,            # Establece el número de árboles en el bosque a 100.
           max_depth = 5,              # Limita la profundidad máxima de cada árbol a 5 para evitar el sobreajuste.
        class_weight = 'balanced',     # Ajusta los pesos de las clases para manejar el desbalance en los datos; 
                                       # las clases menos frecuentes recibirán más peso.
        random_state = 42              # Establece una semilla aleatoria para asegurar la reproducibilidad del modelo.
)

random_forest.fit(X_train_prepared, y_train)


import pickle

# Guardar el modelo RandomForest
pickle.dump(random_forest, open('rf_pinguinos.pkl', 'wb'))

# Guardar el OneHotEncoder
pickle.dump(encoder, open('encoder_pinguinos.pkl', 'wb'))

# Guardar el StandardScaler
pickle.dump(scaler, open('scaler_pinguinos.pkl', 'wb'))

# Guardar el DataFrame `X_train_prepared` completo para referencia en predicciones
pickle.dump(X_train_prepared, open('X_train_prepared_pinguinos.pkl', 'wb'))