# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

DATABASE_NAME = 'news_media' 
INPUT_TABLE_NAME = 'horn_africa_model_escbin_emb_confhist_lagpca_m61_gld'
EXP_ID = '3636233935423920'
database_name = "news_media"
#data_table = "horn_africa_model_escbin_emb_confhist_lag_m4_gld"
target_col = "binary_escalation_30"
time_col = "STARTDATE"

# COMMAND ----------

data = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")

# COMMAND ----------

df = data.toPandas()

# COMMAND ----------

### Split train-val-test 60-20-20 on date column
import math

# split date
all_time = df[time_col].unique()
all_time.sort()
train_end = math.ceil(len(all_time) * 0.6)
val_end = math.ceil(len(all_time) * 0.8)
train_dt = all_time[ :train_end]
val_dt = all_time[train_end:val_end]
test_dt = all_time[val_end: ]

# create col for splitting
df['_automl_split_col_0000'] = ''
df.loc[df[time_col].isin(train_dt), '_automl_split_col_0000'] = 'train'
df.loc[df[time_col].isin(val_dt), '_automl_split_col_0000'] = 'val'
df.loc[df[time_col].isin(test_dt), '_automl_split_col_0000'] = 'test'

# COMMAND ----------

df

# COMMAND ----------

df['_automl_sample_weight_0000'] = 1
df.loc[(df['_automl_split_col_0000']=='train') & (df[target_col]==0), '_automl_sample_weight_0000'] = 1.4838198687485855

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
import numpy as np
import pandas as pd

conf_hist = ['Battles', 'Explosions_Remote_violence', 'Protests', 'Riots', 'Strategic_developments', 'Violence_against_civilians']
embed = np.arange(50)
static = ['mean_pop_dense_2020', 'conflict_trend_1', 'conflict_trend_2']

# create t-x column names
conf_hist = [f'{col}_t-{x}' for col in conf_hist for x in np.arange(1,5)]
embed = [f'{col}_t-{x}' for col in embed for x in np.arange(1,5)]
# column selector
supported_cols = conf_hist + embed + static

col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# take out unneeded columns
keep_cols = supported_cols + [target_col, '_automl_sample_weight_0000', '_automl_split_col_0000']
df = df[keep_cols].copy()


# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), supported_cols)) # not really needed, but doing it just in case

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, supported_cols)]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers
preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

df_loaded = df
# AutoML completed train - validation - test split internally and used _automl_split_col_0000 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "test"]

# Separate target column from features and drop _automl_split_col_0000
X_train = split_train_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_test = split_test_df[target_col]



# COMMAND ----------

X_train.shape

# COMMAND ----------

# AutoML balanced the data internally and use _automl_sample_weight_0000 to calibrate the probability distribution
sample_weight = X_train.loc[:, "_automl_sample_weight_0000"].to_numpy()
X_train = X_train.drop(["_automl_sample_weight_0000"], axis=1)
X_val = X_val.drop(["_automl_sample_weight_0000"], axis=1)
X_test = X_test.drop(["_automl_sample_weight_0000"], axis=1)

# COMMAND ----------

# reshaping input for LSTM

data = X_train
windowSize=4
featuresPerWindow=227
X_train_3d=np.zeros((len(data)-windowSize, windowSize ,featuresPerWindow))
for i in range(len(data)-windowSize):
    for j in range(windowSize):
        for k in range(featuresPerWindow):
            X_train_3d[i,j,k]=data.iloc[i+j,k]

data = X_val
windowSize=4
featuresPerWindow=227
X_val_3d=np.zeros((len(data)-windowSize, windowSize ,featuresPerWindow))
for i in range(len(data)-windowSize):
    for j in range(windowSize):
        for k in range(featuresPerWindow):
            X_val_3d[i,j,k]=data.iloc[i+j,k]

data = X_test
windowSize=4
featuresPerWindow=227
X_test_3d=np.zeros((len(data)-windowSize, windowSize ,featuresPerWindow))
for i in range(len(data)-windowSize):
    for j in range(windowSize):
        for k in range(featuresPerWindow):
            X_test_3d[i,j,k]=data.iloc[i+j,k]


# COMMAND ----------

X_train_3d.shape

# COMMAND ----------



# COMMAND ----------

pip install tensorflow

# COMMAND ----------

#pip uninstall tensorflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!pip install scikeras

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.keras
import mlflow.tensorflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow import pyfunc

from hyperopt import hp, tpe, fmin, STATUS_OK, SparkTrials

# COMMAND ----------

pip install --upgrade protobuf

# COMMAND ----------

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

# COMMAND ----------

# model builder
def create_model(layer_choice, units0, units1, dropout1, activation, theoptimizer):
    model = Sequential()
    # input layer
    model.add(LSTM(64, input_shape=(None, 4, 227), return_sequences=False))
    #model.add(TimeDistributed(Dense(227)))
    model.add(Dense(int(units0), input_dim=INPUT_DIM, activation=activation))
    # hidden layers #
    model.add(Dense(int(units1), activation=activation))
    model.add(Dropout(dropout1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10))
    if layer_choice['layers'] == 'two':
        model.add(Dense(int(layer_choice['units2']), activation=activation))
        model.add(Dropout(layer_choice['dropout2']))
    elif layer_choice['layers'] == 'three':
        model.add(Dense(int(layer_choice['units2_']), activation=activation))
        model.add(Dropout(layer_choice['dropout2_']))
        model.add(Dense(int(layer_choice['units3']), activation=activation))
        model.add(Dropout(layer_choice['dropout3']))
    # output layer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=theoptimizer, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# COMMAND ----------

X_train = X_test_3d
X_val = X_val_3d
X_test = X_test_3d
INPUT_DIM = X_train.shape[2]

# COMMAND ----------

def objective(params):
    with mlflow.start_run(experiment_id=EXP_ID) as mlflow_run:
        # classifier
        clf = KerasClassifier(build_fn=create_model, layer_choice=params['choice'], units0=params['units0'], units1=params['units1'], dropout1=params['dropout1'], activation=params['activation'], theoptimizer=params['opt'])
        # build pipeline
        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(
            log_input_examples=True,
            silent=True)

        # fit the model
        model.fit(X_train, y_train, classifier__batch_size=params['batch_size'], classifier__epochs=100, classifier__callbacks=EarlyStopping(patience=10, monitor="val_loss"), classifier__validation_data=(X_val_processed, y_val), classifier__sample_weight=sample_weight)

        # Log metrics for the training set
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = pyfunc.PyFuncModel(model_meta=mlflow_model, model_impl=model)
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(target_col):y_train}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "training_" , "pos_label": 1, "sample_weight": sample_weight }
        )
        training_metrics = training_eval_result.metrics
        # Log metrics for the validation set
        val_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_val.assign(**{str(target_col):y_val}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "val_" , "pos_label": 1 }
        )
        val_metrics = val_eval_result.metrics
        # Log metrics for the test set
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(target_col):y_test}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "test_" , "pos_label": 1 }
        )
        test_metrics = test_eval_result.metrics

        loss = -val_metrics["val_f1_score"]

        # Truncate metric key names so they can be displayed together
        val_metrics = {k.replace("val_", ""): v for k, v in val_metrics.items()}
        test_metrics = {k.replace("test_", ""): v for k, v in test_metrics.items()}

        return {
        "loss": loss,
        "status": STATUS_OK,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model": model,
        "run": mlflow_run,
        }

# COMMAND ----------

space = {'choice': hp.choice('num_layers',
                    [
                        {'layers':'one'},
                        {'layers':'two',
                         'units2': hp.choice('units2', [126, 256, 512, 1024]),
                         'dropout2': hp.uniform('dropout2', .25, .5)
                         },
                        {'layers':'three',
                         'units2_': hp.choice('units2_', [126, 256, 512, 1024]),
                         'dropout2_': hp.uniform('dropout2_', .25, .5),
                         'units3': hp.choice('units3', [126, 256, 512, 1024]),
                         'dropout3': hp.uniform('dropout3', .25, .5)
                         }
                    ]),
         'units0': hp.choice('units0', [126, 256, 512, 1024]),
         'units1': hp.choice('units1', [126, 256, 512, 1024]),
         'dropout1': hp.uniform('dropout1', .25, .5),
         'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
         'opt': hp.choice('optimizer', ['adam', 'rmsprop']),
         'batch_size': hp.choice('batch_size', [16, 32, 64])
        }

# COMMAND ----------

# run trials
trials = SparkTrials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=10,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------



# COMMAND ----------

model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(None, 2072)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))
print(model.summary())

# COMMAND ----------

d = df.iloc[:, 3:]

# COMMAND ----------

d = d.drop(['FATALSUM','abs_change','pct_increase', 'binary_escalation_50', 'binary_escalation_100', 'binary_escalation_5_30', 'binary_escalation_5_50', 'binary_escalation_5_100'], axis=1)

# COMMAND ----------

d

# COMMAND ----------

X = d.drop(['binary_escalation_30'], axis = 1)
Y = d['binary_escalation_30']

# COMMAND ----------

a = int(0.8*(len(X)))
b = int(0.9*(len(X)))
x_train = X[:a]
x_val = X[a:b]
x_test = X[b:]
y_train = Y[:a]
y_val = Y[a:b]
y_test = Y[b:]

# COMMAND ----------

len(x_test)+ len(x_val)+ len(x_train)

# COMMAND ----------

len(X)

# COMMAND ----------

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

# COMMAND ----------

len(X.iloc[1])

# COMMAND ----------

model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=1)

# COMMAND ----------


