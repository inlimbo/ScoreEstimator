import csv
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import pandas as pd
import printdot

dataset_fp = "StudentsPerformance.csv"

column_names = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']


train_dataset_fp = pd.read_csv(dataset_fp, names = column_names, na_values = "?", comment ='\t',
                                    sep = ",", skipinitialspace = True)

print(train_dataset_fp)
gender = train_dataset_fp.pop('gender')
train_dataset_fp['female'] = (gender == 'female')*1.0
train_dataset_fp['male'] = (gender == 'male')*1.0

race_ethnicity = train_dataset_fp.pop('race/ethnicity')
train_dataset_fp['group A'] = (race_ethnicity == 'group A')*1.0
train_dataset_fp['group B'] = (race_ethnicity == 'group B')*1.0
train_dataset_fp['group C'] = (race_ethnicity == 'group C')*1.0
train_dataset_fp['group D'] = (race_ethnicity == 'group D')*1.0
train_dataset_fp['group E'] = (race_ethnicity == 'group E')*1.0

parental_level_of_education = train_dataset_fp.pop('parental_level_of_education')
train_dataset_fp['some high school'] = (parental_level_of_education == 'some high school')*1.0
train_dataset_fp['high school'] = (parental_level_of_education == 'high school')*1.0
train_dataset_fp['college'] = (parental_level_of_education == 'college')*1.0
train_dataset_fp["associate's degree"] = (parental_level_of_education == "associate's degree")*1.0
train_dataset_fp["bachelor's degree"] = (parental_level_of_education == "bachelor's degree")*1.0
train_dataset_fp["master's degree"] = (parental_level_of_education == "master's degree")*1.0

lunch = train_dataset_fp.pop('lunch')
train_dataset_fp['standard'] = (lunch == 'standard')*1.0
train_dataset_fp['free/reduced'] = (lunch == 'free/reduced')*1.0

test_preparation_course = train_dataset_fp.pop('test_preparation_course')
train_dataset_fp['none'] = (test_preparation_course == 'none')*1.0
train_dataset_fp['completed'] = (test_preparation_course == 'completed')*1.0

train_dataset = train_dataset_fp.sample(frac=0.85)
test_dataset = train_dataset_fp.drop(train_dataset.index)

print(train_dataset_fp)
train_labels = train_dataset.pop('writing_score')
#train_labels = train_labels.pop('writing_score')
#train_labels = train_labels.pop('reading_score')
test_labels = test_dataset.pop('math_score')

print(train_labels)

train_stats = train_dataset.describe()
#train_stats.pop('math_score')
#train_stats.pop('writing_score')
#train_stats.pop('reading_score')
train_stats = train_stats.transpose()

#feature_df = pd.DataFrame(feature_names)
#feature_onehot = pd.get_dummies(feature_df)

batch_size = 100
EPOCHS = 100

model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),  # input shape required
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss = 'mse',
            optimizer = optimizer,
            metrics=['mae','mse'])


train_loss_results = []
train_accuracy_results = []

def norm(x):
    return(x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)

history = model.fit(
    normed_train_data, train_labels,
    epochs = EPOCHS, validation_split = 0.15, verbose = 0,
    callbacks=[printdot.PrintDot()]
)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

