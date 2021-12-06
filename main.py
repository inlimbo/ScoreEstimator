import csv
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import pandas as pd
import printdot

dataset_fp = "StudentsPerformance.csv"

column_names = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']

feature_names = column_names[:-1]
label_name = column_names[-1]

train_dataset_fp = pd.read_csv(dataset_fp, names = column_names, na_values = "?", comment ='\t',
                                    sep = " ", skipinitialspace = True)

gender = train_dataset_fp.pop('gender')
train_dataset_fp['female'] = (gender == 1)*1.0
train_dataset_fp['male'] = (gender == 2)*1.0
train_dataset_fp.tail()

race_ethnicity = train_dataset_fp.pop('race/ethnicity')
train_dataset_fp['group A'] = (race_ethnicity == 1)*1.0
train_dataset_fp['group B'] = (race_ethnicity == 2)*1.0
train_dataset_fp['group C'] = (race_ethnicity == 3)*1.0
train_dataset_fp['group D'] = (race_ethnicity == 4)*1.0
train_dataset_fp['group E'] = (race_ethnicity == 5)*1.0

parental_level_of_education = train_dataset_fp.pop('parental_level_of_education')
train_dataset_fp['some high school'] = (parental_level_of_education == 1)*1.0
train_dataset_fp['high school'] = (parental_level_of_education == 2)*1.0
train_dataset_fp['college'] = (parental_level_of_education == 3)*1.0
train_dataset_fp["associate's degree"] = (parental_level_of_education == 4)*1.0
train_dataset_fp["bachelor's degree"] = (parental_level_of_education == 5)*1.0
train_dataset_fp["master's degree"] = (parental_level_of_education == 6)*1.0

lunch = train_dataset_fp.pop('lunch')
train_dataset_fp['standard'] = (lunch == 1)*1.0
train_dataset_fp['free/reduced'] = (lunch == 2)*1.0

test_preparation_course = train_dataset_fp.pop('test_preparation_course')
train_dataset_fp['none'] = (test_preparation_course == 1)*1.0
train_dataset_fp['completed'] = (test_preparation_course == 2)*1.0

train_dataset = train_dataset_fp.sample(frac=0.85)
test_dataset = train_dataset_fp.drop(train_dataset.index)

train_lables = train_dataset.pop('math score', 'writing score', 'reading score')
test_labels = test_dataset.pop('math score', 'writing score', 'reading score')



#feature_df = pd.DataFrame(feature_names)
#feature_onehot = pd.get_dummies(feature_df)

batch_size = 100
EPOCHS = 100

""""
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=feature_names,
    label_name=label_name,
    num_epochs=EPOCHS)
"""
#features, labels = next(iter(train_dataset))

def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=len[train_dataset.keys()]),  # input shape required
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss = 'mse',
            optimizer = optimizer,
            metrics=['mae','mse'])


train_loss_results = []
train_accuracy_results = []

history = model.fit(
    train_dataset, feature_names,
    epochs = EPOCHS, validation_split = 0.15, verbose = 0,
    callbacks=[PrintDot()]
)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

