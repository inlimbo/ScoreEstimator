import trees
import csv
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import pandas as pd

tree = [1,2,3]
bush = [4,5,6]
tree = tree + bush

def printTree():
    for i in tree:
        print(i)

new_trees = trees.Trees()
new_trees.add_tree("Mahogany")
new_trees.print_trees()

with open("StudentsPerformance.csv") as f:
    file_data = csv.DictReader(f)


train_dataset_fp = "StudentsPerformance.csv"


column_names = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']


feature_names = column_names[:-3]
math_score = 'math_score'
print(math_score)
print(feature_names)

#feature_df = pd.DataFrame(feature_names)
#feature_onehot = pd.get_dummies(feature_df)

batch_size = 100

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=feature_names,
    label_name=math_score,
    num_epochs=1)

features, labels = next(iter(train_dataset))

def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(5,)),  # input shape required
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.01)

model.compile(loss = 'mse',
            optimizer = optimizer,
            metrics=['mae','mse'])

