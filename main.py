# import useful libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
from typing import List

# Global settings
TRAIN_CSV_URL = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
TRAIN_IMG_URL_BASE = """https://s3.amazonaws.com/
                    google-landmark/train/images_{}.tar"""
BASE_TAR_NAME = "images_{}.tar"
TAR_FILE_NUM = 500
NUM_PRALLEL = 1
BATCH_SIZE = 32
IMAGE_SHAPE = [192, 192]

# Download data by keras api
for num in range(TAR_FILE_NUM):
    if num < 10:
        num_str = '00' + str(num)
    elif num < 100:
        num_str = '0' + str(num)
    else:
        num_str = str(num)
    img_data_url = TRAIN_IMG_URL_BASE.format(num_str)
    # Download data
    keras.utils.get_file(fname=BASE_TAR_NAME.format(
        num_str), origin=img_data_url, extract=True)
# Download tran.csv
train_csv_path = keras.utils.get_file(fnamm='train.csv', origin=TRAIN_CSV_URL)
base_path = Path(train_csv_path).parent

# Preprocessing the data

# Step 1: get the image_path_list and label_list, label_num
train_df = pd.read_csv(train_csv_path)
label_num = train_df['landmark_id'].max()
image_count = train_df.shape[0]
image_path_list = []
label_list = []

for index in range(image_count):
    row = train_df.iloc[index]
    each_id = row['id']
    each_label = row['landmark_id']
    each_path = base_path / each_id[0] / \
        each_id[1] / each_id[2] / (each_id + '.jpg')
    image_path_list.append(each_path)
    label_list.append(each_label)

# Step 1 end

# Step 2, prepare the tf.Dataset

# Define image loading and preprocessing funciton
# TODO: You can apply any data enhancement method, like scale, rotate. etc....
# please check the tensorflow alpha api document


def load_and_preprocess_image(path: Path) -> tf.Tensor:
    img_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    image = tf.image.resize(image, IMAGE_SHAPE)
    image /= 255.0

    return image


path_ds = tf.data.Dataset.from_tensor_slices(image_path_list)
image_ds = path_ds.map(load_and_preprocess_image)
label_ds = tf.data.Dataset.from_tensor_slices(label_list)
ds = tf.data.Dataset.zip(image_ds, label_ds)

# Step 2: prepare dataset end

# Step 3: setting for dataset
# TODO: change the batch_size and num_pallel for better training result
# Note: batch_size is a global variable, you can change the value above

ds = ds.cache()
ds = ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(NUM_PRALLEL)

# Model and train

# Step 1: Download the pretrain the modle
# TODO: you can change the pretrain the model type
# please check the keras api document

pretrained_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3), include_top=False)
pretrained_model.trainable = False

# the mobile_net require input image data range (-1, 1),
# so we need to change the data range
# if your custom model range is different,
# please comment corresponding lines below


def change_range(image: tf.Tensor, label: tf.Tensor) -> List[tf.Tensor]:
    return 2*image, label


ds = ds.map(change_range)


# Step 2: define our model
# TODO: you change change the last layers to get better accuracy
model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(label_num)])

# Step 2: set up the model
# TODO: change the optim and learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

# Step 4: train the model
model.fit(ds, epochs=1, steps_per_epoch=3, validation_split=0.2)
