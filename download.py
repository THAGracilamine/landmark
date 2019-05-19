import wget

TRAIN_IMG_URL_BASE = "https://s3.amazonaws.com/" + \
    "google-landmark/train/images_{}.tar"
TARGET_IMG_PATH_BASE = "./inputs/google-landmark/image_{}.tar"

for index in range(500):
    if index == 0:
        continue
    elif index < 10:
        continue
    elif index < 100:
        str_index = '0' + str(index)
    else:
        str_index = str(index)
    print('download image_{}.tar'.format(str_index))
    download_url = TRAIN_IMG_URL_BASE.format(str_index)
    target_path = TARGET_IMG_PATH_BASE.format(str_index)
    wget.download(download_url, target_path)
