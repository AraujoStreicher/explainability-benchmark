import os
import shutil
import random
from collections import defaultdict

BASE_DIR = "./data/CUB-200-2011/CUB_200_2011"
IMAGES_DIR = os.path.join(BASE_DIR, "images")

TRAIN_DIR = "./data/CUB-200-2011/images/train"
TEST_DIR = "./data/CUB-200-2011/images/test"

TRAIN_RATIO = 0.8 
SEED = 42

random.seed(SEED)



IMAGES_FILE = os.path.join(BASE_DIR, "images.txt")
images_by_class = defaultdict(list)

with open(IMAGES_FILE, "r") as f:
    for line in f:
        _, rel_path = line.strip().split(" ")
        class_name = rel_path.split("/")[0]  # Ex: '001.Black_footed_Albatross'
        images_by_class[class_name].append(rel_path)

print(f"Encontradas {len(images_by_class)} classes.")


os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for class_name, img_list in images_by_class.items():
    random.shuffle(img_list)

    n_train = int(len(img_list) * TRAIN_RATIO)
    train_imgs = img_list[:n_train]
    test_imgs = img_list[n_train:]

    # Cria pastas das classes
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copiar arquivos
    for rel_path in train_imgs:
        src = os.path.join(IMAGES_DIR, rel_path)
        dst = os.path.join(TRAIN_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    for rel_path in test_imgs:
        src = os.path.join(IMAGES_DIR, rel_path)
        dst = os.path.join(TEST_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

print("Nova divisão criada com sucesso!")
print(f"Treino: {TRAIN_DIR}")
print(f"Teste: {TEST_DIR}")