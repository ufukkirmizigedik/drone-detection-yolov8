from pathlib import Path
import shutil
import random

# Proje kök dizini (bu dosyanın olduğu yer)
BASE_DIR = Path(__file__).resolve().parent

# Ham veri klasörleri
IMAGES_RAW = BASE_DIR / "dataset" / "images_raw"
LABELS_RAW = BASE_DIR / "dataset" / "labels_raw"

# Hedef klasörler
IMAGES_TRAIN = BASE_DIR / "dataset" / "images" / "train"
IMAGES_VAL   = BASE_DIR / "dataset" / "images" / "val"
LABELS_TRAIN = BASE_DIR / "dataset" / "labels" / "train"
LABELS_VAL   = BASE_DIR / "dataset" / "labels" / "val"

# Klasörleri oluştur
for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    d.mkdir(parents=True, exist_ok=True)

# Kullanılacak resim uzantıları
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Tüm resimleri listele
all_images = [p for p in IMAGES_RAW.iterdir() if p.suffix.lower() in IMAGE_EXTS]

print(f"Toplam resim sayısı: {len(all_images)}")

# Karıştır ve %80 train, %20 val olarak böl
random.seed(42)  # tekrarlanabilirlik için
random.shuffle(all_images)

train_ratio = 0.8
split_index = int(len(all_images) * train_ratio)

train_images = all_images[:split_index]
val_images   = all_images[split_index:]

print(f"Train: {len(train_images)} resim")
print(f"Val  : {len(val_images)} resim")

def copy_pair(img_path: Path, target_img_dir: Path, target_lbl_dir: Path):
    # Görseli kopyala
    dst_img = target_img_dir / img_path.name
    shutil.copy2(img_path, dst_img)

    # Aynı isimli label var mı?
    label_name = img_path.stem + ".txt"
    src_label = LABELS_RAW / label_name
    if src_label.exists():
        dst_label = target_lbl_dir / label_name
        shutil.copy2(src_label, dst_label)
    else:
        # Dronsuz resim → label yok, sorun değil
        print(f"[INFO] Label yok (dronsuz resim olabilir): {img_path.name}")

# Train seti
for img in train_images:
    copy_pair(img, IMAGES_TRAIN, LABELS_TRAIN)

# Val seti
for img in val_images:
    copy_pair(img, IMAGES_VAL, LABELS_VAL)

print("Bölme ve kopyalama tamamlandı.")
