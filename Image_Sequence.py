# Adding Image sequence as a separate file to be utilized in
# Food detector training.py and Model Evaluation.py file

import numpy as np
import random
import cv2
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


# -------- Advanced Augmentation Function --------
def augment_image(img):
    # Random horizontal flip
    if random.random() < 0.5:
        img = img[:, ::-1, :]

    # Random brightness adjustment
    if random.random() < 0.3:
        factor = 1.0 + random.uniform(-0.3, 0.3)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # Random rotation (-15 to +15 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Optional: add more, e.g., random zoom, crop, noise, etc.

    return img

# -------- NEW: Preprocessing helper (for app.py & debugging) --------
def preprocess_image(img, target_size=(224, 224), augment=False):
    """Preprocess a single image (numpy array OR file path) for inference/training."""
    if isinstance(img, str):
        # If path is passed, read from disk
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)

    if augment:
        img = augment_image(img)

    # ✅ Use EfficientNet preprocess_input for consistency
    img = preprocess_input(img.astype("float32"))

    return img




# -------- Data generator via Keras Sequence --------
class ImageSequence(Sequence):
    def __init__(self, paths, labels, batch_size=32, img_size=(224, 224), num_classes=1,
                 shuffle=True, augment=False,  **kwargs):
        super().__init__(**kwargs)  # ✅ REQUIRED for compatibility with Keras 3+
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.paths[i] for i in batch_idx]
        batch_y = [self.labels[i] for i in batch_idx]
        batch_imgs = []
        for path in batch_x:
            img = cv2.imread(path)
            if img is None:
                continue
            #img = cv2.resize(img, self.img_size)

            # ---- AUGMENTATION applied only if self.augment is True ----
            #if self.augment:
            #    img = augment_image(img)

            #img = preprocess_input(img)
            img = preprocess_image(img, target_size=self.img_size, augment=self.augment)
            batch_imgs.append(img)
        X = np.array(batch_imgs, dtype="float32")
        y = to_categorical(np.array(batch_y), num_classes=self.num_classes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


