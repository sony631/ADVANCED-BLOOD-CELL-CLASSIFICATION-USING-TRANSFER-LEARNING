img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
Here's a comprehensive guide to building a **blood cell classification model** using **transfer learning**, under the project title:

---

# **HEMATO VISION: Advanced Blood Cell Classification Using Transfer Learning**

---

## 🧠 **Overview**

Blood cell classification is crucial in diagnosing diseases such as leukemia, anemia, infections, and more. Deep learning, especially **transfer learning**, significantly improves accuracy in classifying images of different blood cell types.

This project will:

* Use a pre-trained CNN (e.g., ResNet50 or EfficientNet).
* Train it on a dataset of blood cell images.
* Classify different types (e.g., neutrophils, eosinophils, lymphocytes, monocytes).
* Achieve high accuracy with minimal training time due to transfer learning.

---

## 📁 **Step 1: Dataset**

Use a labeled blood cell dataset like:

* **Blood Cell Count and Detection** (from Kaggle)

  * Classes: Neutrophil, Eosinophil, Monocyte, Lymphocyte
  * Image size: \~300x300 px

📦 Example directory structure:

```
/blood_cells
  /train
    /neutrophil
    /eosinophil
    /monocyte
    /lymphocyte
  /val
    ...
```

---

## ⚙️ **Step 2: Install Required Libraries**

```bash
pip install tensorflow keras matplotlib scikit-learn
```

---

## 🧪 **Step 3: Code for Transfer Learning**

### ✅ Load Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
```

---

### 📊 Load and Augment Data

```python
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'blood_cells/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'blood_cells/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

---

### 🧠 Load ResNet50 + Custom Layers

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 🏋️ Train the Model

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

### 📈 Plot Results

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

### 🔍 Evaluate Model

```python
val_generator.reset()
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

---

## 📦 Optional: Save the Model

```python
model.save("hematovision_model.h5")
```

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
VV

---

## ✅ **Project Deliverables**

* Python code notebook/script
* Pre-trained model file (`.h5`)
* Accuracy and loss plots
* Confusion matrix
* Performance report

---

## 🚀 Future Improvements

* Fine-tune more layers in ResNet50
* Try EfficientNet or MobileNet for speed
* Use Grad-CAM for visual explanations
* Deploy with Streamlit or Flask as a web app

---

Would you like:

* A Colab-ready notebook?
* Streamlit UI code for live prediction?
* Dataset link from Kaggle?

Let me know what you want next!
