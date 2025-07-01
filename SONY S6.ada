To implement **data visualization** for your project *"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"*, you’ll want to include visual tools that show how your model works, how the data is distributed, and how predictions are made.

Here’s a full example with Python code (using `matplotlib`, `seaborn`, and `plotly`) for **visualizing blood cell data and model performance**.

---

## ✅ Visualization Features to Include

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| Dataset sample viewer        | Show example images of RBC, WBC, Platelets   |
| Class distribution plot      | Show how many samples per class              |
| Model training metrics       | Accuracy & loss curves during training       |
| Confusion matrix             | Visualize classification accuracy per class  |
| Grad-CAM heatmaps (optional) | Show what part of image the model focuses on |

---

## 🧪 1. **Dataset Sample Viewer**

```python
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img

def plot_sample_images(dataset_path):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        img_file = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size=(128, 128))
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.suptitle("Sample Images per Class")
    plt.show()
```

📌 **Usage**:

```python
plot_sample_images("dataset/train")
```

---

## 📊 2. **Class Distribution Plot**

```python
import seaborn as sns
import pandas as pd

def plot_class_distribution(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        count = len(os.listdir(os.path.join(dataset_path, class_name)))
        data.append({'Class': class_name, 'Count': count})

    df = pd.DataFrame(data)
    sns.barplot(x='Class', y='Count', data=df)
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of Images")
    plt.show()
```

📌 **Usage**:

```python
plot_class_distribution("dataset/train")
```

---

## 📈 3. **Model Training Curves (Accuracy & Loss)**

Assuming you saved the training history:

```python
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()
```

📌 **Usage**:

```python
# Assume you have the `history` object from model.fit()
plot_training_history(history)
```

---

## 🤖 4. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
```

📌 **Usage**:

```python
# y_true = true class indices
# y_pred = predicted class indices
plot_confusion_matrix(y_true, y_pred, ['RBC', 'WBC', 'Platelet'])
```

---

## 🔥 5. (Optional) **Grad-CAM: Visualizing What the Model Sees**

Use Grad-CAM to see which part of the image the model focuses on.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

To visualize:

```python
def display_gradcam(img_path, model, last_conv_layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
```

---

## 🔚 Summary

| Visualization          | Code Ready   |
| ---------------------- | ------------ |
| Dataset preview        | ✅            |
| Class distribution     | ✅            |
| Training accuracy/loss | ✅            |
| Confusion matrix       | ✅            |
| Grad-CAM (attention)   | ✅ (optional) |

---

Would you like all this wrapped in a Jupyter notebook or integrated into your web app interface with interactive buttons?
CTo implement **data visualization** for your project *"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"*, you’ll want to include visual tools that show how your model works, how the data is distributed, and how predictions are made.

Here’s a full example with Python code (using `matplotlib`, `seaborn`, and `plotly`) for **visualizing blood cell data and model performance**.

---

## ✅ Visualization Features to Include

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| Dataset sample viewer        | Show example images of RBC, WBC, Platelets   |
| Class distribution plot      | Show how many samples per class              |
| Model training metrics       | Accuracy & loss curves during training       |
| Confusion matrix             | Visualize classification accuracy per class  |
| Grad-CAM heatmaps (optional) | Show what part of image the model focuses on |

---

## 🧪 1. **Dataset Sample Viewer**

```python
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img

def plot_sample_images(dataset_path):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        img_file = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size=(128, 128))
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.suptitle("Sample Images per Class")
    plt.show()
```

📌 **Usage**:

```python
plot_sample_images("dataset/train")
```

---

## 📊 2. **Class Distribution Plot**

```python
import seaborn as sns
import pandas as pd

def plot_class_distribution(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        count = len(os.listdir(os.path.join(dataset_path, class_name)))
        data.append({'Class': class_name, 'Count': count})

    df = pd.DataFrame(data)
    sns.barplot(x='Class', y='Count', data=df)
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of Images")
    plt.show()
```

📌 **Usage**:

```python
plot_class_distribution("dataset/train")
```

---

## 📈 3. **Model Training Curves (Accuracy & Loss)**

Assuming you saved the training history:

```python
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()
```

📌 **Usage**:

```python
# Assume you have the `history` object from model.fit()
plot_training_history(history)
```

---

## 🤖 4. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
```

📌 **Usage**:

```python
# y_true = true class indices
# y_pred = predicted class indices
plot_confusion_matrix(y_true, y_pred, ['RBC', 'WBC', 'Platelet'])
```

---

## 🔥 5. (Optional) **Grad-CAM: Visualizing What the Model Sees**

Use Grad-CAM to see which part of the image the model focuses on.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

To visualize:

```python
def display_gradcam(img_path, model, last_conv_layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
```

---

## 🔚 Summary

| Visualization          | Code Ready   |
| ---------------------- | ------------ |
| Dataset preview        | ✅            |
| Class distribution     | ✅            |
| Training accuracy/loss | ✅            |
| Confusion matrix       | ✅            |
| Grad-CAM (attention)   | ✅ (optional) |

---

Would you like all this wrapped in a Jupyter notebook or integrated into your web app interface with interactive buttons?
To implement **data visualization** for your project *"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"*, you’ll want to include visual tools that show how your model works, how the data is distributed, and how predictions are made.

Here’s a full example with Python code (using `matplotlib`, `seaborn`, and `plotly`) for **visualizing blood cell data and model performance**.

---

## ✅ Visualization Features to Include

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| Dataset sample viewer        | Show example images of RBC, WBC, Platelets   |
| Class distribution plot      | Show how many samples per class              |
| Model training metrics       | Accuracy & loss curves during training       |
| Confusion matrix             | Visualize classification accuracy per class  |
| Grad-CAM heatmaps (optional) | Show what part of image the model focuses on |

---

## 🧪 1. **Dataset Sample Viewer**

```python
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img

def plot_sample_images(dataset_path):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        img_file = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size=(128, 128))
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.suptitle("Sample Images per Class")
    plt.show()
```

📌 **Usage**:

```python
plot_sample_images("dataset/train")
```

---

## 📊 2. **Class Distribution Plot**

```python
import seaborn as sns
import pandas as pd

def plot_class_distribution(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        count = len(os.listdir(os.path.join(dataset_path, class_name)))
        data.append({'Class': class_name, 'Count': count})

    df = pd.DataFrame(data)
    sns.barplot(x='Class', y='Count', data=df)
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of Images")
    plt.show()
```

📌 **Usage**:

```python
plot_class_distribution("dataset/train")
```

---

## 📈 3. **Model Training Curves (Accuracy & Loss)**

Assuming you saved the training history:

```python
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()
```

📌 **Usage**:

```python
# Assume you have the `history` object from model.fit()
plot_training_history(history)
```

---

## 🤖 4. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
```

📌 **Usage**:

```python
# y_true = true class indices
# y_pred = predicted class indices
plot_confusion_matrix(y_true, y_pred, ['RBC', 'WBC', 'Platelet'])
```

---

## 🔥 5. (Optional) **Grad-CAM: Visualizing What the Model Sees**

Use Grad-CAM to see which part of the image the model focuses on.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

To visualize:

```python
def display_gradcam(img_path, model, last_conv_layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
```

---

## 🔚 Summary

| Visualization          | Code Ready   |
| ---------------------- | ------------ |
| Dataset preview        | ✅            |
| Class distribution     | ✅            |
| Training accuracy/loss | ✅            |
| Confusion matrix       | ✅            |
| Grad-CAM (attention)   | ✅ (optional) |

---

Would you like all this wrapped in a Jupyter notebook or integrated into your web app interface with interactive buttons?
To implement **data visualization** for your project *"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"*, you’ll want to include visual tools that show how your model works, how the data is distributed, and how predictions are made.

Here’s a full example with Python code (using `matplotlib`, `seaborn`, and `plotly`) for **visualizing blood cell data and model performance**.

---

## ✅ Visualization Features to Include

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| Dataset sample viewer        | Show example images of RBC, WBC, Platelets   |
| Class distribution plot      | Show how many samples per class              |
| Model training metrics       | Accuracy & loss curves during training       |
| Confusion matrix             | Visualize classification accuracy per class  |
| Grad-CAM heatmaps (optional) | Show what part of image the model focuses on |

---

## 🧪 1. **Dataset Sample Viewer**

```python
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img

def plot_sample_images(dataset_path):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        img_file = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size=(128, 128))
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.suptitle("Sample Images per Class")
    plt.show()
```

📌 **Usage**:

```python
plot_sample_images("dataset/train")
```

---

## 📊 2. **Class Distribution Plot**

```python
import seaborn as sns
import pandas as pd

def plot_class_distribution(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        count = len(os.listdir(os.path.join(dataset_path, class_name)))
        data.append({'Class': class_name, 'Count': count})

    df = pd.DataFrame(data)
    sns.barplot(x='Class', y='Count', data=df)
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of Images")
    plt.show()
```

📌 **Usage**:

```python
plot_class_distribution("dataset/train")
```

---

## 📈 3. **Model Training Curves (Accuracy & Loss)**

Assuming you saved the training history:

```python
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()
```

📌 **Usage**:

```python
# Assume you have the `history` object from model.fit()
plot_training_history(history)
```

---

## 🤖 4. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
```

📌 **Usage**:

```python
# y_true = true class indices
# y_pred = predicted class indices
plot_confusion_matrix(y_true, y_pred, ['RBC', 'WBC', 'Platelet'])
```

---

## 🔥 5. (Optional) **Grad-CAM: Visualizing What the Model Sees**

Use Grad-CAM to see which part of the image the model focuses on.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

To visualize:

```python
def display_gradcam(img_path, model, last_conv_layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
```

---

## 🔚 Summary

| Visualization          | Code Ready   |
| ---------------------- | ------------ |
| Dataset preview        | ✅            |
| Class distribution     | ✅            |
| Training accuracy/loss | ✅            |
| Confusion matrix       | ✅            |
| Grad-CAM (attention)   | ✅ (optional) |

---

Would you like all this wrapped in a Jupyter notebook or integrated into your web app interface with interactive buttons?
To implement **data visualization** for your project *"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"*, you’ll want to include visual tools that show how your model works, how the data is distributed, and how predictions are made.

Here’s a full example with Python code (using `matplotlib`, `seaborn`, and `plotly`) for **visualizing blood cell data and model performance**.

---

## ✅ Visualization Features to Include

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| Dataset sample viewer        | Show example images of RBC, WBC, Platelets   |
| Class distribution plot      | Show how many samples per class              |
| Model training metrics       | Accuracy & loss curves during training       |
| Confusion matrix             | Visualize classification accuracy per class  |
| Grad-CAM heatmaps (optional) | Show what part of image the model focuses on |

---

## 🧪 1. **Dataset Sample Viewer**

```python
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img

def plot_sample_images(dataset_path):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        img_file = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size=(128, 128))
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.suptitle("Sample Images per Class")
    plt.show()
```

📌 **Usage**:

```python
plot_sample_images("dataset/train")
```

---

## 📊 2. **Class Distribution Plot**

```python
import seaborn as sns
import pandas as pd

def plot_class_distribution(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        count = len(os.listdir(os.path.join(dataset_path, class_name)))
        data.append({'Class': class_name, 'Count': count})

    df = pd.DataFrame(data)
    sns.barplot(x='Class', y='Count', data=df)
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of Images")
    plt.show()
```

📌 **Usage**:

```python
plot_class_distribution("dataset/train")
```

---

## 📈 3. **Model Training Curves (Accuracy & Loss)**

Assuming you saved the training history:

```python
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()
```

📌 **Usage**:

```python
# Assume you have the `history` object from model.fit()
plot_training_history(history)
```

---

## 🤖 4. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
```

📌 **Usage**:

```python
# y_true = true class indices
# y_pred = predicted class indices
plot_confusion_matrix(y_true, y_pred, ['RBC', 'WBC', 'Platelet'])
```

---

## 🔥 5. (Optional) **Grad-CAM: Visualizing What the Model Sees**

Use Grad-CAM to see which part of the image the model focuses on.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

To visualize:

```python
def display_gradcam(img_path, model, last_conv_layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
```

---

## 🔚 Summary

| Visualization          | Code Ready   |
| ---------------------- | ------------ |
| Dataset preview        | ✅            |
| Class distribution     | ✅            |
| Training accuracy/loss | ✅            |
| Confusion matrix       | ✅            |
| Grad-CAM (attention)   | ✅ (optional) |

---

Would you like all this wrapped in a Jupyter notebook or integrated into your web app interface with interactive buttons?
To implement **data visualization** for your project *"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"*, you’ll want to include visual tools that show how your model works, how the data is distributed, and how predictions are made.

Here’s a full example with Python code (using `matplotlib`, `seaborn`, and `plotly`) for **visualizing blood cell data and model performance**.

---

## ✅ Visualization Features to Include

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| Dataset sample viewer        | Show example images of RBC, WBC, Platelets   |
| Class distribution plot      | Show how many samples per class              |
| Model training metrics       | Accuracy & loss curves during training       |
| Confusion matrix             | Visualize classification accuracy per class  |
| Grad-CAM heatmaps (optional) | Show what part of image the model focuses on |

---

## 🧪 1. **Dataset Sample Viewer**

```python
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img

def plot_sample_images(dataset_path):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        img_file = random.choice(os.listdir(class_folder))
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size=(128, 128))
        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.suptitle("Sample Images per Class")
    plt.show()
```

📌 **Usage**:

```python
plot_sample_images("dataset/train")
```

---

## 📊 2. **Class Distribution Plot**

```python
import seaborn as sns
import pandas as pd

def plot_class_distribution(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        count = len(os.listdir(os.path.join(dataset_path, class_name)))
        data.append({'Class': class_name, 'Count': count})

    df = pd.DataFrame(data)
    sns.barplot(x='Class', y='Count', data=df)
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of Images")
    plt.show()
```

📌 **Usage**:

```python
plot_class_distribution("dataset/train")
```

---

## 📈 3. **Model Training Curves (Accuracy & Loss)**

Assuming you saved the training history:

```python
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()
```

📌 **Usage**:

```python
# Assume you have the `history` object from model.fit()
plot_training_history(history)
```

---

## 🤖 4. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
```

📌 **Usage**:

```python
# y_true = true class indices
# y_pred = predicted class indices
plot_confusion_matrix(y_true, y_pred, ['RBC', 'WBC', 'Platelet'])
```

---

## 🔥 5. (Optional) **Grad-CAM: Visualizing What the Model Sees**

Use Grad-CAM to see which part of the image the model focuses on.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

To visualize:

```python
def display_gradcam(img_path, model, last_conv_layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
```

---

## 🔚 Summary

| Visualization          | Code Ready   |
| ---------------------- | ------------ |
| Dataset preview        | ✅            |
| Class distribution     | ✅            |
| Training accuracy/loss | ✅            |
| Confusion matrix       | ✅            |
| Grad-CAM (attention)   | ✅ (optional) |

---

Would you like all this wrapped in a Jupyter notebook or integrated into your web app interface with interactive buttons?
