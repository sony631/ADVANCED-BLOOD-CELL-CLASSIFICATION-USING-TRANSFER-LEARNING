To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
To **run a web application** for a project like **"HEMATO VISION: Advanced Blood Cell Classification using Transfer Learning"**, you typically need:

1. A trained deep learning model (e.g., using TensorFlow/Keras or PyTorch)
2. A web app frontend/backend (usually Flask, FastAPI, or Django for backend)
3. Code to load the model and perform inference
4. UI to upload images and see results

---

### ✅ Here's a basic **Flask** web app implementation:

Let’s break it down step by step:

---

## 🔧 1. **Project Structure**

```
hematovision/
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   └── hematovision_model.h5  ← trained model
├── app.py
└── requirements.txt
```

---

## 📦 2. `requirements.txt`

```txt
Flask
tensorflow
numpy
Pillow
```

---

## 🧠 3. Model Training (summary)

You need a model trained using **transfer learning**, like this:

```python
# train_model.py (just for training, not in app)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assume 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your blood cell dataset here...
# model.fit(...)

model.save("model/hematovision_model.h5")
```

---

## 🌐 4. `app.py` (Flask Web App)

```python
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/hematovision_model.h5'

model = load_model(MODEL_PATH)
classes = ['Red Blood Cell', 'White Blood Cell', 'Platelet']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖼️ 5. `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>HematoVision</title>
</head>
<body>
    <h2>Upload Blood Cell Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
        <img src="{{ img_path }}" width="300">
    {% endif %}
</body>
</html>
```

---

### ▶️ How to Run

1. Make sure you’re in the `hematovision/` directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser at: [http://localhost:5000](http://localhost:5000)

---

### 🧪 Need Help with Model Training or Dataset?

If you need a ready dataset (e.g., **BCCD**: Blood Cell Count Dataset), I can guide you through loading and training on it.

---

Would you like this code as a downloadable zip, or do you want help adapting it to a specific dataset (e.g., BCCD)?
