from tkinter import *
from tkinter import filedialog
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# =========================================
# MAIN WINDOW
# =========================================

root = Tk()
root.title("Emotion Recognition From Speech")
root.geometry("950x700")
root.config(bg="#0f172a")

# =========================================
# DUMMY TRAINING DATA
# =========================================

X_train = np.random.rand(100, 40)

y_train = np.random.choice(
    ["Happy", "Sad", "Angry", "Neutral"],
    100
)

# Encode Labels
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y_train)

# =========================================
# DEEP LEARNING MODEL
# =========================================

model = Sequential()

model.add(Dense(
    128,
    activation="relu",
    input_shape=(40,)
))

model.add(Dense(
    64,
    activation="relu"
))

model.add(Dense(
    4,
    activation="softmax"
))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
model.fit(
    X_train,
    y_encoded,
    epochs=5,
    verbose=0
)

# =========================================
# EXTRACT MFCC FEATURES
# =========================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(
        mfcc.T,
        axis=0
    )

    return mfcc_scaled

# =========================================
# PREDICT EMOTION
# =========================================

def predict_emotion():

    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav")]
    )

    if file_path:

        features = extract_features(file_path)

        features = np.expand_dims(
            features,
            axis=0
        )

        prediction = model.predict(
            features,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction
        )

        emotion = encoder.inverse_transform(
            [predicted_class]
        )[0]

        result_label.config(
            text=f"Detected Emotion: {emotion}",
            fg="#22c55e"
        )

# =========================================
# HEADER
# =========================================

top_frame = Frame(
    root,
    bg="#111827",
    height=100
)

top_frame.pack(fill="x")

Label(
    top_frame,
    text="EMOTION RECOGNITION FROM SPEECH",
    font=("Arial", 28, "bold"),
    bg="#111827",
    fg="#38bdf8"
).pack(pady=25)

# =========================================
# MAIN CARD
# =========================================

main_frame = Frame(
    root,
    bg="#1e293b"
)

main_frame.pack(
    pady=60,
    ipadx=50,
    ipady=40
)

# =========================================
# TITLE
# =========================================

Label(
    main_frame,
    text="Upload Speech Audio File (.wav)",
    font=("Arial", 22, "bold"),
    bg="#1e293b",
    fg="white"
).pack(pady=20)

# =========================================
# BUTTON
# =========================================

Button(
    main_frame,
    text="UPLOAD AUDIO",
    command=predict_emotion,
    font=("Arial", 16, "bold"),
    bg="#06b6d4",
    fg="black",
    padx=25,
    pady=12,
    bd=0,
    cursor="hand2"
).pack(pady=25)

# =========================================
# RESULT LABEL
# =========================================

result_label = Label(
    root,
    text="",
    font=("Arial", 30, "bold"),
    bg="#0f172a"
)

result_label.pack(pady=40)

# =========================================
# FOOTER
# =========================================

Label(
    root,
    text="Developed Using Deep Learning, MFCC & TensorFlow",
    font=("Arial", 12),
    bg="#0f172a",
    fg="#94a3b8"
).pack(side=BOTTOM, pady=15)

# =========================================
# RUN APP
# =========================================

root.mainloop()