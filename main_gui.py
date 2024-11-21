import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

rf_arousal = joblib.load("./model/rf_arousal_model_summary.pkl")
rf_valence = joblib.load("./model/rf_valence_model_summary.pkl")

root = tk.Tk()
root.title("Song Arousal and Valence Prediction")
root.geometry("500x500")

song_listbox = tk.Listbox(root, width=50, height=10)
song_listbox.pack(pady=20)

arousal_label = tk.Label(root, text="Arousal: --", font=("Helvetica", 14))
arousal_label.pack(pady=10)
valence_label = tk.Label(root, text="Valence: --", font=("Helvetica", 14))
valence_label.pack(pady=10)

true_labels = []  

emotion_means = {
    "Happy": (0.7, 0.7),
    "Sad": (-0.7, -0.7),
    "Calm": (0.0, -0.7),
    "Anger": (-0.7, 0.7),
    "Joy": (0.9, 0.9),
    "Fear": (-0.9, 0.9),
    "Mixed": (0.0, 0.0),
}

def extract_features_for_song(song_path):
    y, sr = librosa.load(song_path, sr=None)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) 
    energy = float(np.sum(y ** 2) / len(y)) 
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)))))
    
    
    features = [
        tempo, energy, spectral_centroid, zcr, spectral_contrast, 
        tempo, energy, spectral_centroid, zcr, spectral_contrast   # Duplicate 5 features for valence
    ]
    
    return features



def predict_arousal_valence(song_path):
    features = extract_features_for_song(song_path)
    features = np.array(features).reshape(1, -1)
    arousal = rf_arousal.predict(features)[0]
    valence = rf_valence.predict(features)[0]
    return arousal, valence

def get_emotion_label(arousal, valence):
    if -1 <= valence <= -0.5 and -1 <= arousal <= -0.5:
        return "Sad", "#ff7f0e"  
    elif 0 <= valence <= 1 and 0 <= arousal <= 1:
        return "Happy", "#1f77b4"
    elif 0 <= valence <= 1 and -1 <= arousal <= 0:
        return "Calm", "#2ca02c"
    elif -1 <= valence <= 0 and 0 <= arousal <= 1:
        return "Anger", "#d62728"
    elif 0.7 <= valence <= 1 and 0.7 <= arousal <= 1:
        return "Joy", "#9467bd"
    elif -1 <= valence <= -0.7 and 0.7 <= arousal <= 1:
        return "Fear", "#8c564b"
    else:
        return "Mixed", "#7f7f7f"

def plot_arousal_valence(arousal, valence):
    plt.figure(figsize=(6, 6))
    
    emotion, color = get_emotion_label(arousal, valence)
    
    plt.scatter(valence, arousal, color=color, label=f"Emotion: {emotion}", s=100)
    plt.xlabel("Valence (Negative to Positive)")
    plt.ylabel("Arousal (Low to High)")
    plt.title("Arousal vs. Valence for Song")
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.show()

def add_song():
    song_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
    if song_path:
        song_listbox.insert(tk.END, song_path)
        # Dropdown for user to select the true emotion
        emotion_selection = tk.simpledialog.askstring("Input", "Select the emotion (e.g., Happy, Sad, Calm, Anger, Joy, Fear, Mixed):")
        
        if emotion_selection in emotion_means:
            true_labels.append(emotion_selection)
        else:
            messagebox.showerror("Invalid Input", "Please enter a valid emotion label.")

def show_arousal_valence():
    selected_song_idx = song_listbox.curselection()
    if not selected_song_idx:
        messagebox.showwarning("No Song Selected", "Please select a song to analyze.")
        return

    song_path = song_listbox.get(selected_song_idx)
    arousal, valence = predict_arousal_valence(song_path)
    arousal_label.config(text=f"Arousal: {arousal:.2f}")
    valence_label.config(text=f"Valence: {valence:.2f}")
    
    plot_arousal_valence(arousal, valence)

def show_confusion_matrix():
    if not true_labels:
        messagebox.showwarning("No Test Songs", "Please add and label songs to test.")
        return
    
    predicted_labels = []
    for idx in range(song_listbox.size()):
        song_path = song_listbox.get(idx)
        arousal, valence = predict_arousal_valence(song_path)
        predicted_emotion, _ = get_emotion_label(arousal, valence)
        predicted_labels.append(predicted_emotion)
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(emotion_means.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(emotion_means.keys()))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix for Emotion Prediction")
    plt.show()

add_song_button = tk.Button(root, text="Add Song", command=add_song)
add_song_button.pack(pady=10)

analyze_button = tk.Button(root, text="Show Arousal and Valence", command=show_arousal_valence)
analyze_button.pack(pady=10)

confusion_matrix_button = tk.Button(root, text="Show Confusion Matrix", command=show_confusion_matrix)
confusion_matrix_button.pack(pady=10)


root.mainloop()
