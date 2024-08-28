import os
import random
import numpy as np
from skimage import io, transform
import nengo
import nengo_dl
import tensorflow as tf

# Funkcja do ładowania losowych zdjęć z gwarancją stałej liczby próbek
def load_random_images_and_labels(base_path, img_size=(28, 28), num_train=100, num_test=50):
    images = []
    labels = []

    # Zakres katalogów, z których będziemy losować
    directories = [f"video_{i:04d}" for i in range(1, 347)]
    
    # Zbieranie plików z istniejących katalogów
    all_files = []
    for sub_dir in directories:
        sub_dir_path = os.path.join(base_path, sub_dir)
        if os.path.exists(sub_dir_path):
            files = [f for f in os.listdir(sub_dir_path) if f.endswith(".png")]
            all_files.extend([(sub_dir_path, f) for f in files])
    
    # Upewnienie się, że mamy wystarczającą liczbę plików
    if len(all_files) < num_train + num_test:
        raise ValueError(f"Za mało zdjęć w katalogach, znaleziono tylko {len(all_files)} plików.")
    
    # Losowe wybieranie zdjęć do zestawu treningowego i testowego
    selected_files = random.sample(all_files, num_train + num_test)
    
    for sub_dir_path, file in selected_files:
        label = int(file.split('-')[-1].split('.')[0])  # Wyciąganie etykiety na podstawie nazwy pliku

        # Wczytywanie obrazu
        image = io.imread(os.path.join(sub_dir_path, file))
        image = transform.resize(image, img_size)  # Przeskalowanie do 28x28 pikseli
        image = image.flatten()  # Spłaszczenie obrazu
        
        images.append(image)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Podział na zestawy treningowe i testowe
    return images[:num_train], labels[:num_train], images[num_train:], labels[num_train:]

# Ścieżka do katalogu z danymi
base_path = r"C:\Users\krzys\OneDrive\Pulpit\jaad\dataset_jaad\bad_weather"

# Losowanie obrazów i przypisywanie etykiet
X_train, y_train, X_test, y_test = load_random_images_and_labels(base_path)

# Dodanie wymiaru czasowego
X_train = X_train[:, None, :]  # Kształt: (batch_size, n_steps, dimensions)
X_test = X_test[:, None, :]    # Kształt: (batch_size, n_steps, dimensions)
y_train = y_train[:, None, None]  # Kształt: (batch_size, n_steps, 1)
y_test = y_test[:, None, None]    # Kształt: (batch_size, n_steps, 1)

# Budowa modelu spikingowej sieci neuronowej
with nengo.Network() as net:
    # Wejście do sieci - obraz jako wejście
    inp = nengo.Node(output=nengo.processes.PresentInput(X_train, presentation_time=0.1))
    
    # Warstwa ukryta - neurony LIF
    hidden = nengo.Ensemble(
        n_neurons=X_train.shape[-1],  # Liczba neuronów dopasowana do liczby pikseli
        dimensions=1,                 # Przekształcenie jednowymiarowe
        neuron_type=nengo.LIF(),      # Neurony spikingowe LIF
    )
    
    # Połączenie warstwy wejściowej z ukrytą
    nengo.Connection(inp, hidden.neurons, synapse=0.1)
    
    # Warstwa wyjściowa
    out = nengo.Probe(hidden.neurons, synapse=0.1)  # Monitorowanie aktywności

# Klasa do logowania postępu trenowania
class LogProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: loss = {logs['loss']:.4f}")
        if 'accuracy' in logs:
            print(f" - accuracy = {logs['accuracy']:.4f}")

# Symulacja modelu przy użyciu NengoDL z określonym `minibatch_size`
with nengo_dl.Simulator(net, minibatch_size=50, seed=42) as sim:
    # Konfiguracja treningu
    sim.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    # Trenowanie modelu z logowaniem postępu
    sim.fit({inp: X_train}, {out: y_train}, epochs=10, callbacks=[LogProgress()])
    
    # Ewaluacja modelu na danych testowych z takim samym `minibatch_size`
    results = sim.evaluate({inp: X_test}, {out: y_test})
    loss = results[0]
    accuracy = results[1]
    print(f"Test accuracy: {accuracy:.4f}")
