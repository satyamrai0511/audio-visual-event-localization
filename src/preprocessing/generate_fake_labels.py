import os
import numpy as np

def generate_labels(audio_dir, out_dir, num_classes=5):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(audio_dir):
        if file.endswith("_logmel.npy"):
            sample_name = file.replace("_logmel.npy", "")
            audio_path = os.path.join(audio_dir, file)
            audio_feat = np.load(audio_path)
            T = audio_feat.shape[1]

            # Random labels for each time step
            labels = np.random.randint(0, num_classes, size=(T,))
            out_path = os.path.join(out_dir, f"{sample_name}_labels.npy")
            np.save(out_path, labels)
            print(f"Saved labels: {out_path} [{T} steps]")

if __name__ == "__main__":
    generate_labels(audio_dir="outputs/features", out_dir="outputs/labels", num_classes=5)
