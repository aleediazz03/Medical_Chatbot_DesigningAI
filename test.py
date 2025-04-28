import h5py

filename = "model_weights.h5"

try:
    with h5py.File(filename, 'r') as f:
        print("✅ File opened successfully!")
        print("Contents:", list(f.keys()))
except Exception as e:
    print("❌ File could not be opened:", e)