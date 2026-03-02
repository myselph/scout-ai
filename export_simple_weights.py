"""
One-time script to convert SimplePolicyNet .pth weights to numpy .npz format.
Requires PyTorch. Run once; the output .npz file is used by numpy_neural_player.py.
"""
import os
import torch
import numpy as np


def export(pth_path: str, npz_path: str) -> None:
    state_dict = torch.load(pth_path, weights_only=True, map_location="cpu")
    arrays = {k: v.numpy() for k, v in state_dict.items()}
    np.savez(npz_path, **arrays)
    print(f"Saved {len(arrays)} tensors to {npz_path}:")
    for k, v in arrays.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(base_dir, "simple_agent_weights")
    for fname in os.listdir(weights_dir):
        if fname.endswith(".pth"):
            stem = fname[:-4]
            pth_path = os.path.join(weights_dir, fname)
            npz_path = os.path.join(weights_dir, stem + ".npz")
            print(f"\nExporting {fname} ...")
            export(pth_path, npz_path)
