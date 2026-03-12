import torch

obj_embeddings = torch.load('object_embeddings.pt')
pred_embeddings = torch.load('predicate_embeddings.pt')

print(f"Object embeddings shape: {obj_embeddings.shape} (Expected: 36, 512)")
print(f"Predicate embeddings shape: {pred_embeddings.shape} (Expected: 133, 512)")

assert obj_embeddings.shape[-1] == 512, f"Expected 512 channels, got {obj_embeddings.shape[-1]}"
assert pred_embeddings.shape[-1] == 512, f"Expected 512 channels, got {pred_embeddings.shape[-1]}"

print("Dimensions correctly match the model's expected 512 input channels.")