from ane_transformers.huggingface.test_distilbert import TestDistilBertForSequenceClassification
import torch

print("=== Building test fixture ===")

TestDistilBertForSequenceClassification.setUpClass()

obj = TestDistilBertForSequenceClassification

print("\n=== self.inputs ===")
for name, tensor in obj.inputs.items():
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

print("\n=== self.inputs_list ===")
for i, tensor in enumerate(obj.inputs_list):
    print(f"[{i}] shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

print("\n=== Tracing ===")
traced = torch.jit.trace(obj.models['test'], obj.inputs_list)

print("\n=== TorchScript graph inputs ===")
graph_inputs = list(traced.graph.inputs())

for i, inp in enumerate(graph_inputs):
    print(f"[{i}] {inp}")
    print(f"     type = {inp.type()}")

print("\n=== TorchScript graph ===")
print(traced.graph)
