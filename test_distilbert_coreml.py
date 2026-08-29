import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

print("Torch:", torch.__version__)
print("CoreMLTools:", ct.__version__)

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

inputs = tokenizer(
    ["hello world", "test sentence"],
    return_tensors="pt",
    padding="max_length",
    max_length=256,
    truncation=True,
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)


class DistilBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


wrapped_model = DistilBertWrapper(model)
wrapped_model.eval()

# Verify wrapper
with torch.no_grad():
    pytorch_output = wrapped_model(
        input_ids,
        attention_mask,
    )

print("PyTorch output:", pytorch_output.shape)

# Trace tensor-only output
print("\nTracing model...")

traced_model = torch.jit.trace(
    wrapped_model,
    (input_ids, attention_mask),
)

print("TorchScript tracing: SUCCESS")

# Core ML conversion
print("\nConverting to Core ML...")

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=input_ids.shape,
            dtype=int,
        ),
        ct.TensorType(
            name="attention_mask",
            shape=attention_mask.shape,
            dtype=int,
        ),
    ],
    convert_to="mlprogram",
)

print("\nCORE ML CONVERSION: SUCCESS")

output_path = "DistilBERT_test.mlpackage"
mlmodel.save(output_path)

print("Saved:", output_path)
