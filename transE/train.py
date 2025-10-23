from pykeen.pipeline import pipeline
import torch

result = pipeline(
    dataset="fb15k",
    model="TransE",
    epochs=100,
    model_kwargs={"embedding_dim": 100},
)
torch.save(result.model, "transE_fb15k.pth")

loaded_model = torch.load("transE_fb15k.pth")
entity_embeddings = loaded_model.entity_representations[0]().detach().numpy()