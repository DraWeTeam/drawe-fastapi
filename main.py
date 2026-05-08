import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import CLIPProcessor, CLIPModel

# 환경변수 로드
load_dotenv()

# 환경변수 읽기 (기본값 제공)
MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
DEVICE_PREF = os.getenv("DEVICE", "auto")  # auto / cuda / cpu

app = FastAPI()

# device 결정
if DEVICE_PREF == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE_PREF

print(f"CLIP 모델 로딩 중... model={MODEL_NAME}, device={device}")
print("첫 실행 시 다운로드에 5-10분 소요")

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print(f"로드 완료. device: {device}")


class TextEmbedRequest(BaseModel):
    text: str


class TextEmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": MODEL_NAME}


@app.post("/embed/text", response_model=TextEmbedResponse)
def embed_text(req: TextEmbedRequest):
    inputs = processor(
        text=[req.text], return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        elif hasattr(text_features, "last_hidden_state"):
            text_features = text_features.last_hidden_state[:, 0, :]

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embedding = text_features[0].cpu().tolist()
    return TextEmbedResponse(embedding=embedding, dimension=len(embedding))
