import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

class Detector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features

        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.feature_extractor(x)
        f = torch.flatten(f, 1)
        return self.bbox_head(f), self.cls_head(f)

def load_label_map(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)

    label_to_id = data.get("label_to_id", {})
    id_to_label = data.get("id_to_label", {})

    id_to_label_norm = {}
    for k, v in id_to_label.items():
        try:
            id_to_label_norm[int(k)] = v
        except Exception:
            pass

    return label_to_id, id_to_label_norm

def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class InferenceService:
    def __init__(self, model_path: str, label_map_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        self.label_to_id, self.id_to_label = load_label_map(label_map_path)
        num_classes = len(self.label_to_id) if self.label_to_id else len(self.id_to_label)
        if num_classes <= 0:
            raise RuntimeError("label_map.pkl icinden sinif sayisi okunamadi.")

        self.model = Detector(num_classes=num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform = build_transform()

    @torch.no_grad()
    def predict(self, pil_img: Image.Image):
        img = pil_img.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        _, logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        cls_id = int(probs.argmax().item())
        prob = float(probs[cls_id].item())

        label = self.id_to_label.get(cls_id, str(cls_id))
        return label, prob
