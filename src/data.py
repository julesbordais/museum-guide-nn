"""Dataset: WikiArt subset (Hugging Face `huggan/wikiart`), classified by art STYLE.

Run once:  python -m src.data --per_class 300
Creates data/wikiart/{train,val,test}/<style>/*.jpg  (random 70/15/15 split per image, seed 42)
plus data/wikiart/metadata.csv (path, split, style, artist, genre) used by the
question-artwork similarity module.
"""
import argparse, csv, io
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .common import DATA, seed_everything

STYLES = ["Impressionism", "Baroque", "Cubism", "Ukiyo_e", "Pop_Art", "Renaissance_style"]
HF_STYLES = {"Impressionism": "Impressionism", "Baroque": "Baroque", "Cubism": "Cubism",
             "Ukiyo_e": "Ukiyo_e", "Pop_Art": "Pop_Art", "Renaissance_style": "High_Renaissance"}
ROOT = DATA / "wikiart"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats


def download(per_class: int, seed: int = 42):
    from datasets import load_dataset, Image as HFImage
    from PIL import Image as PILImage
    import random
    random.seed(seed)
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    feats = ds.features
    ds = ds.cast_column("image", HFImage(decode=False))   # decode only the images we keep
    style_names, artist_names, genre_names = (feats["style"].names, feats["artist"].names,
                                              feats["genre"].names)
    wanted = {style_names.index(v): k for k, v in HF_STYLES.items()}
    counts, rows = defaultdict(int), []
    for ex in ds:
        s = ex["style"]
        if s not in wanted or counts[s] >= per_class:
            continue
        label = wanted[s]
        i = counts[s]; counts[s] += 1
        r = random.random()
        split = "train" if r < .70 else "val" if r < .85 else "test"
        out = ROOT / split / label; out.mkdir(parents=True, exist_ok=True)
        p = out / f"{label}_{i:04d}.jpg"
        img = PILImage.open(io.BytesIO(ex["image"]["bytes"])).convert("RGB")
        img.resize((256, 256)).save(p, quality=90)
        rows.append([str(p.relative_to(ROOT)), split, label,
                     artist_names[ex["artist"]], genre_names[ex["genre"]]])
        if all(counts[k] >= per_class for k in wanted):
            break
        if sum(counts.values()) % 200 == 0:
            print(dict(counts))
    with open(ROOT / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "split", "style", "artist", "genre"]); w.writerows(rows)
    print("done", dict(counts))


def tf_mlp(size=64):
    """Small images for the feed-forward net (flattened 3*64*64 = 12 288 inputs)."""
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def tf_resnet(train: bool):
    if train:
        return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                   transforms.Normalize(MEAN, STD)])
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def loaders(kind="mlp", batch_size=64, workers=2):
    out = {}
    for split in ["train", "val", "test"]:
        tf = tf_mlp() if kind == "mlp" else tf_resnet(split == "train")
        ds = datasets.ImageFolder(ROOT / split, transform=tf)
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                num_workers=workers)
    return out, out["train"].dataset.classes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per_class", type=int, default=300)
    a = ap.parse_args(); seed_everything(); download(a.per_class)
