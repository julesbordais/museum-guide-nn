"""Part 4: sentence embeddings for visitor-question -> artwork similarity.

Each artwork gets a catalogue text (genre, artist, style description). Visitor questions
are generated with paraphrase templates in *visitor* vocabulary, so lexical overlap with
the catalogue style text is low. Metrics: Recall@1, Recall@5, MRR, latency per query.
Neural: all-MiniLM-L6-v2 (384-d, cosine). Classical: TF-IDF 1-2 grams + cosine.
"""
import random, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .common import seed_everything, save_json
from .data import ROOT

STYLE_DESC = {
    "Impressionism": "loose visible brushstrokes, outdoor light, everyday modern life, 19th-century France",
    "Baroque": "dramatic chiaroscuro, rich colour, movement and grandeur, 17th-century religious and court art",
    "Cubism": "fragmented geometric planes, multiple viewpoints at once, early 20th-century avant-garde",
    "Ukiyo_e": "Japanese woodblock prints, flat colour areas, landscapes, actors and courtesans, Edo period",
    "Pop_Art": "mass culture, advertising and comics imagery, bold flat colours, 1950s-60s",
    "Renaissance_style": "linear perspective, balanced composition, classical ideals, Italian High Renaissance",
}
STYLE_Q = {
    "Impressionism": ["paintings with sunlight and quick dabs of paint", "pictures of Paris cafés and gardens with blurry strokes"],
    "Baroque": ["very dark dramatic paintings with strong spotlight", "theatrical religious scenes from the 1600s"],
    "Cubism": ["paintings where faces are broken into shapes", "art that shows objects from several angles"],
    "Ukiyo_e": ["old Japanese prints of waves and mountains", "Japanese art with flat colours made from wood blocks"],
    "Pop_Art": ["art that looks like comic books or adverts", "colourful pictures of soup cans and celebrities"],
    "Renaissance_style": ["Italian paintings with perfect perspective", "harmonious classical paintings from Florence and Rome"],
}
TEMPLATES = ["Where can I see {q} by {a}?", "Do you have {q}? I like {a}.",
             "I'm looking for {a}'s work, the {q} kind.", "Show me {q} — something from {a}."]


def pretty(a):
    return a.replace("-", " ").title()


def build():
    df = pd.read_csv(ROOT / "metadata.csv")
    df = df[df.artist != "Unknown Artist"].reset_index(drop=True)
    df["text"] = [f"{g.replace('_',' ')} by {pretty(a)}. Style: {s.replace('_',' ')} — {STYLE_DESC[s]}."
                  for a, s, g in zip(df.artist, df["style"], df.genre)]
    rng = random.Random(0); qs = []
    for (a, s), _ in df.groupby(["artist", "style"]):
        for _ in range(2):
            qs.append(dict(q=rng.choice(TEMPLATES).format(q=rng.choice(STYLE_Q[s]), a=pretty(a)),
                           artist=a, style=s))
    return df, pd.DataFrame(qs)


def score(S, df, qdf, k=5):
    r1 = r5 = mrr = 0.0
    for i, row in qdf.iterrows():
        order = np.argsort(-S[i])
        ok = (df.artist.values[order] == row.artist) & (df["style"].values[order] == row["style"])
        hits = np.where(ok)[0]
        rank = hits[0] + 1 if len(hits) else 10**9
        r1 += rank == 1; r5 += rank <= k; mrr += 1 / rank
    n = len(qdf)
    return dict(recall_at_1=r1 / n, recall_at_5=r5 / n, mrr=mrr / n)


def main():
    seed_everything()
    df, qdf = build()
    print(f"{len(df)} artworks, {len(qdf)} questions. Example: {qdf.q.iloc[0]}")
    out = {}
    t = time.time(); vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(df.text)
    D = vec.transform(df.text); fit_s = time.time() - t
    t = time.time(); S = cosine_similarity(vec.transform(qdf.q), D); q_ms = (time.time() - t) / len(qdf) * 1e3
    out["tfidf"] = dict(**score(S, df, qdf), index_build_s=fit_s, ms_per_query=q_ms, vocab=len(vec.vocabulary_))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t = time.time(); D = m.encode(df.text.tolist(), normalize_embeddings=True, batch_size=64); fit_s = time.time() - t
    t = time.time(); Q = m.encode(qdf.q.tolist(), normalize_embeddings=True); q_ms = (time.time() - t) / len(qdf) * 1e3
    S2 = Q @ D.T
    out["minilm"] = dict(**score(S2, df, qdf), index_build_s=fit_s, ms_per_query=q_ms, dim=int(D.shape[1]))
    out["hybrid_0.5"] = score(0.5 * S + 0.5 * S2, df, qdf)
    for name, M in [("tfidf", S), ("minilm", S2)]:
        out[name]["top1_style_acc"] = float(np.mean(
            [df["style"].values[np.argmax(M[i])] == qdf["style"].iloc[i] for i in range(len(qdf))]))
    out["examples"] = [dict(question=qdf.q.iloc[i], minilm_top1=df.text.iloc[int(np.argmax(S2[i]))],
                            tfidf_top1=df.text.iloc[int(np.argmax(S[i]))]) for i in range(5)]
    save_json(out, "qa_results.json")


if __name__ == "__main__":
    main()
