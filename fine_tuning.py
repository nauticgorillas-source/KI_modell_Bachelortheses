from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import random
import itertools
import numpy as np
import os
import math
import torch

#--->
#Grundkonfiguration
#--->

FAELLE_JSON = "faelle.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUTH_PATH = "custom_model"

# hier wollen wir auf die CPU ansetzen
try:
    torch.set_num_threads(4)
except Exception:
    pass


#--->
# Daten laden
#--->
with open(FAELLE_JSON, "r", encoding="utf-8") as f:
    faelle = json.load(f)
print(f"Fälle geladen: {len(faelle)}")


#--->
#Scoring: wie ähnlich sind zwei Fälle? (Schwerpunkt + Stufenabstand)
#--->
def calc_score(a, b):
    if a["schwerpunkt"] == b["schwerpunkt"]:
        diff = abs(int(a["stufe"]) - int(b["stufe"]))
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.8
        elif diff == 2:
            return 0.6
        else:
            return 0.4
    else:
        return 0.0


#--->
#Train/Test-Split (80/20), nach Schwerpunkt stratifiziert (wenn sinnvoll)
#--->
schwerpunkt = [f["schwerpunkt"] for f in faelle]
train_ids, test_ids = train_test_split(
    list(range(len(faelle))),
    test_size=0.20,
    random_state=42,
    stratify=schwerpunkt if len(set(schwerpunkt)) > 1 else None
)
train_set = [faelle[i] for i in train_ids]
test_set  = [faelle[i] for i in test_ids]
print(f"Aufteilung (Fall-basiert)): {len(train_set)} Train / {len(test_set)} Test")


#--->
#Paarbildung: positive (gleiches Thema, stufengewichtet) + negative (anderes Thema)
#neg_ratio = Verhältnis Negaitv/Positiv
#--->
def make_pairs(cases, neg_ratio=1.0, seed=42):

    random.seed(seed)
    by_sp = {}
    for c in cases:
        by_sp.setdefault(c["schwerpunkt"],[]).append(c)

    #Positive Paare (jede Kombi innerhalb eines Schwerpunkts)
    positives = []
    for sp, group in by_sp.items():
        for a, b in itertools.combinations(group, 2):
            s = calc_score(a, b)
            if s > 0.0:
                positives.append(InputExample(texts=[a["fallschilderung"], b["fallschilderung"]], label=s))

    #negative Paare erzeugen (verschiedene Schwerpunkte)
    neg_needed = int(len(positives) * neg_ratio) if len(positives) > 0 else 0
    negatives = []
    all_cases = cases[:]
    if neg_needed > 0:
        tries = 0
        max_tries = neg_needed * 10
        while len(negatives) < neg_needed and tries < max_tries:
            tries +=1
            a = random.choice(all_cases)
            b = random.choice(all_cases)
            if a is b:
                continue
            if a["schwerpunkt"] != b["schwerpunkt"]:
                negatives.append(InputExample(texts=[a["fallschilderung"], b["fallschilderung"]], label=0.0))
    exampels = positives + negatives
    random.shuffle(exampels)
    return exampels, len(positives), len(negatives)

train_exampels, pos_train, neg_train = make_pairs(train_set, neg_ratio=1.0)
test_exampels, pos_test, neg_test = make_pairs(test_set, neg_ratio=1.0)
print(f"Generierte Trainingspaare: {len(train_exampels)}: {pos_train} /: {neg_train}")
print(f"Generierte Testpaare: {len(test_exampels)}: {pos_test} /: {neg_test}")


#--->
#Modell aufsetzen (Transformer + pooling)
#--->
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#--->
#Training vorbereiten
#--->
batch_size = 12 # and Hardware anpasen
train_dataloader = DataLoader(
    train_exampels, shuffle=True, batch_size=batch_size,
    num_workers=0, pin_memory=False
)
#erwartet labels im bereich 0.0 - 1.0
train_loss = losses.CosineSimilarityLoss(model=model)     


#---->
#Trainingsparameter
#---->
TOTAL_EPOCHE = 20
steps_per_epoch = math.ceil(len(train_exampels) / max(1, batch_size))
WARMUP_STEPS = max(100, int(0.10 * steps_per_epoch * TOTAL_EPOCHE))
print("Training startet")

#--->
#Training
#--->
model.fit(
    train_objectives= [(train_dataloader, train_loss)],
    epochs=TOTAL_EPOCHE,
    warmup_steps=WARMUP_STEPS,
    show_progress_bar=True,
    use_amp=False,
    optimizer_params= {"lr": 2e-5}
)
model.save(OUTPUTH_PATH)
print(f"Modell gespechert nach {OUTPUTH_PATH}")


#---->
#hier wollen wir unsere Evaluation aufbauen
#wie gut das Modell stufengewichtet gelernt hat
#wie gut es positive/negative paare trennt
#wie stark die trennschärfe insgesamt ist
#wie gut das modell mit binärem schwellenewrt arbeitet
#---->
print("Evaluation läuft")

def _pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    x = x - x.mean(); y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / denom) if denom > 0 else 0.0

def _spearman(x, y ):
    x = np.asarray(x); y = np.asanyarray(y)
    rx = np.argsort(np.argsirt(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx, ry)


scores_true = []
sims_pred   = []
y_true_bin  = []
for ex in test_exampels:
    text_a, text_b = ex.texts
    emb_a = model.encode(text_a)
    emb_b = model.encode(text_b)
    sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    # Cosine-Ähnlichkeit der Embeddings
    sims_pred.append(sim)
    scores_true.append(float(ex.label))
    y_true_bin.append(1 if ex.label > 0.5 else 0)

#Schwelle 0.5 - einfach, nachvollziehabr
y_pred_bin = [1 if s > 0.5 else 0 for s in sims_pred]

acc  = accuracy_score(y_true_bin, y_pred_bin)
prec = precision_score(y_true_bin, y_pred_bin, zero_division= 0)
rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)

#AUCs können bei rein 0/1-Labels mit Scores stabiler sein als mit harten klassen
try:
    roc = roc_auc_score(y_true_bin, sims_pred)
except Exception:
    roc = float("nan")
try:
    pr_auc = average_precision_score(y_true_bin, sims_pred)
except Exception:
    pr_auc = float("nan")

cm = confusion_matrix(y_true_bin, y_pred_bin)

pear  = _pearson(sims_pred, scores_true)
spear = _spearman(sims_pred, scores_true)

quantiles = np.percentile(sims_pred, [1, 5, 25, 50, 75, 95, 99])



print("\n📈 Evaluationsergebnisse (Test-Set):")
print(f"Accuracy:  {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1-Score:  {f1:.2f}")
print(f"ROC-AUC:   {roc:.3f}")
print(f"PR-AUC:    {pr_auc:.3f}")
print(f"Pearson r: {pear:.3f}")
print("Konfusionsmatrix [[TN, FP],[FN, TP]]:")
print (cm)
print("Cosine-Quantile [1, 5, 25, 50, 75, 95, 99]%:")
print([round(q, 3) for q in quantiles])