import json
import re
import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

# --->
# Grundkonfiguration
# --->
PERSIST_DIR     = "./chroma_db"
COLLECTION      = "schwerpunkte"
MODEL_NAME      = "custom_model"
UNTERSTUETZUNG_JSON = "unterstuetzung.json"

N_RESULTS       = 50
TOP_K           = 3
MIN_FILTER_HITS = 10
MAX_REL_INCREASE = 0.35

# --->
#Stichworte pro Schwerpunkt (für den einfachen Keyword-Filter)
# --->
SCHWERPUNKT_KEYWORDS = {
    "Wohnsituation": [r"\bwohnung\b", r"\bwohn", r"\bobdachlos\b", r"\bnotunterkunft\b", r"\bunterkunft\b", r"\bwohnungslos\b", r"\bzwangs ?r?äumung\b"],
    "Gesundheit": [r"\bgesund", r"\btherapie\b", r"\barzt\b", r"\bklinik\b", r"\bpsych", r"\bkrank(heit)?\b", r"\bmedikament", r"\bbehandlung\b"],
    "Sprachkenntnisse": [r"\bsprache\b", r"\bdeutsch\b", r"\bsprachkurs\b", r"\bsprachkenntn", r"\bdolmetsch", r"\bverständigung\b"],
    "Bewerbung und Arbeitsmarktintegration": [r"\bbewerbung\b", r"\barbeit\b", r"\bjob\b", r"\bpraktikum\b", r"\bausbildung\b", r"\blebenslauf\b", r"\bvorstellungsgespräch\b", r"\bjobcenter\b"],
    "Alltagsmanagement": [r"\balltag\b", r"\borganisation\b", r"\bstruktur\b", r"\btermin\b", r"\bhaushalt\b", r"\bhygiene\b", r"\bbehörde\b"],
    "Wirtschaftliche Verhältnisse": [r"\bgeld\b", r"\bfinanz", r"\bschuld(en)?\b", r"\bkonto\b", r"\barmut\b", r"\bbürgergeld\b", r"\bsozialhilfe\b", r"\bmiete\b"],
    "Mobilität": [r"\bmobilität\b", r"\bauto\b", r"\bführerschein\b", r"\bfahrkarte\b", r"\bbus\b", r"\bzug\b", r"\bfahrtkosten\b", r"\btransport\b"],
    "Familie und soziales Umfeld": [r"\bfamilie\b", r"\bkind(er)?\b", r"\bpartner\b", r"\beltern\b", r"\bfreunde\b", r"\bkonflikt\b", r"\btrauma\b", r"\bisoliert\b"],
}

# --->
#Laden der Resourcen
# --->
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_collection():
    return PersistentClient(path=PERSIST_DIR).get_collection(COLLECTION)

@st.cache_resource(show_spinner=False)
def load_unterstuetzung():
    with open(UNTERSTUETZUNG_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Key klein + getrimmt, damit Zugriff robust sind    
    return {str(k).strip().lower(): v for k, v in raw.items()}


# --->
#Schwerpunkt grob über KEywords erkennen
# --->
def detect_schwerpunkte(text: str):
    t = text.lower()
    return [sp for sp, pats in SCHWERPUNKT_KEYWORDS.items() if any(re.search(p, t) for p in pats)]

#--->
#hier bauen wir eine partnerstufen-logik auf
# --->
def partner_logik(stufe: int) -> Tuple[int, int]:
    lo = ((stufe - 1) // 2) * 2 + 1
    return lo, lo + 1


# --->
#Unterstützungstexte für Schwerpunkt + Stufe holen
#(liefert ggf. Hinweise, wenn nichts hinterlegt ist)
# --->
def get_unterstuetzung(u_data, schwerpunkt: str, stufe: int):
    d = u_data.get((schwerpunkt or "").strip().lower())
    if not d:
        return ["Keine Unterstützung definiert."]
    if stufe <= 0:
        return ["Keine Stufe im Treffer hinterlegt."]
    lo, hi = partner_logik(stufe)
    out = []
    #Beide Partnerstufen prüfen
    for s in (lo, hi):
        txt = d.get(str(s))
        if txt and str(txt).strip():
            out.append(f"Stufe {s}: {str(txt).strip()}")
    return out or ["Für diese Stufe ist aktuell kein Unterstützungstext hinterlegt."]


#--->
#Ähnlichkeitssuche: zuerst gefiltert (falls Schwerpunkt erkannt), sonst global
#--->
def query(collection, qvec, detected):
    if detected:
        raw = collection.query(
            query_embeddings=[qvec],
            n_results=N_RESULTS,
            where={"schwerpunkt": {"$in": detected}},
            include=["metadatas", "distances"],
        )
        metas = (raw.get("metadatas") or [[]])[0]
        if len(metas) >= MIN_FILTER_HITS:
            return raw, "filtered"
    # Fallback: global, wenn kein schwerpunkt erkannt wird
    raw = collection.query(
        query_embeddings=[qvec],
        n_results=N_RESULTS,
        include=["metadatas", "distances"],
    )
    return raw, "global"


#--->
#Top-Treffer auswählen (dynamische Schwerlle + Deduplizierung nach Stufenpaar)
#--->
def select_hits(raw) -> Tuple[List[Tuple[float, str, int]], Optional[float]]:
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    #Treffer werden soriert niedrige distnaz = bessere Ähnlichkeit
    hits = sorted(((float(d), dict(m or {})) for d, m in zip(dists, metas)), key=lambda x: x[0])
    if not hits:
        return [], None

    #hier bestimmen wir eine dynamische schwelle 
    best_dist = hits[0][0]
    max_dist = best_dist * (1 + MAX_REL_INCREASE)

    #Auswahl + Dedupe (gleiches Stufenpaar pro Schwerpunkt nur einmal)
    selected, seen = [], set()
    for dist, meta in hits:
        if dist > max_dist:
            break
        sp = str(meta.get("schwerpunkt", "")).strip() or "Unbekannt"
        try:
            stufe = int(meta.get("stufe", 0))
        except Exception:
            stufe = 0
        key = (sp, partner_logik(stufe) if stufe > 0 else ("?", "?"))
        if key in seen:
            continue
        seen.add(key)
        selected.append((dist, sp, stufe))
        if len(selected) >= TOP_K:
            break
    return selected, best_dist



# -----------------------------
# Einfache UI
# -----------------------------
st.set_page_config(page_title="Chatbot (Prototyp-Test)")
st.title("Chatbot (Prototyp-Test)")

model = load_model()
collection = load_collection()
u_data = load_unterstuetzung()

st.caption(
    f"Collection: {collection.name} | count: {collection.count()} | "
    f"N_RESULTS={N_RESULTS}  TOP_K={TOP_K}  MAX_REL_INCREASE={int(MAX_REL_INCREASE*100)}%"
)

text = st.text_area("Fallschilderung", height=180)

if st.button("Suchen") and text.strip():
    detected = detect_schwerpunkte(text)
    #Embedding für die Anfrage
    qvec = model.encode(text).tolist()

    #Ähnlichkeitssuche
    raw, mode = query(collection, qvec, detected)
    best, best_dist = select_hits(raw)

    if best_dist is not None:
        dist_topk = ", ".join(f"{d:.3f}" for d, _, _ in best) if best else "-"
        st.caption(
            f"Modus: {mode}"
            + (f"| Keywords: {', '.join(detected)}" if detected else " keine Keywords erkannt")
            + f" | dist Top-{len(best)}: {dist_topk}"
        )

    if not best:
        st.warning("Keine Treffer gefunden.")
        st.stop()

    for i, (dist, sp, stufe) in enumerate(best, 1):
        st.markdown(f"### Treffer {i}")
        st.write(f"**Schwerpunkt:** {sp}")
        st.write(f"**dist:** {dist:.3f}")
        st.write("**Unterstützungspotenzial:**")
        for u in get_unterstuetzung(u_data, sp, stufe):
            st.write("- " + u)