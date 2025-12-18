import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple

def preprocessing():
    """
    Lit les fichiers TSV bruts, nettoie et structure les données,
    puis sauvegarde deux DataFrames picklés : full_title_df et full_body_df.
    """

    # -----------------------
    # Load raw data
    # -----------------------
    # title_df = pd.read_table("../data/soc-redditHyperlinks-title.tsv")
    # body_df = pd.read_table("../data/soc-redditHyperlinks-body.tsv")
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / ".." / ".." / "data").resolve()
    print("Found tsv files in:", data_dir)
    raw_title = data_dir / "soc-redditHyperlinks-title.tsv"
    raw_body  = data_dir / "soc-redditHyperlinks-body.tsv"
    title_df = pd.read_table(raw_title)
    body_df = pd.read_table(raw_body)
    
    # -----------------------
    # Property names
    # -----------------------
    POST_PROPERTIES = {
        1: "Number of characters",
        2: "Number of characters without counting white space",
        3: "Fraction of alphabetical characters",
        4: "Fraction of digits",
        5: "Fraction of uppercase characters",
        6: "Fraction of white spaces",
        7: "Fraction of special characters, such as comma, exclamation mark, etc.",
        8: "Number of words",
        9: "Number of unique words",
        10: "Number of long words (at least 6 characters)",
        11: "Average word length",
        12: "Number of unique stopwords",
        13: "Fraction of stopwords",
        14: "Number of sentences",
        15: "Number of long sentences (at least 10 words)",
        16: "Average number of characters per sentence",
        17: "Average number of words per sentence",
        18: "Automated readability index",
        19: "Positive sentiment calculated by VADER",
        20: "Negative sentiment calculated by VADER",
        21: "Compound sentiment calculated by VADER",
        22: "LIWC_Funct",
        23: "LIWC_Pronoun",
        24: "LIWC_Ppron",
        25: "LIWC_I",
        26: "LIWC_We",
        27: "LIWC_You",
        28: "LIWC_SheHe",
        29: "LIWC_They",
        30: "LIWC_Ipron",
        31: "LIWC_Article",
        32: "LIWC_Verbs",
        33: "LIWC_AuxVb",
        34: "LIWC_Past",
        35: "LIWC_Present",
        36: "LIWC_Future",
        37: "LIWC_Adverbs",
        38: "LIWC_Prep",
        39: "LIWC_Conj",
        40: "LIWC_Negate",
        41: "LIWC_Quant",
        42: "LIWC_Numbers",
        43: "LIWC_Swear",
        44: "LIWC_Social",
        45: "LIWC_Family",
        46: "LIWC_Friends",
        47: "LIWC_Humans",
        48: "LIWC_Affect",
        49: "LIWC_Posemo",
        50: "LIWC_Negemo",
        51: "LIWC_Anx",
        52: "LIWC_Anger",
        53: "LIWC_Sad",
        54: "LIWC_CogMech",
        55: "LIWC_Insight",
        56: "LIWC_Cause",
        57: "LIWC_Discrep",
        58: "LIWC_Tentat",
        59: "LIWC_Certain",
        60: "LIWC_Inhib",
        61: "LIWC_Incl",
        62: "LIWC_Excl",
        63: "LIWC_Percept",
        64: "LIWC_See",
        65: "LIWC_Hear",
        66: "LIWC_Feel",
        67: "LIWC_Bio",
        68: "LIWC_Body",
        69: "LIWC_Health",
        70: "LIWC_Sexual",
        71: "LIWC_Ingest",
        72: "LIWC_Relativ",
        73: "LIWC_Motion",
        74: "LIWC_Space",
        75: "LIWC_Time",
        76: "LIWC_Work",
        77: "LIWC_Achiev",
        78: "LIWC_Leisure",
        79: "LIWC_Home",
        80: "LIWC_Money",
        81: "LIWC_Relig",
        82: "LIWC_Death",
        83: "LIWC_Assent",
        84: "LIWC_Dissent",
        85: "LIWC_Nonflu",
        86: "LIWC_Filler"
    }

    # -----------------------
    # Process and save both datasets
    # -----------------------
    full_title_df = process_df(title_df, POST_PROPERTIES)
    full_body_df = process_df(body_df, POST_PROPERTIES)

    # full_title_df.to_pickle("../data/full_title_df.pkl")
    # full_body_df.to_pickle("../data/full_body_df.pkl")

    full_title_df.to_pickle(data_dir / "full_title_df.pkl")
    full_body_df.to_pickle(data_dir / "full_body_df.pkl")

    print("✅ DataFrames saved successfully as pickle files.")


def process_df(df, POST_PROPERTIES):
    # Build a base clean_df directly from the raw dataframe (no Python loops)
    clean_df = df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT", "PROPERTIES"]].copy()
    
    # Fast, vectorized expansion of comma-separated properties into columns
    prop_names = [POST_PROPERTIES[i] for i in sorted(POST_PROPERTIES.keys())]
    props_df = clean_df["PROPERTIES"].str.split(",", expand=True)
    props_df = props_df.iloc[:, :len(prop_names)].astype(float)
    props_df.columns = prop_names
    
    # Concatenate and drop the original string column
    clean_df = pd.concat([clean_df.drop(columns=["PROPERTIES"]), props_df], axis=1)
    
    # Minimal, coherent typing for key fields
    clean_df["TIMESTAMP"] = pd.to_datetime(clean_df["TIMESTAMP"], errors="coerce", utc=True)
    clean_df["LINK_SENTIMENT"] = (
        pd.to_numeric(clean_df["LINK_SENTIMENT"], errors="coerce")
          .round()
          .astype("Int64")
          .astype(int)
    )
    # 4) Explicitly enforce float dtype on all LIWC_* columns (redundant but safe)
    prop_cols = list(POST_PROPERTIES.values())
    liwc_cols = [c for c in prop_cols if c.startswith("LIWC_")]
    if liwc_cols:
        clean_df[liwc_cols] = clean_df[liwc_cols].astype(float)
    return clean_df
    


# ==== Embeddings CSV parsing and preprocessing ====

def _parse_embeddings_csv(
    file_path: Path,
    id_col_name: str,
    expected_dim: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse un CSV SNAP de la forme:
        <ID>,<v1>,<v2>,...,<vD>
    où la première virgule sépare l'ID du vecteur, et le vecteur contient D flottants.

    Paramètres
    ----------
    file_path : Path
        Chemin vers le fichier CSV.
    id_col_name : str
        Nom de la colonne ID ("USER_ID" ou "SUBREDDIT").
    expected_dim : int | None
        Si fourni, seules les lignes ayant exactement cette dimension sont conservées.
        Sinon, on conserve la dimension la plus fréquente observée.
    verbose : bool
        Affiche des infos de parsing.

    Retour
    ------
    pd.DataFrame
        Colonnes: [id_col_name, emb_0, ..., emb_{D-1}, emb_norm]
    """
    ids: List[str] = []
    vectors: List[List[float]] = []

    if verbose:
        print(f"Reading: {file_path}")

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            # Sauter un éventuel header
            if i == 0 and ("USER_ID" in line or "SUBREDDIT" in line or "VECTOR" in line):
                continue

            # Séparer ID et vecteur: split sur la première virgule uniquement
            try:
                id_part, vec_part = line.split(",", 1)
            except ValueError:
                # Ligne invalide (pas de virgule)
                continue

            id_part = id_part.strip()
            if not id_part:
                continue

            # Parser le reste en floats
            try:
                vec = [float(x) for x in vec_part.split(",") if x != ""]
                # Strictly enforce fixed dimension (default 300)
                if len(vec) != int(expected_dim):
                    raise ValueError(
                        f"Row with ID '{id_part}' has length {len(vec)} but expected {expected_dim}. "
                        "Ensure the SNAP embeddings file is intact and comma-separated."
                    )
            except ValueError:
                # Si une valeur n'est pas convertible → on skippe la ligne
                continue

            ids.append(id_part)
            vectors.append(vec)

    dim_to_keep = int(expected_dim)
    ids_kept = ids
    vecs_kept = vectors

    if len(vecs_kept) == 0:
        raise ValueError(f"No vectors parsed from   {file_path}. Check file encoding and delimiters.")

    emb_cols = [f"emb_{j}" for j in range(dim_to_keep)]
    arr = np.asarray(vecs_kept, dtype=np.float32)
    df = pd.DataFrame(arr, columns=emb_cols)
    df.insert(0, id_col_name, ids_kept)

    # Norme L2 pour sanity check / filtrage éventuel
    df["emb_norm"] = np.linalg.norm(arr, axis=1)

    if verbose:
        print(
            f"Parsed {len(df):,} rows | dim={dim_to_keep} | "
            f"norm mean={df['emb_norm'].mean():.3f} ± {df['emb_norm'].std():.3f}"
        )

    return df


def embeddings_preprocessing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit, parse et persiste les embeddings Reddit (users & subreddits).

    Entrées attendues (dans ../data depuis ce fichier):
        - web-redditEmbeddings-users.csv
        - web-redditEmbeddings-subreddits.csv

    Sorties créées:
        - user_embeddings_df.pkl
        - subreddit_embeddings_df.pkl

    Retourne
    --------
    (user_df, sub_df) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    # -----------------------
    # Load raw data (paths)
    # -----------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = Path(os.path.join(base_dir, "../../data")).resolve()

    user_csv = data_dir / "web-redditEmbeddings-users.csv"
    sub_csv  = data_dir / "web-redditEmbeddings-subreddits.csv"
    out_user = data_dir / "user_embeddings_df.pkl"
    out_sub  = data_dir / "subreddit_embeddings_df.pkl"

    if not user_csv.exists():
        raise FileNotFoundError(f"Missing file: {user_csv}")
    if not sub_csv.exists():
        raise FileNotFoundError(f"Missing file: {sub_csv}")

    # -----------------------
    # Process and save both datasets
    # -----------------------
    # 1) Parse users with fixed dimension 300
    user_df = _parse_embeddings_csv(user_csv, id_col_name="USER_ID", expected_dim=300, verbose=True)
    inferred_dim = 300

    # 2) Parse subreddits with the same fixed dimension
    sub_df = _parse_embeddings_csv(sub_csv, id_col_name="SUBREDDIT", expected_dim=300, verbose=True)

    # 4) Persistences en pickle (aligné avec load_data.py)
    user_df.to_pickle(out_user)
    sub_df.to_pickle(out_sub)

    print(f"✅ Saved {out_user} ({len(user_df):,} rows, dim={inferred_dim})")
    print(f"✅ Saved {out_sub} ({len(sub_df):,} rows, dim={inferred_dim})")

    return user_df, sub_df