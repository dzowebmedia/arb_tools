import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# ------------------------------
# Basic config
# ------------------------------

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = datetime.now(timezone.utc).year
VARIANTS_PER_BLOCK = 4  # exactly 4 variants per block

LANG_CONFIG = {
    "English": {
        "code": "en",
        "region": "US",
        "hint": "Use natural US English ad copy."
    },
    "French": {
        "code": "fr",
        "region": "FR",
        "hint": "Use standard French for France with correct capitalization."
    },
    "German": {
        "code": "de",
        "region": "DE",
        "hint": "Use German for Germany with correct capitalization of nouns."
    },
    "Italian": {
        "code": "it",
        "region": "IT",
        "hint": "Use standard Italian for Italy."
    },
    "Dutch": {
        "code": "nl",
        "region": "NL",
        "hint": "Use standard Dutch for the Netherlands."
    },
    "Portuguese (Brazil)": {
        "code": "pt",
        "region": "BR",
        "hint": "Use Brazilian Portuguese with correct capitalization."
    },
    "Spanish (US)": {
        "code": "es",
        "region": "US",
        "hint": "Use Spanish suitable for US Hispanic audience."
    },
    "Spanish (Spain)": {
        "code": "es",
        "region": "ES",
        "hint": "Use European Spanish for Spain with correct grammar."
    },
}

# ------------------------------
# Helper functions
# ------------------------------

def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:48]

def _title_case(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _pluralize(kw: str) -> str:
    kw = kw.strip()
    if not kw:
        return kw
    if kw.lower().endswith("s"):
        return kw
    return kw + "s"

def df_to_autofit_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1", index: bool = False) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        ws = writer.sheets[sheet_name]
        for i, col in enumerate(df.columns):
            series_as_str = df[col].astype(str)
            maxlen = max([len(str(col))] + [len(s) for s in series_as_str])
            ws.set_column(i, i, min(maxlen + 2, 60))
    return out.getvalue()

def two_tables_to_excel_bytes(
    upper_df: pd.DataFrame,
    lower_df: pd.DataFrame,
    upper_title: str = "Keyword Blocks",
    lower_title: str = "Ad Creatives",
    sheet_name: str = "Campaign"
) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        bold = wb.add_format({"bold": True})
        ws = wb.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = ws

        ws.write(0, 0, upper_title, bold)
        upper_df.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, index=False)

        for i, col in enumerate(upper_df.columns):
            maxlen = max(len(str(col)), *(len(str(x)) for x in upper_df[col].astype(str).values))
            ws.set_column(i, i, min(maxlen + 2, 60))

        start_row = 1 + len(upper_df) + 2
        ws.write(start_row, 0, lower_title, bold)

        lower_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1, startcol=0, index=False)

        for j, col in enumerate(lower_df.columns):
            maxlen = max(len(str(col)), *(len(str(x)) for x in lower_df[col].astype(str).values))
            ws.set_column(j, j, min(maxlen + 2, 60))

    return out.getvalue()

# ------------------------------
# OpenAI translation helpers
# ------------------------------

def translate_text_list(texts, target_language_label: str, is_headline: bool) -> list[str]:
    if target_language_label == "English":
        return texts

    cfg = LANG_CONFIG.get(target_language_label)
    if not cfg:
        return texts

    max_chars = 30 if is_headline else 90

    items = [str(t or "").strip() for t in texts]
    if not any(items):
        return texts

    numbered = [f"{idx+1}|||{t}" for idx, t in enumerate(items)]
    joined = "\n".join(numbered)

    system_msg = (
        "You are an expert ad copy translator. "
        "Translate each input line into the target language keeping the same meaning and intent. "
        "Respect these constraints: "
        "1 For headlines keep each output to at most 30 characters including spaces "
        "2 For descriptions and keyword phrases keep each output to at most 90 characters including spaces "
        "3 Return results as a JSON array of strings in the same order as the input lines meaning item 1 corresponds to line 1 etc."
        "You may freely change word order and sentence structure so the translation sounds natural in the target language."
        "If a good translation would slightly exceed the limit shorten naturally without breaking grammar."
    )

    user_msg = (
        f"Target language code {cfg['code']} region {cfg['region']}. "
        f"{cfg['hint']} "
        "Translate the following lines. Each line is in the format 'index|||text'. "
        "Ignore the index in the translation. "
        "Return a JSON array with one translated string per input in the same order. "
        "Input lines\n"
        f"{joined}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        translated = json.loads(raw)

        out = []
        for orig, tr in zip(items, translated):
            s = str(tr or "").strip()
            if not s:
                s = orig
            if len(s) > max_chars:
                s = s[:max_chars].rstrip()
            out.append(s)
        if len(out) < len(items):
            out.extend(items[len(out):])
        return out
    except Exception:
        return texts

def translate_keyword_blocks_df(df: pd.DataFrame, target_language_label: str) -> pd.DataFrame:
    if target_language_label == "English":
        return df

    df = df.copy()
    cols = ["block_title", "variant_term_1", "variant_term_2", "variant_term_3", "variant_term_4"]
    for col in cols:
        if col in df.columns:
            df[col] = translate_text_list(df[col].tolist(), target_language_label, is_headline=False)
    return df

def translate_ad_creatives_df(df: pd.DataFrame, target_language_label: str) -> pd.DataFrame:
    if target_language_label == "English":
        return df

    df = df.copy()

    head_cols = [f"headline_{i}" for i in range(1, 6)]
    for col in head_cols:
        if col in df.columns:
            df[col] = translate_text_list(df[col].tolist(), target_language_label, is_headline=True)

    desc_cols = [f"description_{i}" for i in range(1, 6)]
    for col in desc_cols:
        if col in df.columns:
            df[col] = translate_text_list(df[col].tolist(), target_language_label, is_headline=False)

    return df

# ------------------------------
# Keyword block logic
# ------------------------------

def compose_blocks(seed: str) -> list[str]:
    base = _title_case(seed)
    fams = [
        "{base}",
        "Best {base}",
        "{base} Near Me",
        "{base} Prices",
        "{year} {base}",
    ]
    blocks = []
    seen = set()
    for t in fams:
        b = t.format(base=base, year=CURRENT_YEAR).strip()
        if b not in seen:
            seen.add(b)
            blocks.append(b)
    return blocks

def generate_variants_for_block(block_title: str, seed: str) -> list[str]:
    title = _title_case(block_title)
    seed_title = _title_case(seed)
    tokens = seed_title.split()
    low = title.lower()

    def toggle_plural_word(w: str) -> str:
        if re.search(r"s$", w, re.I):
            return re.sub(r"s+$", "", w)
        return w + "s"

    def plural_main_noun(phrase: str) -> str:
        toks = phrase.split()
        if not toks:
            return phrase
        idx = len(toks) - 1
        if len(toks) >= 2 and toks[-2].lower() == "near" and toks[-1].lower() == "me":
            idx = len(toks) - 3
        elif len(toks) >= 1 and toks[-1].lower() == "prices":
            idx = len(toks) - 2
        if 0 <= idx < len(toks):
            toks[idx] = toggle_plural_word(toks[idx])
        return " ".join(toks)

    def brand_plural_seed(phrase: str) -> str:
        toks = phrase.split()
        if not toks:
            return phrase
        if toks[0].isdigit() and len(toks) >= 2:
            toks[1] = toggle_plural_word(toks[1])
        else:
            toks[0] = toggle_plural_word(toks[0])
        return " ".join(toks)

    def reorder_three(toks: list[str]) -> tuple[str, str]:
        if len(toks) >= 3:
            a, b, c = toks[:3]
            r1 = f"{c} {a} {b}"
            r2 = f"{b} {c} {a}"
            return r1, r2
        if len(toks) == 2:
            a, b = toks
            r1 = f"{b} {a}"
            r2 = f"{toggle_plural_word(a)} {b}"
            return r1, r2
        if len(toks) == 1:
            return toks[0], toggle_plural_word(toks[0])
        return "", ""

    variants: list[str] = []

    if low.startswith("best "):
        core = seed_title
        r1, r2 = reorder_three(tokens)
        v1 = f"Best {core}"
        v2 = f"Best {r1}"
        v3 = f"Best {r2}"
        v4 = f"Best {plural_main_noun(core)}"
        variants = [v1, v2, v3, v4]
    elif "near me" in low:
        core_seed = seed_title + " Near Me"
        v1 = core_seed
        v2 = plural_main_noun(core_seed)
        v3 = brand_plural_seed(core_seed)
        v4 = f"Near Me {seed_title}"
        variants = [v1, v2, v3, v4]
    elif low.endswith(" prices"):
        seed_prices = seed_title + " Prices"
        v1 = seed_prices
        v2 = plural_main_noun(seed_prices)
        v3 = "Prices " + brand_plural_seed(seed_title)
        v4 = seed_title + " Cost"
        variants = [v1, v2, v3, v4]
    elif title.split()[0].isdigit():
        year_tok = title.split()[0]
        core_seed = f"{year_tok} {seed_title}"
        v1 = core_seed
        v2 = f"{year_tok} {plural_main_noun(seed_title)}"
        v3 = f"{brand_plural_seed(seed_title)} {year_tok}"
        v4 = f"{seed_title} {year_tok}"
        variants = [v1, v2, v3, v4]
    else:
        core = seed_title
        r1, r2 = reorder_three(tokens)
        v1 = core
        v2 = plural_main_noun(core)
        v3 = r1
        v4 = r2
        variants = [v1, v2, v3, v4]

    clean: list[str] = []
    seen = set()
    for v in variants:
        v_clean = v.strip()
        if not v_clean:
            continue
        key = v_clean.lower()
        if key not in seen:
            seen.add(key)
            clean.append(v_clean)

    clean = clean[:VARIANTS_PER_BLOCK]
    if len(clean) < VARIANTS_PER_BLOCK:
        clean += [""] * (VARIANTS_PER_BLOCK - len(clean))
    return clean

# ------------------------------
# Headline and description logic
# ------------------------------

def build_headlines(seed: str) -> list[str]:
    base = _title_case(seed)
    plural = _pluralize(base)
    return [
        f"{base} Sales",
        f"{base} Clearance",
        f"Best {plural}",
        f"Amazing {base} Deals",
        f"New {base} Discounted",
    ]

def build_descriptions(seed: str) -> list[str]:
    base = seed.strip()
    if not base:
        base = "these options"
    plural = _pluralize(base)

    return [
        f"Check out the amazing {base} offers happening now in these searches.",
        f"Check the top searches for {base} discounts deals and options. Shop smart and save.",
        f"The hottest prices on {base} are here. Check top searches for best deals around.",
        f"Run a fast search for {base} and get the best deals available right now.",
        f"Check live pricing for {plural} in these searches. Grab big savings offers now.",
    ]

# ------------------------------
# Campaign builder
# ------------------------------

def generate_campaign(seed: str, language_label: str) -> dict:
    seed_clean = seed.strip()
    seed_title = _title_case(seed_clean)
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    campaign_name = f"{_slugify(seed_clean)}-{today_str}"

    blocks = compose_blocks(seed_clean)
    rows = []
    for b in blocks:
        variants = generate_variants_for_block(b, seed_clean)

        rows.append({
            "campaign_name": campaign_name,
            "seed_keyword": seed_title,
            "block_title": b,
            "variant_term_1": variants[0] if len(variants) > 0 else "",
            "variant_term_2": variants[1] if len(variants) > 1 else "",
            "variant_term_3": variants[2] if len(variants) > 2 else "",
            "variant_term_4": variants[3] if len(variants) > 3 else "",
        })

    keyword_blocks_df = pd.DataFrame(rows)

    heads = build_headlines(seed_clean)
    descs = build_descriptions(seed_clean)

    ad_creatives_df = pd.DataFrame([{
        "campaign_name": campaign_name,
        "seed_keyword": seed_title,
        "headline_1": heads[0],
        "headline_2": heads[1],
        "headline_3": heads[2],
        "headline_4": heads[3],
        "headline_5": heads[4],
        "description_1": descs[0],
        "description_2": descs[1],
        "description_3": descs[2],
        "description_4": descs[3],
        "description_5": descs[4],
    }])

    keyword_blocks_df = translate_keyword_blocks_df(keyword_blocks_df, language_label)
    ad_creatives_df = translate_ad_creatives_df(ad_creatives_df, language_label)

    return {
        "campaign_name": campaign_name,
        "keyword_blocks_df": keyword_blocks_df,
        "ad_creatives_df": ad_creatives_df,
    }

# ------------------------------
# Copy blocks
# ------------------------------

def build_headline_block(ad_creatives_df: pd.DataFrame) -> str:
    row = ad_creatives_df.iloc[0]
    heads = [row[f"headline_{i}"] for i in range(1, 6)]
    lines = []
    for h in heads:
        h = str(h or "").strip()
        if h:
            lines.append(h)
    return "\n".join(lines)

def build_description_block(ad_creatives_df: pd.DataFrame) -> str:
    row = ad_creatives_df.iloc[0]
    descs = [row[f"description_{i}"] for i in range(1, 6)]
    lines = []
    for d in descs:
        d = str(d or "").strip()
        if d:
            lines.append(d)
    return "\n".join(lines)

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Three Click Builder", layout="wide")
st.title("Three Click Builder")

st.subheader("1 Enter Seed Keyword")

language_label = st.selectbox(
    "Output Language",
    list(LANG_CONFIG.keys()),
    index=0,
)

seed = st.text_input("Seed Keyword Phrase", value="crossover suvs")

if st.button("Generate Campaign"):
    if not seed.strip():
        st.error("Please enter a keyword.")
    else:
        result = generate_campaign(seed, language_label)
        cname = result["campaign_name"]

        st.success(f"Campaign Created {cname} in {language_label}")

        st.markdown("**Keyword Blocks**")
        keyword_blocks_df = result["keyword_blocks_df"]
        st.dataframe(keyword_blocks_df, width="stretch")

        st.markdown("**Copy Keyword Blocks Per Block**")
        for idx, row in keyword_blocks_df.iterrows():
            block_title = str(row.get("block_title", "")).strip()
            if not block_title:
                continue

            variants = []
            for i in range(1, VARIANTS_PER_BLOCK + 1):
                col_term = f"variant_term_{i}"
                if col_term in row and str(row[col_term]).strip():
                    variants.append(str(row[col_term]).strip())

            block_text = "\n".join(variants)

            st.text_area(
                label=f"Block {idx+1} {block_title}",
                value=block_text,
                height=140,
            )

        st.markdown("**Ad Headlines And Descriptions**")
        st.dataframe(result["ad_creatives_df"], width="stretch")

        headline_block = build_headline_block(result["ad_creatives_df"])
        description_block = build_description_block(result["ad_creatives_df"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Copy Headlines**")
            st.text_area(
                "One headline per line",
                value=headline_block,
                height=300,
            )
        with c2:
            st.markdown("**Copy Descriptions**")
            st.text_area(
                "One description per line",
                value=description_block,
                height=300,
            )

        xbytes = two_tables_to_excel_bytes(
            upper_df=result["keyword_blocks_df"],
            lower_df=result["ad_creatives_df"],
            upper_title="Keyword Blocks",
            lower_title="Ad Creatives",
            sheet_name="Campaign"
        )

        file_path = OUTPUTS_DIR / f"campaign_{cname}.xlsx"
        with open(file_path, "wb") as f:
            f.write(xbytes)

        st.download_button(
            label=f"Download campaign_{cname}.xlsx",
            data=xbytes,
            file_name=f"campaign_{cname}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption(f"Saved to {file_path}")

st.markdown("---")
st.caption("This version supports multi language output using OpenAI translation while keeping your three click structure intact.")