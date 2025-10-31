
# app_rpc_only.py — Arb LaunchOps RPC-only — Daily RPC Scanner + Daily Plan
import io
import os
import json
from datetime import date,timedelta,datetime
import pandas as pd
import streamlit as st
import re
import yaml
import socket
import time
import requests

# =====================
# Paths & folders RPC-only
# =====================
BASE_DIR=os.path.dirname(__file__)
OUTPUTS=os.path.join(BASE_DIR,"outputs_rpc")
EXPANSIONS=os.path.join(BASE_DIR,"expansions_rpc")
os.makedirs(OUTPUTS,exist_ok=True)
os.makedirs(EXPANSIONS,exist_ok=True)

# Dedupe keys
DEDUPE_KEYS=("Keyword","Device","Geo")

BLOCKLIST_PATH=os.path.join(OUTPUTS,"custom_blocklist.txt")
HISTORY_PATH=os.path.join(OUTPUTS,"history_rpc.parquet")
LAUNCH_LOG_PATH=os.path.join(OUTPUTS,"launched_log_rpc.parquet")

# ---- session defaults
DEFAULTS={
    "ps_open":False,
    "ps_running_rpc":False,
    "dod_rpc":pd.DataFrame(),
    "wow_rpc":pd.DataFrame(),
    "sur_rpc":pd.DataFrame(),
    "today_roll_rpc":pd.DataFrame(),
    "aug_max_seeds":50,
    "aug_geo":"US",
    "aug_tf":"7d",
    "aug_top_per_seed":5,
    "aug_min_score":0.35,
    "aug_multi_only":False,
    "ps_manual_seeds_raw":"",
}

def _init_state():
    ss=st.session_state
    for k,v in DEFAULTS.items():
        if k not in ss:
            ss[k]=v

# =====================
# Storage helpers
# =====================
def load_launch_log()->pd.DataFrame:
    if os.path.exists(LAUNCH_LOG_PATH):
        try:
            return pd.read_parquet(LAUNCH_LOG_PATH)
        except Exception:
            csv_fallback=LAUNCH_LOG_PATH.replace(".parquet",".csv")
            if os.path.exists(csv_fallback):
                return pd.read_csv(csv_fallback,parse_dates=["launched_at"])
    return pd.DataFrame(columns=["Keyword","Device","Geo","launched_at","source"])

def save_launch_log(df:pd.DataFrame):
    try:
        df.to_parquet(LAUNCH_LOG_PATH,index=False)
    except Exception:
        df.to_csv(LAUNCH_LOG_PATH.replace(".parquet",".csv"),index=False)

def append_launch_log_rpc(plan_df:pd.DataFrame,source:str="rpc_daily_plan")->None:
    if plan_df is None or plan_df.empty:
        return
    needed=["Keyword","Device","Geo"]
    df=plan_df.copy()
    for c in needed:
        if c not in df.columns:
            df[c]=""
    df=df[needed]
    df["launched_at"]=pd.Timestamp.now()
    df["source"]=source
    log=load_launch_log()
    log=pd.concat([log,df],ignore_index=True)
    save_launch_log(log)

@st.cache_data(show_spinner=False)
def _load_history_cache(mtime:int)->pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_parquet(HISTORY_PATH)
        except Exception:
            csv_fallback=HISTORY_PATH.replace(".parquet",".csv")
            if os.path.exists(csv_fallback):
                return pd.read_csv(csv_fallback,parse_dates=["date"])
    return pd.DataFrame(columns=["date","Keyword","Device","Geo","RPC"])

def load_history()->pd.DataFrame:
    mtime=int(os.path.getmtime(HISTORY_PATH)) if os.path.exists(HISTORY_PATH) else 0
    return _load_history_cache(mtime)

def save_history(df_hist:pd.DataFrame):
    try:
        df_hist.to_parquet(HISTORY_PATH,index=False)
    except Exception:
        df_hist.to_csv(HISTORY_PATH.replace(".parquet",".csv"),index=False)

def save_csv(df:pd.DataFrame,name:str)->str:
    p=os.path.join(OUTPUTS,name)
    df.to_csv(p,index=False)
    return p

# =====================
# Blacklist + ranking
# =====================
def _read_yaml_list(path,key):
    try:
        with open(path,"r",encoding="utf-8") as f:
            data=yaml.safe_load(f) or {}
        items=data.get(key,[])
        return [str(x).strip() for x in items if str(x).strip()]
    except Exception:
        return []

DEFAULT_HEALTH=["diabetes","cancer","asthma","arthritis","depression","anxiety","autism","adhd","hiv","covid"]
DEFAULT_MEDS=["ozempic","wegovy","metformin","statin","lipitor","xanax","adderall","viagra"]
DEFAULT_BRANDS=["amazon","walmart","facebook","google","apple","microsoft","netflix","tesla","verizon","t-mobile","att","bankrate","cnet","the verge"]

def load_custom_blocklist()->list[str]:
    try:
        if os.path.exists(BLOCKLIST_PATH):
            with open(BLOCKLIST_PATH,"r",encoding="utf-8") as f:
                items=[ln.strip() for ln in f.read().splitlines() if ln.strip()]
            return sorted(set(items),key=str.lower)
    except Exception:
        pass
    return []

def save_custom_blocklist(items:list[str]):
    uniq=sorted(set([i.strip() for i in items if str(i).strip()]),key=str.lower)
    os.makedirs(OUTPUTS,exist_ok=True)
    with open(BLOCKLIST_PATH,"w",encoding="utf-8") as f:
        f.write("\n".join(uniq))

def add_block_terms(new_text:str)->list[str]:
    terms=[t.strip() for t in re.split(r"[,\n]",str(new_text or "")) if t.strip()]
    merged=sorted(set(load_custom_blocklist()+terms),key=str.lower)
    save_custom_blocklist(merged)
    return merged

def remove_block_terms(to_remove:list[str])->list[str]:
    cur=load_custom_blocklist()
    drop=set([t.strip().lower() for t in (to_remove or [])])
    kept=[t for t in cur if t.lower() not in drop]
    save_custom_blocklist(kept)
    return kept

def compile_blacklist():
    health=_read_yaml_list(os.path.join(BASE_DIR,"filters.yaml"),"health_terms") or DEFAULT_HEALTH
    meds=_read_yaml_list(os.path.join(BASE_DIR,"filters.yaml"),"medications") or DEFAULT_MEDS
    brands=_read_yaml_list(os.path.join(BASE_DIR,"filters.yaml"),"brands") or DEFAULT_BRANDS
    try:
        custom=load_custom_blocklist()
    except Exception:
        custom=[]
    words=sorted({w.strip().lower() for w in (health+meds+brands+custom)}- {""})
    if not words:
        return None
    return re.compile(r"(?i)\b("+ "|".join(re.escape(w) for w in words)+ r")\b")

def apply_blacklist(df:pd.DataFrame,patt,text_cols=("Keyword",))->pd.DataFrame:
    if df is None or df.empty or patt is None:
        return df
    mask=pd.Series(False,index=df.index)
    for col in text_cols:
        if col in df.columns:
            mask=mask | df[col].astype(str).str.contains(patt,na=False)
    return df[~mask]

def rank_and_pick_top_rpc(dod:pd.DataFrame,wow:pd.DataFrame,sur:pd.DataFrame,top_n:int=25)->pd.DataFrame:
    parts=[]
    if isinstance(dod,pd.DataFrame) and not dod.empty:
        d=dod.copy(); d["segment"]="DoD"; d["lift_metric"]=d.get("rpc_lift_pct",0.0); parts.append(d)
    if isinstance(wow,pd.DataFrame) and not wow.empty:
        w=wow.copy(); w["segment"]="WoW"; w["lift_metric"]=w.get("rpc_lift_pct",0.0); parts.append(w)
    if isinstance(sur,pd.DataFrame) and not sur.empty:
        s=sur.copy(); s["segment"]="Surge"; s["lift_metric"]=s.get("lift_vs_base_pct",0.0); parts.append(s)
    if not parts:
        return pd.DataFrame()
    allw=pd.concat(parts,ignore_index=True)
    allw.sort_values(["lift_metric","RPC"],ascending=[False,False],inplace=True)
    return allw.head(int(top_n))

def pick_topN_avoiding_recent_rpc(plan_pool:pd.DataFrame,N:int,prevent_relaunch:bool,prevent_days:int)->pd.DataFrame:
    if not isinstance(plan_pool,pd.DataFrame) or plan_pool.empty:
        return pd.DataFrame(columns=(plan_pool.columns if isinstance(plan_pool,pd.DataFrame) else []))
    if not prevent_relaunch:
        return plan_pool.head(int(N)).copy()
    launch_log=load_launch_log()
    if launch_log is None or launch_log.empty:
        return plan_pool.head(int(N)).copy()
    cutoff=pd.Timestamp.today().normalize()-pd.Timedelta(days=int(prevent_days))
    if "launched_at" in launch_log.columns:
        launch_log["launched_at"]=pd.to_datetime(launch_log["launched_at"],errors="coerce")
        recent=launch_log[launch_log["launched_at"]>=cutoff]
    else:
        recent=launch_log
    rec=recent.copy(); pp=plan_pool.copy()
    for c in DEDUPE_KEYS:
        if c not in rec.columns: rec[c]=""
        if c not in pp.columns: pp[c]=""
    keys_list=list(DEDUPE_KEYS)
    if rec.empty:
        return pp.head(int(N)).copy()
    recent_keys=set(tuple(str(row[c]) for c in keys_list) for _,row in rec[keys_list].iterrows())
    mask=pp.apply(lambda row: tuple(str(row[c]) for c in keys_list) in recent_keys,axis=1)
    return pp[~mask].head(int(N)).copy()

# =====================
# Ingest & normalization
# =====================
@st.cache_data(show_spinner=False)
def load_csv(file_bytes:bytes)->pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.read_excel(io.BytesIO(file_bytes))

def _find_col_by_prefix(header_map:dict,*prefixes:str)->str|None:
    for pref in prefixes:
        for low,orig in header_map.items():
            if low.startswith(pref):
                return orig
    return None

def _coerce_rpc(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if "RPC" in out.columns:
        out["RPC"]=out["RPC"].astype(str).str.replace(r"[^0-9.\-]","",regex=True)
        out["RPC"]=pd.to_numeric(out["RPC"],errors="coerce").fillna(0.0)
    return out

def ensure_core_columns_rpc(df:pd.DataFrame,default_geo:str,device_filter:str|None=None):
    df=df.copy()
    header_map={c.lower().strip():c for c in df.columns}
    kw_col=header_map.get("keyword") or header_map.get("kw") or header_map.get("relatedkw") or header_map.get("realtedkw") or _find_col_by_prefix(header_map,"keyword","kw","relatedkw","realtedkw")
    dev_col=header_map.get("device") or header_map.get("name") or header_map.get("platform") or _find_col_by_prefix(header_map,"device","name","platform")
    rpc_col=header_map.get("y rpc") or header_map.get("rpc") or _find_col_by_prefix(header_map,"y rpc","rpc")
    geo_col=header_map.get("geo") or header_map.get("country") or header_map.get("region") or _find_col_by_prefix(header_map,"geo","country","region")
    if kw_col and kw_col!="Keyword": df.rename(columns={kw_col:"Keyword"},inplace=True)
    if dev_col and dev_col!="Device": df.rename(columns={dev_col:"Device"},inplace=True)
    if rpc_col and rpc_col!="RPC": df.rename(columns={rpc_col:"RPC"},inplace=True)
    if geo_col and geo_col!="Geo": df.rename(columns={geo_col:"Geo"},inplace=True)
    if device_filter and "Device" in df.columns:
        df=df[df["Device"].astype(str).str.lower()==device_filter.strip().lower()]
    if "Keyword" not in df.columns: df["Keyword"]=""
    if "Device" not in df.columns: df["Device"]=""
    if "RPC" not in df.columns: df["RPC"]=0.0
    if "Geo" not in df.columns: df["Geo"]=default_geo
    df=_coerce_rpc(df)
    return df,{"rpc":"RPC","geo":"Geo","device":"Device","kw":"Keyword"}

def rollup_today(df:pd.DataFrame,cols:dict,day:date)->pd.DataFrame:
    day_str=day.strftime("%Y-%m-%d")
    df=_coerce_rpc(df)
    agg=df.groupby(["Keyword","Device",cols["geo"]],as_index=False).agg({cols["rpc"]:"mean"})
    agg.rename(columns={cols["geo"]:"Geo",cols["rpc"]:"RPC"},inplace=True)
    agg.insert(0,"date",day_str)
    return agg

def _add_norm_cols(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    for c in ("Keyword","Niche","Device","Geo"):
        if c in out.columns:
            out[c]=out[c].astype(str)
    if "Keyword" in out.columns: out["k_norm"]=out["Keyword"].str.strip().str.casefold()
    if "Niche" in out.columns: out["n_norm"]=out["Niche"].str.strip().str.casefold()
    if "Device" in out.columns: out["d_norm"]=out["Device"].str.strip().str.casefold()
    if "Geo" in out.columns: out["g_norm"]=out["Geo"].str.strip().str.upper()
    return out

def _norm_join_keys(df:pd.DataFrame)->list[str]:
    keys=[]
    if "k_norm" in df.columns: keys.append("k_norm")
    if "n_norm" in df.columns: keys.append("n_norm")
    elif "d_norm" in df.columns: keys.append("d_norm")
    if "g_norm" in df.columns: keys.append("g_norm")
    return keys

def winners_DoD(hist:pd.DataFrame)->pd.DataFrame:
    if hist.empty: return pd.DataFrame()
    df=hist.copy(); df["date"]=pd.to_datetime(df["date"],errors="coerce")
    dates=sorted([d for d in df["date"].unique() if pd.notna(d)])
    if len(dates)<2: return pd.DataFrame()
    today,yest=dates[-1],dates[-2]
    join_cols=[c for c in ("Keyword","Niche","Device","Geo") if c in df.columns]
    base_cols=join_cols+["RPC"]
    d0=_add_norm_cols(df[df["date"]==today][base_cols])
    d1=_add_norm_cols(df[df["date"]==yest][base_cols])
    keys=_norm_join_keys(d0)
    prior=d1[keys+["RPC"]].rename(columns={"RPC":"RPC_prior"})
    out=d0.merge(prior,on=keys,how="left")
    out["RPC_prior"]=out["RPC_prior"].fillna(out["RPC"])
    out["rpc_lift_pct"]=(out["RPC"]-out["RPC_prior"])/out["RPC_prior"].replace(0,1)
    out=out.sort_values(["rpc_lift_pct","RPC"],ascending=[False,False])
    drop_norm=[c for c in ("k_norm","n_norm","d_norm","g_norm") if c in out.columns]
    return out.drop(columns=drop_norm,errors="ignore")

def winners_WoW(hist:pd.DataFrame)->pd.DataFrame:
    if hist.empty: return pd.DataFrame()
    df=hist.copy(); df["date"]=pd.to_datetime(df["date"],errors="coerce")
    last_day=df["date"].max()
    if pd.isna(last_day): return pd.DataFrame()
    this_week_start=last_day-timedelta(days=6)
    prev_week_end=this_week_start-timedelta(days=1)
    prev_week_start=prev_week_end-timedelta(days=6)
    join_cols=[c for c in ("Keyword","Niche","Device","Geo") if c in df.columns]
    wk=df[(df["date"]>=this_week_start) & (df["date"]<=last_day)]
    pw=df[(df["date"]>=prev_week_start) & (df["date"]<=prev_week_end)]
    w0=wk.groupby(join_cols,as_index=False).agg(RPC=("RPC","mean"))
    w1=pw.groupby(join_cols,as_index=False).agg(RPC_prior=("RPC","mean"))
    w0=_add_norm_cols(w0); w1=_add_norm_cols(w1)
    keys=_norm_join_keys(w0)
    out=w0.merge(w1[keys+["RPC_prior"]],on=keys,how="left")
    out["RPC_prior"]=out["RPC_prior"].fillna(out["RPC"])
    out["rpc_lift_pct"]=(out["RPC"]-out["RPC_prior"])/out["RPC_prior"].replace(0,1)
    out=out.sort_values(["rpc_lift_pct","RPC"],ascending=[False,False])
    drop_norm=[c for c in ("k_norm","n_norm","d_norm","g_norm") if c in out.columns]
    return out.drop(columns=drop_norm,errors="ignore")

def surges_low_to_high(hist:pd.DataFrame,baseline_weeks:int,surge_threshold:float):
    if hist.empty: return pd.DataFrame()
    df=hist.copy(); df["date"]=pd.to_datetime(df["date"])
    last_day=df["date"].max()
    baseline_start=last_day-timedelta(days=7*baseline_weeks)
    base=df[(df["date"]>=baseline_start) & (df["date"]<last_day)]
    today=df[df["date"]==last_day]
    if base.empty or today.empty: return pd.DataFrame()
    b=base.groupby(["Keyword","Device","Geo"],as_index=False).agg(base_rpc=("RPC","mean"))
    t=today[["Keyword","Device","Geo","RPC"]]
    out=t.merge(b,on=["Keyword","Device","Geo"],how="left").fillna({"base_rpc":t["RPC"].median() if len(t) else 0.0})
    if len(out)>=4:
        q1=out["base_rpc"].quantile(0.25)
        out=out[out["base_rpc"]<=q1]
    out["lift_vs_base_pct"]=(out["RPC"]-out["base_rpc"])/out["base_rpc"].replace(0,1)
    out=out[out["lift_vs_base_pct"]>=surge_threshold]
    out.sort_values(["lift_vs_base_pct","RPC"],ascending=[False,False],inplace=True)
    return out

def write_payload(dod:pd.DataFrame,wow:pd.DataFrame,sur:pd.DataFrame):
    top_dod=dod.head(10)[["Keyword","Device","Geo","RPC","rpc_lift_pct"]] if not dod.empty else pd.DataFrame()
    top_wow=wow.head(10)[["Keyword","Device","Geo","RPC","rpc_lift_pct"]] if not wow.empty else pd.DataFrame()
    top_sur=sur.head(10)[["Keyword","Device","Geo","RPC","lift_vs_base_pct"]] if not sur.empty else pd.DataFrame()
    lines=[
        "Arb LaunchOps — RPC-only Daily RPC Summary",
        "\nDoD Winners Top 10:\n"+(top_dod.to_csv(index=False) if not top_dod.empty else "(none)"),
        "\nWoW Winners Top 10:\n"+(top_wow.to_csv(index=False) if not top_wow.empty else "(none)"),
        "\nSurges Top 10:\n"+(top_sur.to_csv(index=False) if not top_sur.empty else "(none)"),
    ]
    txt="\n".join(lines)
    with open(os.path.join(OUTPUTS,"chat_payload_rpc.txt"),"w",encoding="utf-8") as f:
        f.write(txt)
    plan={
        "generated_at":datetime.now().isoformat(timespec="seconds"),
        "dod":top_dod.to_dict(orient="records"),
        "wow":top_wow.to_dict(orient="records"),
        "surges":top_sur.to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUTS,"plan_input_rpc.json"),"w",encoding="utf-8") as f:
        json.dump(plan,f,indent=2)

# =====================
# UI — moved under main
# =====================
def _detect_keyword_col_preview(df):
    cand={c.lower().strip():c for c in df.columns}
    for k in ("keyword","kw","relatedkw","realtedkw"):
        if k in cand:
            return cand[k]
    return "Keyword" if "Keyword" in df.columns else None

def _collect_seeds_preview(limit:int):
    seeds,source=[], ""
    for key in ("dod","wow","sur","dod_rpc","wow_rpc","sur_rpc"):
        df=st.session_state.get(key)
        if isinstance(df,pd.DataFrame) and not df.empty:
            kw_col=_detect_keyword_col_preview(df)
            if kw_col:
                seeds.extend(df[kw_col].astype(str).tolist())
                source=source or f"session:{key}"
    if not seeds:
        for key in ("today_roll","today_roll_rpc"):
            tr=st.session_state.get(key)
            if isinstance(tr,pd.DataFrame) and not tr.empty:
                kw_col=_detect_keyword_col_preview(tr)
                if kw_col:
                    if "Clicks" in tr.columns:
                        seeds.extend(tr.sort_values("Clicks",ascending=False)[kw_col].astype(str).tolist())
                    else:
                        seeds.extend(tr[kw_col].astype(str).tolist())
                    source=f"session:{key}"
                    break
    if not seeds:
        for fname in ("rollup_daily.csv","rollup_daily_rpc.csv"):
            path=os.path.join(OUTPUTS,fname)
            if os.path.exists(path):
                df=pd.read_csv(path)
                kw_col=_detect_keyword_col_preview(df)
                if kw_col:
                    if "Clicks" in df.columns:
                        seeds.extend(df.sort_values("Clicks",ascending=False)[kw_col].astype(str).tolist())
                    else:
                        seeds.extend(df[kw_col].astype(str).tolist())
                    source=f"file:{fname}"
                    break
    if not seeds:
        hist=load_history()
        if isinstance(hist,pd.DataFrame) and not hist.empty:
            hist["date"]=pd.to_datetime(hist["date"],errors="coerce")
            latest=hist["date"].max()
            if pd.notna(latest):
                day=hist[hist["date"]==latest]
                kw_col=_detect_keyword_col_preview(day)
                if kw_col:
                    if "Clicks" in day.columns:
                        seeds.extend(day.sort_values("Clicks",ascending=False)[kw_col].astype(str).tolist())
                    else:
                        seeds.extend(day[kw_col].astype(str).tolist())
                    source="file:history(latest)"
    seen,out=set(),[]
    for s in seeds:
        s=str(s).strip(); key=s.casefold()
        if s and key not in seen:
            seen.add(key); out.append(s)
    patt_local=compile_blacklist()
    if patt_local:
        out=[s for s in out if not patt_local.search(s)]
    return out[: int(st.session_state.get("aug_max_seeds",50))],source

def main():
    st.set_page_config(page_title="Arb LaunchOps — RPC-only Daily Scanner",layout="wide")
    _init_state()

    st.title("Arb LaunchOps — RPC-only Daily Scanner")

    with st.expander("Upload & Run",expanded=True):
        file=st.file_uploader("Upload Daily CSV/XLSX Keyword + Device + RPC",type=["csv","xlsx"],accept_multiple_files=False)
        c1,c2,c3=st.columns(3)
        with c1:
            default_geo=st.text_input("Default GEO","US")
            device_filter=st.text_input("Filter by Device equals optional","")
            baseline_weeks=st.number_input("Surge Baseline weeks",1,52,8,1)
        with c2:
            rpc_lift_gate=st.number_input("Lift Gate rpc_lift_pct",0.0,5.0,0.25,0.01,format="%.2f")
            surge_gate=st.number_input("Surge Gate lift_vs_base",0.0,5.0,0.50,0.01,format="%.2f")
        with c3:
            prevent_relaunch=st.checkbox("Prevent relaunch if seen in last N days",True)
            prevent_days=st.number_input("N days window dedupe",1,90,30,1)
            build_plan=st.checkbox("Build Daily Plan outputs",True)
            TOP_N_PLAN=st.number_input("Daily plan size",5,200,25,5)

    with st.expander("Custom Blocklist persisted",expanded=False):
        st.caption("Add keywords or phrases to exclude from RPC Daily Plan. Stored in outputs/custom_blocklist.txt")
        cur=load_custom_blocklist()
        st.write({"count":len(cur)})
        if cur:
            st.dataframe(pd.DataFrame({"blocked":cur}),use_container_width=True,height=200)
        add_text=st.text_area("Add terms comma or newline separated","",height=80)
        cba,cbb,cbc=st.columns([1,1,1])
        with cba:
            if st.button("Add terms",key="rpc_bl_add"):
                new_list=add_block_terms(add_text)
                st.success(f"Added. Total blocked: {len(new_list)}")
        rem_sel=st.multiselect("Remove selected blocked terms",cur,key="rpc_bl_remove_sel")
        with cbb:
            if st.button("Remove selected",key="rpc_bl_remove_btn"):
                new_list=remove_block_terms(rem_sel)
                st.success(f"Removed. Total blocked: {len(new_list)}")
        with cbc:
            if st.button("Clear all",key="rpc_bl_clear_all"):
                save_custom_blocklist([])
                st.success("Custom blocklist cleared.")

    run=st.button("Run RPC Scan",key="run_rpc_scan_btn")

    if run and file is not None:
        raw=load_csv(file.read())
        raw,cols=ensure_core_columns_rpc(raw,default_geo,device_filter if device_filter else None)
        today=date.today()
        hist=load_history()
        today_roll=rollup_today(raw,cols,today)
        if today_roll is None or today_roll.empty or "date" not in today_roll.columns:
            st.error("No rows in todays rollup. Check upload or device filter.")
            st.stop()
        if not hist.empty and "date" in hist.columns:
            day_key=today.strftime("%Y-%m-%d")
            hist=hist[hist["date"].astype(str)!=day_key]
            hist=pd.concat([hist,today_roll],ignore_index=True)
        else:
            hist=today_roll.copy()
        save_history(hist)
        save_csv(today_roll,"rollup_daily_rpc.csv")

        dod=winners_DoD(hist)
        wow=winners_WoW(hist)
        sur=surges_low_to_high(hist,baseline_weeks=int(baseline_weeks),surge_threshold=float(surge_gate))

        if not dod.empty: dod=dod[dod["rpc_lift_pct"]>=float(rpc_lift_gate)]
        if not wow.empty: wow=wow[wow["rpc_lift_pct"]>=float(rpc_lift_gate)]

        cols_dod=["Keyword","Device","Geo","RPC","rpc_lift_pct"]
        cols_wow=["Keyword","Device","Geo","RPC","rpc_lift_pct"]
        cols_sur=["Keyword","Device","Geo","RPC","lift_vs_base_pct"]

        save_csv(dod if not dod.empty else pd.DataFrame(columns=cols_dod),"winners_today_rpc.csv")
        save_csv(wow if not wow.empty else pd.DataFrame(columns=cols_wow),"winners_week_rpc.csv")
        save_csv(sur if not sur.empty else pd.DataFrame(columns=cols_sur),"surges_low_to_high_rpc.csv")

        st.session_state["dod_rpc"]=dod.copy() if isinstance(dod,pd.DataFrame) else pd.DataFrame(columns=cols_dod)
        st.session_state["wow_rpc"]=wow.copy() if isinstance(wow,pd.DataFrame) else pd.DataFrame(columns=cols_wow)
        st.session_state["sur_rpc"]=sur.copy() if isinstance(sur,pd.DataFrame) else pd.DataFrame(columns=cols_sur)
        st.session_state["today_roll_rpc"]=today_roll.copy() if isinstance(today_roll,pd.DataFrame) else pd.DataFrame()

        plan_pool=rank_and_pick_top_rpc(dod,wow,sur,top_n=int(TOP_N_PLAN)*3)
        patt=compile_blacklist()
        plan_pool=apply_blacklist(plan_pool,patt,text_cols=("Keyword",))
        if not plan_pool.empty:
            plan_pool.sort_values(["lift_metric","RPC"],ascending=[False,False],inplace=True)
            plan_top=pick_topN_avoiding_recent_rpc(plan_pool,int(TOP_N_PLAN),bool(prevent_relaunch),int(prevent_days))
        else:
            plan_top=pd.DataFrame()

        if not plan_top.empty and "segment" in plan_top.columns:
            dod_plan=plan_top[plan_top["segment"]=="DoD"].copy()
            wow_plan=plan_top[plan_top["segment"]=="WoW"].copy()
            sur_plan=plan_top[plan_top["segment"]=="Surge"].copy()
        else:
            dod_plan= pd.DataFrame(); wow_plan=pd.DataFrame(); sur_plan=pd.DataFrame()

        plan_cols=["segment","Keyword","Device","Geo","RPC","lift_metric"]
        def _pick_cols(df):
            if df is None or df.empty:
                return pd.DataFrame(columns=plan_cols)
            for c in plan_cols:
                if c not in df.columns:
                    df[c]=None
            return df[plan_cols]
        daily_plan=pd.concat([_pick_cols(dod_plan),_pick_cols(wow_plan),_pick_cols(sur_plan)],ignore_index=True)

        if build_plan:
            write_payload(dod_plan,wow_plan,sur_plan)
            daily_plan_path=os.path.join(OUTPUTS,"daily_plan_rpc.csv")
            daily_plan.to_csv(daily_plan_path,index=False)
            st.success(f"Saved RPC daily plan CSV top {int(TOP_N_PLAN)} → {daily_plan_path}")

        try:
            if not daily_plan.empty:
                append_launch_log_rpc(daily_plan[["Keyword","Device","Geo"]],source="rpc_daily_plan")
        except Exception as e:
            st.warning(f"Could not append to rpc launch log: {e}")

        try:
            _llp=LAUNCH_LOG_PATH
            if os.path.exists(_llp) and not daily_plan.empty:
                _log=pd.read_parquet(_llp)
                _log["launched_at"]=pd.to_datetime(_log["launched_at"],errors="coerce")
                _cut=pd.Timestamp.today().normalize()-pd.Timedelta(days=30)
                _recent=_log[_log["launched_at"]>=_cut].copy()
                for c in ("Keyword","Device","Geo"):
                    if c not in _recent.columns:
                        _recent[c]=""
                keys_recent=set(zip(_recent["Keyword"].astype(str),_recent["Device"].astype(str),_recent["Geo"].astype(str)))
                dupes_in_plan=sum((str(r["Keyword"]),str(r.get("Device","")),str(r.get("Geo",""))) in keys_recent for _,r in daily_plan.iterrows())
                st.caption(f"RPC plan sanity: {int(dupes_in_plan)} of {len(daily_plan)} appear in launch log within 30 days.")
        except Exception:
            pass

        st.success("RPC-only scan complete.")
        st.caption(f"Outputs folder: {OUTPUTS}")

    elif run and file is None:
        st.warning("Please upload a file before running the RPC scan.")

    # =====================
    # Public Signals — idle view
    # =====================
    _ps=st.empty()
    with _ps.container():
        if not st.session_state.get("ps_running_rpc",False):

            with st.expander("Public Signals — Settings feeds.yaml / modifiers.yaml",expanded=False):
                FEEDS_PATH=os.path.join(BASE_DIR,"feeds.yaml")
                MODS_PATH=os.path.join(BASE_DIR,"modifiers.yaml")

                def _safe_load_yaml(path,default):
                    try:
                        if os.path.exists(path):
                            with open(path,"r",encoding="utf-8") as f:
                                data=yaml.safe_load(f) or {}
                        else:
                            data={}
                    except Exception as e:
                        st.warning(f"Could not read {os.path.basename(path)}: {e}"); data={}
                    out=default.copy(); out.update({k:v for k,v in (data or {}).items() if v is not None})
                    return out

                def _safe_dump_yaml(path,data):
                    try:
                        with open(path,"w",encoding="utf-8") as f:
                            yaml.safe_dump(data,f,sort_keys=False,allow_unicode=True)
                        return True
                    except Exception as e:
                        st.error(f"Failed to save {os.path.basename(path)}: {e}")
                        return False

                default_feeds={"feeds":["https://www.bankrate.com/rss/","https://www.theverge.com/rss/index.xml","https://www.cnet.com/rss/news/"]}
                default_mods={"modifiers":[{"term":"prices","weight":1.0,"cpc_adj_pct":"+5"},{"term":"deals","weight":0.9,"cpc_adj_pct":"+5"},{"term":"near me","weight":0.6,"cpc_adj_pct":"-10"},{"term":"2025","weight":0.7,"cpc_adj_pct":"0"},{"term":"guide","weight":0.5,"cpc_adj_pct":"0"},{"term":"no contract","weight":0.6,"cpc_adj_pct":"0"},{"term":"for seniors","weight":0.8,"cpc_adj_pct":"0"}]}

                feeds_obj=_safe_load_yaml(FEEDS_PATH,default_feeds)
                mods_obj=_safe_load_yaml(MODS_PATH,default_mods)

                c1,c2=st.columns(2)
                with c1:
                    st.caption(f"feeds.yaml • {len(feeds_obj.get('feeds',[]))} feeds")
                    feeds_text=st.text_area("Edit feeds.yaml",value=yaml.safe_dump(feeds_obj,sort_keys=False,allow_unicode=True),height=220,key="feeds_yaml_editor")
                    if st.button("Save feeds.yaml",key="save_feeds_yaml"):
                        try:
                            parsed=yaml.safe_load(feeds_text) or {}
                            if not isinstance(parsed.get("feeds",[]),list):
                                raise ValueError("Expected top-level key 'feeds' with a list.")
                            if _safe_dump_yaml(FEEDS_PATH,parsed):
                                st.success(f"Saved {FEEDS_PATH}")
                        except Exception as e:
                            st.error(f"Invalid YAML: {e}")
                with c2:
                    st.caption(f"modifiers.yaml • {len(mods_obj.get('modifiers',[]))} modifiers")
                    mods_text=st.text_area("Edit modifiers.yaml",value=yaml.safe_dump(mods_obj,sort_keys=False,allow_unicode=True),height=220,key="mods_yaml_editor")
                    if st.button("Save modifiers.yaml",key="save_mods_yaml"):
                        try:
                            parsed=yaml.safe_load(mods_text) or {}
                            if not isinstance(parsed.get("modifiers",[]),list):
                                raise ValueError("Expected top-level key 'modifiers' with a list.")
                            for m in parsed["modifiers"]:
                                if "cpc_adj_pct" in m:
                                    m["cpc_adj_pct"]=str(m["cpc_adj_pct"]).strip()
                                    if m["cpc_adj_pct"] not in ("0","+0","-0"):
                                        if not m["cpc_adj_pct"].startswith(("+","-")):
                                            m["cpc_adj_pct"]="+"+m["cpc_adj_pct"]
                            if _safe_dump_yaml(MODS_PATH,parsed):
                                st.success(f"Saved {MODS_PATH}")
                        except Exception as e:
                            st.error(f"Invalid YAML: {e}")

                st.caption("Quick add")
                f_add=st.text_input("Add a feed URL",key="feed_add")
                m_term=st.text_input("Add modifier term",key="mod_add_term")
                m_weight=st.number_input("Weight",0.0,5.0,0.5,0.1,key="mod_add_weight")
                m_pct=st.text_input("CPC % adjust e.g. +5 -10 0","+0",key="mod_add_pct")
                c3,c4,c5=st.columns([1,1,2])
                with c3:
                    if st.button("➕ Add feed",key="btn_add_feed"):
                        feeds_obj["feeds"]=feeds_obj.get("feeds",[])
                        if f_add:
                            feeds_obj["feeds"].append(f_add.strip())
                            _safe_dump_yaml(FEEDS_PATH,feeds_obj)
                            st.success("Feed added.")
                with c4:
                    if st.button("➕ Add modifier",key="btn_add_mod"):
                        mods_obj["modifiers"]=mods_obj.get("modifiers",[])
                        if m_term:
                            mods_obj["modifiers"].append({"term":m_term.strip(),"weight":float(m_weight),"cpc_adj_pct":m_pct.strip()})
                            _safe_dump_yaml(MODS_PATH,mods_obj)
                            st.success("Modifier added.")
                with c5:
                    if st.button("Restore defaults",key="btn_restore_defaults"):
                        _safe_dump_yaml(FEEDS_PATH,default_feeds)
                        _safe_dump_yaml(MODS_PATH,default_mods)
                        st.success("Defaults restored.")

            with st.expander("Public Signals Trends + Suggest + RSS + Modifiers",expanded=False):
                st.text_input("AUG GEO","US",key="aug_geo")
                st.selectbox("AUG Timeframe",["7d","14d","30d","60d"],index=0,key="aug_tf")
                st.number_input("Max seeds",1,1000,10,1,key="aug_max_seeds")
                st.number_input("Top per seed per source",1,50,5,1,key="aug_top_per_seed")
                st.slider("Min score",0.0,1.0,0.35,0.01,key="aug_min_score")
                st.checkbox("Only multi-signal hits",value=False,key="aug_multi_only")
                st.text_area("Manual seed keywords comma or newline", "",height=100,key="ps_manual_seeds_raw")

                # live preview now that state exists
                _preview,_src=_collect_seeds_preview(int(st.session_state.get("aug_max_seeds",50)))
                _manual_list_preview=[s.strip() for s in re.split(r"[\n,]",st.session_state.get("ps_manual_seeds_raw","")) if s.strip()]
                if _manual_list_preview:
                    seen_m=set(); merged_m=[]
                    for s in list(_preview)+_manual_list_preview:
                        if s not in seen_m:
                            seen_m.add(s); merged_m.append(s)
                    _preview=merged_m[: int(st.session_state.get("aug_max_seeds",50))]
                st.caption(f"Seed source: {_src or 'none'} • manual added: {len(_manual_list_preview)} • total preview: {len(_preview)}")
                st.write({"seed_preview":_preview[:10],"seed_count":len(_preview)})

            if st.button("Run Public Signals",key="public_signals_btn"):
                st.session_state["ps_running_rpc"]=True
                st.rerun()

        # RUN VIEW
        else:
            run_box=st.status("Collecting… 0/0 seeds · 0s elapsed",state="running",expanded=True)
            prog_main=st.progress(0,text="0/0 seeds · 0s elapsed")
            prog_side=st.sidebar.progress(0,text="0/0 seeds · 0s elapsed")
            side_box=st.sidebar.container(); side_box.caption("Public Signals progress")

            def _update_prog(i:int,total_cap:int,started_ts:float):
                cap=max(1,total_cap)
                pct=int((i/cap)*100)
                txt=f"{i}/{cap} seeds · {int(time.time()-started_ts)}s elapsed"
                run_box.update(label=f"Collecting… {txt}")
                prog_main.progress(pct,text=txt)
                prog_side.progress(pct,text=txt)

            try:
                import feedparser
                from pytrends.request import TrendReq
            except Exception:
                run_box.update(label="Missing deps. Install: pytrends feedparser pyyaml requests",state="error")
                st.session_state["ps_running_rpc"]=False
                st.stop()

            socket.setdefaulttimeout(6)
            TRENDS_TIMEOUT=(3,12)
            SUGGEST_TIMEOUT=5

            def _read_yaml(path,key):
                try:
                    with open(path,"r",encoding="utf-8") as f:
                        data=yaml.safe_load(f) or {}
                    return data.get(key,[])
                except Exception:
                    return []

            feeds=_read_yaml(os.path.join(BASE_DIR,"feeds.yaml"),"feeds")
            modifiers=_read_yaml(os.path.join(BASE_DIR,"modifiers.yaml"),"modifiers")

            def _collect_seeds(limit:int):
                def _detect_keyword_col(df):
                    cand={c.lower().strip():c for c in df.columns}
                    for k in ("keyword","kw"):
                        if k in cand: return cand[k]
                    return "Keyword" if "Keyword" in df.columns else None
                seeds,source=[], ""
                for key in ("dod","wow","sur","dod_rpc","wow_rpc","sur_rpc"):
                    df=st.session_state.get(key)
                    if isinstance(df,pd.DataFrame) and not df.empty:
                        kw_col=_detect_keyword_col(df)
                        if kw_col:
                            seeds.extend(df[kw_col].astype(str).tolist())
                            source=source or f"session:{key}"
                if not seeds:
                    for key in ("today_roll","today_roll_rpc"):
                        tr=st.session_state.get(key)
                        if isinstance(tr,pd.DataFrame) and not tr.empty:
                            kw_col=_detect_keyword_col(tr)
                            if kw_col:
                                if "Clicks" in tr.columns:
                                    seeds.extend(tr.sort_values("Clicks",ascending=False)[kw_col].astype(str).tolist())
                                else:
                                    seeds.extend(tr[kw_col].astype(str).tolist())
                                source=f"session:{key}"; break
                if not seeds:
                    for fname in ("rollup_daily.csv","rollup_daily_rpc.csv"):
                        path=os.path.join(OUTPUTS,fname)
                        if os.path.exists(path):
                            df=pd.read_csv(path)
                            kw_col=_detect_keyword_col(df)
                            if kw_col:
                                if "Clicks" in df.columns:
                                    seeds.extend(df.sort_values("Clicks",ascending=False)[kw_col].astype(str).tolist())
                                else:
                                    seeds.extend(df[kw_col].astype(str).tolist())
                                source=f"file:{fname}"; break
                if not seeds:
                    hist=load_history()
                    if isinstance(hist,pd.DataFrame) and not hist.empty:
                        hist["date"]=pd.to_datetime(hist["date"],errors="coerce")
                        latest=hist["date"].max()
                        if pd.notna(latest):
                            day=hist[hist["date"]==latest]
                            kw_col=_detect_keyword_col(day)
                            if kw_col:
                                if "Clicks" in day.columns:
                                    seeds.extend(day.sort_values("Clicks",ascending=False)[kw_col].astype(str).tolist())
                                else:
                                    seeds.extend(day[kw_col].astype(str).tolist())
                                source="file:history(latest)"
                seen,out=set(),[]
                for s in seeds:
                    s=str(s).strip(); key=s.casefold()
                    if s and key not in seen:
                        seen.add(key); out.append(s)
                patt_local=compile_blacklist()
                if patt_local:
                    out=[s for s in out if not patt_local.search(s)]
                return out[: int(limit)],source

            rows=[]
            seeds,_source=_collect_seeds(int(st.session_state.get("aug_max_seeds",10)))
            total=len(seeds)
            start_t=time.time()
            write_every=max(1,min(total,int(st.session_state.get("aug_max_seeds",10)))//4)

            pool_parts=[]
            for key in ("dod","wow","sur","dod_rpc","wow_rpc","sur_rpc"):
                df_ss=st.session_state.get(key)
                if isinstance(df_ss,pd.DataFrame) and not df_ss.empty:
                    pool_parts.append(df_ss)
            winners_pool=pd.concat(pool_parts,ignore_index=True) if pool_parts else pd.DataFrame()
            if "RPC" in winners_pool.columns and not winners_pool.empty:
                rpc_min=float(winners_pool["RPC"].min()); rpc_max=float(winners_pool["RPC"].max())
            else:
                rpc_min,rpc_max=0.0,1.0

            STOPWORDS=set("a an the for of to with and or & near in on at by from 2025 2024".split())
            def _tokens(s:str): return [t for t in re.findall(r"[a-z0-9]+",str(s).lower()) if t not in STOPWORDS and len(t)>2]
            def _text_overlap(a:str,b:str)->float:
                A,B=set(_tokens(a)),set(_tokens(b))
                return 0.0 if not A or not B else len(A & B)/max(1,len(A | B))
            def _nearest_cpc_cap(pool_df,target_rpc):
                if pool_df is None or pool_df.empty or "cpc_cap" not in pool_df.columns:
                    return 0.25
                tmp=pool_df.copy(); tmp["diff"]=(tmp.get("RPC",0.0)-float(target_rpc))**2
                tmp.sort_values("diff",inplace=True)
                return float(tmp.iloc[0].get("cpc_cap",0.25))
            def _trend_norm(scores):
                if not scores: return []
                vals=[v for _,v in scores]; mx=max(vals) or 1.0
                return [(q,(v/mx)) for q,v in scores]
            def _modifier_weight(cand,mod_list):
                cand_l=cand.lower(); best=("",0.0,0)
                for m in (mod_list or []):
                    term=str(m.get("term","")).lower().strip()
                    if term and term in cand_l:
                        w=float(m.get("weight",0.0)); adj=int(str(m.get("cpc_adj_pct",0)).replace("+",""))
                        if w>best[1]: best=(term,w,adj)
                return best
            def _trends(seed:str):
                try:
                    pytrends=TrendReq(hl="en-US",tz=360,timeout=(3,12))
                    timeframe_map={"7d":"now 7-d","14d":"now 7-d","30d":"today 1-m","60d":"today 3-m"}
                    pytrends.build_payload([seed],timeframe=timeframe_map.get(st.session_state.get("aug_tf","7d"),"now 7-d"),geo=st.session_state.get("aug_geo","US"))
                    rq=pytrends.related_queries(); rising=rq.get(seed,{}).get("rising")
                    if rising is None or rising.empty: return []
                    res=[]
                    top=int(st.session_state.get("aug_top_per_seed",5))
                    for _,r in rising.head(top).iterrows():
                        q=str(r.get("query","")).strip(); v=float(r.get("value",0.0))
                        if q: res.append((q,v))
                    return _trend_norm(res)
                except Exception:
                    return []
            def _suggest(seed:str):
                def g(q):
                    try:
                        r=requests.get("https://suggestqueries.google.com/complete/search",params={"client":"firefox","q":q},timeout=5); j=r.json()
                        top=int(st.session_state.get("aug_top_per_seed",5)); return [s for s in j[1][:top] if isinstance(s,str)]
                    except Exception: return []
                def b(q):
                    try:
                        r=requests.get("https://api.bing.com/osjson.aspx",params={"query":q},timeout=5); j=r.json()
                        top=int(st.session_state.get("aug_top_per_seed",5)); return [s for s in j[1][:top] if isinstance(s,str)]
                    except Exception: return []
                def yt(q):
                    try:
                        r=requests.get("https://suggestqueries.google.com/complete/search",params={"client":"firefox","ds":"yt","q":q},timeout=5); j=r.json()
                        top=int(st.session_state.get("aug_top_per_seed",5)); return [s for s in j[1][:top] if isinstance(s,str)]
                    except Exception: return []
                out=[]; intents=[""," prices"," deals"]
                for mod in intents:
                    q=(seed+mod).strip(); out.extend(g(q)); out.extend(b(q)); out.extend(yt(q))
                seen_s,uniq=set(),[]
                for s in out:
                    if s not in seen_s:
                        seen_s.add(s); uniq.append(s)
                top=int(st.session_state.get("aug_top_per_seed",5))
                return uniq[:top]
            def _rss_candidates(cands):
                try:
                    import feedparser
                except Exception:
                    return {c:0.0 for c in cands}
                if not feeds: return {c:0.0 for c in cands}
                out_map={}
                for cand in cands:
                    hits=0; total=0; patt=re.compile(r"\b"+re.escape(cand.lower())+r"\b")
                    for url in feeds:
                        try:
                            total+=1
                            d=feedparser.parse(url)
                            titles=[e.get("title","") for e in (d.entries or [])][:50]
                            joined=" || ".join(titles).lower()
                            if patt.search(joined): hits+=1
                        except Exception:
                            pass
                    out_map[cand]=(hits/total) if total else 0.0
                return out_map

            winners_pool=winners_pool  # from above
            seeds,_source=_collect_seeds(int(st.session_state.get("aug_max_seeds",10)))
            if not seeds:
                run_box.update(label="No seeds found auto or manual.",state="complete")
                st.session_state["ps_running_rpc"]=False
                st.rerun()

            for i,seed in enumerate(seeds[: int(st.session_state.get("aug_max_seeds",10))],start=1):
                if not winners_pool.empty:
                    seed_rows=winners_pool[winners_pool.get("Keyword","").astype(str).str.lower()==str(seed).lower()]
                    seed_rpc=float(seed_rows["RPC"].mean()) if not seed_rows.empty else 0.0
                else:
                    seed_rpc=0.0
                base_cap=_nearest_cpc_cap(winners_pool,seed_rpc)
                t1=_trends(seed)
                t2=_suggest(seed)
                union=set([q for q,_ in t1]) | set(t2)
                rss_map=_rss_candidates(list(union))
                injected=[]
                for m in (modifiers or []):
                    term=str(m.get("term","")).strip()
                    if term:
                        injected.append(f"{seed} {term}".strip())
                union|=set(injected)
                trend_lookup={q:s for q,s in t1}
                suggest_set=set(t2)
                for q in list(union):
                    tn=float(trend_lookup.get(q,0.0))
                    ov=float(rss_map.get(q,0.0))
                    came_from_suggest=q in suggest_set
                    base=0.18 if came_from_suggest else 0.0
                    sim=_text_overlap(q,seed)
                    rpc_norm=0.0 if rpc_max<=rpc_min else max(0.0,min(1.0,(seed_rpc-rpc_min)/(rpc_max-rpc_min)))
                    mod_term,mod_w,mod_adj=_modifier_weight(q,modifiers)
                    rec_cap=round(base_cap*(1.0+(mod_adj/100.0)),2)
                    raw=(0.45*tn)+(0.25*ov)+(0.12*sim)+(0.10*rpc_norm)+(0.08*mod_w)+base
                    score=round(min(1.0,raw),3)
                    rows.append({"seed":seed,"candidate":q,"source_flags":json.dumps({"trends":bool(tn),"suggest":came_from_suggest,"rss":ov>0.0,"injected":q in set(injected)}),"geo":st.session_state.get("aug_geo","US"),"window":st.session_state.get("aug_tf","7d"),"trend_norm":round(tn,3),"overlap":round(ov,3),"similarity":round(sim,3),"seed_rpc_norm":round(rpc_norm,3),"modifier":mod_term,"modifier_weight":round(mod_w,3),"score":score,"rec_cpc_cap":rec_cap})
                _update_prog(i,min(len(seeds),int(st.session_state.get("aug_max_seeds",10))),start_t)
                if i%write_every==0 and rows:
                    pd.DataFrame(rows).sort_values(["score","trend_norm"],ascending=[False,False]).to_csv(os.path.join(EXPANSIONS,"expansion_candidates_rpc.tmp.csv"),index=False)

            if rows:
                df_exp=pd.DataFrame(rows).sort_values(["score","trend_norm"],ascending=[False,False])
                def _count_signals(flags_json):
                    try:
                        d=json.loads(flags_json); return sum(1 for v in d.values() if v)
                    except Exception:
                        return 0
                df_exp["signals"]=df_exp["source_flags"].apply(_count_signals)
                if st.session_state.get("aug_multi_only",False):
                    df_exp=df_exp[df_exp["signals"]>=2]
                min_score=float(st.session_state.get("aug_min_score",0.35))
                df_filt=df_exp[df_exp["score"]>=min_score]
                if df_filt.empty and not df_exp.empty:
                    relaxed=max(0.10,min_score*0.5); df_filt=df_exp[df_exp["score"]>=relaxed]
                if df_filt.empty:
                    run_box.update(label="No candidates met filters.",state="complete")
                    st.info("Lower Min score add more seeds or widen timeframe.")
                else:
                    outp=os.path.join(EXPANSIONS,"expansion_candidates_rpc.csv")
                    df_filt.to_csv(outp,index=False)
                    run_box.update(label=f"Done. Saved {len(df_filt)} candidates to {outp}",state="complete")
                    st.dataframe(df_filt,use_container_width=True)
            else:
                run_box.update(label="No candidates produced.",state="complete")

            st.session_state["ps_running_rpc"]=False
            st.rerun()

    st.caption(f"Outputs folder: {OUTPUTS}")
    st.caption(f"Expansions folder: {EXPANSIONS}")

    # Sidebar snapshot
    with st.sidebar:
        def _cnt(df): return 0 if not isinstance(df,pd.DataFrame) else int(len(df))
        st.caption("Scan snapshot RPC-only")
        st.write({"DoD":_cnt(st.session_state.get("dod_rpc")),"WoW":_cnt(st.session_state.get("wow_rpc")),"Surges":_cnt(st.session_state.get("sur_rpc")),"Rollup rows":_cnt(st.session_state.get("today_roll_rpc"))})
        dp_path=os.path.join(OUTPUTS,"daily_plan_rpc.csv")
        try:
            if os.path.exists(dp_path):
                dp=pd.read_csv(dp_path)
                st.divider(); st.caption(f"Daily Plan RPC preview — {len(dp)} rows")
                if not dp.empty:
                    preview_cols=[c for c in ["segment","Keyword","Device","Geo","RPC","lift_metric"] if c in dp.columns]
                    st.dataframe(dp[preview_cols].head(10),use_container_width=True,height=240)
                    st.download_button("Download daily_plan_rpc.csv",data=dp.to_csv(index=False),file_name="daily_plan_rpc.csv",mime="text/csv",key="dl_daily_plan_rpc")
                else:
                    st.write("(daily_plan_rpc.csv is empty)")
            else:
                st.caption("Daily Plan RPC: not created yet")
        except Exception as e:
            st.warning(f"Daily Plan preview error: {e}")

if __name__=="__main__":
    main()