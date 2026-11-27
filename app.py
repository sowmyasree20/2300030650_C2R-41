import os, re, json
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
st.set_page_config(page_title="Culturally Grounded Multilingual Evaluation", page_icon="🌍", layout="wide")

MILU_DEFAULT = r"C:\Users\nihar\Milu\MILU_Telugu_Cultural_Review.csv"
BLEND_DEFAULT = r"C:\Users\nihar\Downloads\BLEnD-main\BLEnD-main"
INDIC = {"bn":"Bengali","gu":"Gujarati","hi":"Hindi","kn":"Kannada","ml":"Malayalam","mr":"Marathi","or":"Odia","pa":"Punjabi","ta":"Tamil","te":"Telugu"}

def s(x): return "" if x is None else str(x).strip()

def milu_answer(r):
    t=s(r.get("target","")); opts=[s(r.get(f"option{i}","")) for i in range(1,5)]
    if t.upper() in "ABCD" and len(t)==1: return opts[ord(t.upper())-65]
    try:
        n=int(float(t))
        if 1<=n<=4: return opts[n-1]
    except: pass
    return t

@st.cache_data
def load_milu(path):
    df=pd.read_csv(path).fillna("")
    for c in ["question","option1","option2","option3","option4","target","language","domain","subject"]:
        if c not in df: df[c]=""
    df["_answer"]=df.apply(milu_answer,axis=1)
    df["_search"]=df["question"].astype(str)+" "+df["subject"].astype(str)+" "+df["domain"].astype(str)
    return df

def findcol(df, names):
    m={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in m:return m[n.lower()]
    return None

def parse_ann(x):
    out=[]
    def add(v,c=None):
        if isinstance(v,str) and v.strip(): out.append((v.strip(),c))
        elif isinstance(v,dict):
            for k,val in v.items():
                if isinstance(val,(int,float)): out.append((str(k),val))
                elif isinstance(k,str) and k.strip(): out.append((k,None))
    if isinstance(x,dict):
        for k in ["answers","en_answers","accepted_answers","gold_answers"]:
            v=x.get(k)
            if isinstance(v,list):
                for z in v:
                    if isinstance(z,dict): add(z.get("answer") or z.get("text") or z.get("value"),z.get("count"))
                    else:add(z)
            elif isinstance(v,dict): add(v)
        for k in ["annotations","annotation"]:
            v=x.get(k)
            if isinstance(v,list):
                for z in v:
                    if isinstance(z,dict): add(z.get("answer") or z.get("text"),z.get("count"))
    d={}
    for a,c in out:
        if a not in d or (c or 0)>(d[a] or 0): d[a]=c
    return [{"answer":a,"count":c} for a,c in sorted(d.items(),key=lambda z:(z[1] or 0),reverse=True)]

@st.cache_data
def load_blend(root):
    root=Path(root); qdir=root/"data"/"questions"; adir=root/"data"/"annotations"
    if not qdir.exists(): raise FileNotFoundError(f"BLEnD questions folder not found: {qdir}")
    if not adir.exists(): raise FileNotFoundError(f"BLEnD annotations folder not found: {adir}")
    anns={}
    for f in adir.glob("*_data.json"):
        try:data=json.loads(f.read_text(encoding="utf-8"))
        except:continue
        if isinstance(data,dict):
            for k,v in data.items():
                if isinstance(v,dict): anns[str(k)]=v
            for k in ["data","annotations","questions","items"]:
                if isinstance(data.get(k),list):
                    for z in data[k]:
                        if isinstance(z,dict) and z.get("id"): anns[str(z["id"])]=z
        elif isinstance(data,list):
            for z in data:
                if isinstance(z,dict) and z.get("id"): anns[str(z["id"])]=z
    rows=[]
    for f in qdir.glob("*_questions.*"):
        if f.suffix.lower() not in [".xlsx",".xls",".csv",".tsv"]: continue
        try:
            q=pd.read_csv(f) if f.suffix.lower()==".csv" else pd.read_csv(f,sep="\t") if f.suffix.lower()==".tsv" else pd.read_excel(f)
        except: continue
        ic=findcol(q,["id","ID","question_id"]); qc=findcol(q,["question","Question","query"]); tc=findcol(q,["topic","Topic"]); sc=findcol(q,["source","Source"])
        if not qc: continue
        for _,r in q.iterrows():
            rid=s(r[ic]) if ic else ""; rows.append({"id":rid,"question":s(r[qc]),"topic":s(r[tc]) if tc else "","source":s(r[sc]) if sc else f.stem,"accepted_answers":parse_ann(anns.get(rid,{}))})
    return pd.DataFrame(rows)

def retrieve(df,q,kind,k=3):
    if df.empty:return []
    if kind=="MILU": corpus=(df["question"].astype(str)+" "+df["subject"].astype(str)+" "+df["domain"].astype(str)).tolist()
    else: corpus=(df["question"].astype(str)+" "+df["topic"].astype(str)+" "+df["source"].astype(str)).tolist()
    v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True); X=v.fit_transform(corpus); z=cosine_similarity(v.transform([q]),X).ravel()
    return [(float(z[i]),df.iloc[i].to_dict()) for i in z.argsort()[::-1][:k]]

def lang(q):
    if re.search(r"[\u0C00-\u0C7F]",q):return "te"
    if re.search(r"[\u0980-\u09FF]",q):return "bn"
    if re.search(r"[\u0900-\u097F]",q):return "hi"
    if re.search(r"[\u0A80-\u0AFF]",q):return "gu"
    if re.search(r"[\u0C80-\u0CFF]",q):return "kn"
    if re.search(r"[\u0D00-\u0D7F]",q):return "ml"
    if re.search(r"[\u0B00-\u0B7F]",q):return "or"
    if re.search(r"[\u0B80-\u0BFF]",q):return "ta"
    if re.search(r"[\u0A00-\u0A7F]",q):return "pa"
    try:return detect(q)
    except:return "en"

def refs(items,kind):
    a=[]
    for _,r in items:
        if kind=="MILU":
            if s(r.get("_answer")):a.append(s(r["_answer"]))
        else:
            for z in r.get("accepted_answers",[]) or []:
                if s(z.get("answer")):a.append(s(z["answer"]))
    seen=set(); out=[]
    for x in a:
        if x.casefold() not in seen:seen.add(x.casefold());out.append(x)
    return out

def sim(a,b):return SequenceMatcher(None,s(a).casefold(),s(b).casefold()).ratio()
def evaluate(a,rs):
    if not rs:return ("NO_REFERENCE",0,"")
    score,best=max((sim(a,x),x) for x in rs)
    return ("PASS" if score>=.70 else "PARTIAL" if score>=.45 else "REVIEW",score,best)

def api_answer(q,rs,evidence,model):
    key=os.getenv("OPENAI_API_KEY","").strip()
    if not key:return None
    try:
        from openai import OpenAI
        c=OpenAI(api_key=key)
        ctx="\n\n".join(evidence)
        p=f"Answer the cultural question using only the retrieved evidence. Do not invent facts.\nQuestion: {q}\nReference answers: {'; '.join(rs)}\nEvidence:\n{ctx}"
        return c.responses.create(model=model,input=p).output_text
    except Exception as e:
        st.warning(f"API call failed; using demo mode: {type(e).__name__}: {e}")
        return None

st.title("🌍 Culturally Grounded Multilingual Evaluation Prototype")
st.caption("Routing → benchmark retrieval → accepted answers → optional LLM → RAG evidence → remediation → re-evaluation")

with st.sidebar:
    st.header("Configuration")
    mp=st.text_input("MILU CSV",MILU_DEFAULT)
    bp=st.text_input("BLEnD folder",BLEND_DEFAULT)
    model=st.text_input("LLM model (optional)",os.getenv("OPENAI_MODEL","gpt-5.6-luna"))
    if st.button("Reload datasets"):
        load_milu.clear(); load_blend.clear(); st.rerun()

try:milu=load_milu(mp); me=None
except Exception as e:milu=pd.DataFrame();me=str(e)
try:blend=load_blend(bp);be=None
except Exception as e:blend=pd.DataFrame();be=str(e)

a,b,c=st.columns(3);a.metric("MILU questions loaded",len(milu));b.metric("BLEnD questions loaded",len(blend));c.metric("BLEnD annotated questions",sum(bool(x) for x in blend["accepted_answers"]) if not blend.empty else 0)
if me:st.error("MILU loading failed");st.code(me)
if be:st.error("BLEnD loading failed");st.code(be)

q=st.text_area("Enter a cultural question","What is the main dish for Thanksgiving in the US?",height=110)

if st.button("Run End-to-End Evaluation",type="primary"):
    lg=lang(q); kind="MILU" if lg in INDIC else "BLEnD"; items=retrieve(milu if kind=="MILU" else blend,q,kind)
    rs=refs(items,kind)

    st.subheader("1. Routing")
    st.write(f"**Detected language:** `{lg}` — {INDIC.get(lg,'Non-Indic')}")
    st.write(f"**Selected benchmark:** `{kind}`")
    if kind=="BLEnD" and lg=="en":st.info("Prototype note: language-based routing sends English to BLEnD; final work should add explicit culture-context detection.")

    st.subheader("2. Retrieved benchmark evidence + reference answers")
    for score,r in items:
        if kind=="MILU":
            st.markdown(f"**{s(r.get('question'))}** — `score={score:.3f}`")
            st.write(f"Reference answer: **{s(r.get('_answer')) or 'Not available'}**")
        else:
            st.markdown(f"**{s(r.get('id'))}** — `score={score:.3f}` — {s(r.get('source'))}")
            st.write(s(r.get('question')))
            aa=r.get("accepted_answers",[]) or []
            if aa:
                st.write("**Accepted/reference answers:**")
                for z in aa[:10]:st.write(f"- {z['answer']}" + (f" — count {z['count']}" if z.get("count") is not None else ""))
            else:st.warning("No annotation answers found for this item.")

    evidence=[]
    for score,r in items:
        if score<=0:continue
        if kind=="MILU":
            evidence.append(f"[MILU] Question: {s(r.get('question'))}\nReference answer: {s(r.get('_answer'))}")
        else:
            aa=", ".join(x["answer"] for x in (r.get("accepted_answers",[]) or [])[:10])
            evidence.append(f"[BLEnD | {s(r.get('source'))}] Question: {s(r.get('question'))}\nAccepted answers: {aa}")

    ans=api_answer(q,rs,evidence,model)
    if ans:
        st.subheader("3. LLM response");st.success("Real LLM mode");st.write(ans)
    else:
        ans=("Demo grounded response: "+"; ".join(rs[:3])+".") if rs else "Demo mode: no reference answer retrieved."
        st.subheader("3. Response");st.info("No API key configured. This is a deterministic benchmark-grounded demo response, NOT a real LLM generation.");st.write(ans)

    status,score,best=evaluate(ans,rs)
    st.subheader("4. Evaluation before remediation")
    x,y,z=st.columns(3);x.metric("Reference status",status);y.metric("Best similarity",f"{score:.2f}");z.metric("Best reference",best[:70] if best else "None")
    st.write("**Cultural/reference error flag:**", "REVIEW REQUIRED" if status in ["REVIEW","NO_REFERENCE"] else "No error flag")

    st.subheader("5. RAG evidence")
    if evidence:
        for e in evidence:st.code(e)
    else:st.warning("No positive-similarity evidence retrieved.")

    st.subheader("6. Evidence-grounded remediation")
    corrected=("Evidence-grounded demo revision: "+"; ".join(rs[:3])+".") if rs else ans
    st.write(corrected)
    st.caption("This prototype uses benchmark/reference evidence as the retrieval store. Final research evaluation should add independently sourced cultural evidence and human/native-speaker validation.")

    st.subheader("7. Final re-evaluation")
    s2,sc2,b2=evaluate(corrected,rs)
    x,y,z=st.columns(3);x.metric("Final status",s2);y.metric("Final similarity",f"{sc2:.2f}");z.metric("Final reference",b2[:70] if b2 else "None")
    st.success("Pipeline completed: routing → retrieval → reference answers → response → evaluation → RAG evidence → remediation → re-evaluation.")
