import argparse, zipfile, os, pandas as pd, numpy as np, json, re
from collections import defaultdict

TAG_PRIORITIES = {
 "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","Revenues","RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueGoodsNet","SalesRevenueServicesNet","RevenuesNetOfInterestExpense"],
 "Net Profit": ["NetIncomeLoss","ProfitLoss"],
 "Operating Expenses": ["OperatingExpenses","CostsAndExpenses"],
 "Current Assets": ["AssetsCurrent"],
 "Current Liabilities": ["LiabilitiesCurrent"],
 "Total Debt": ["LongTermDebtCurrent","ShortTermBorrowings","LongTermDebt","LongTermDebtNoncurrent"],
 "Shareholders Equity": ["StockholdersEquity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
 "Total Assets": ["Assets"],
 "Total Liabilities": ["Liabilities"],
 "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities"]
}
FLOW={"Revenue","Net Profit","Operating Expenses","Operating Cash Flow"}
TAG_TO_METRIC={t:m for m,ts in TAG_PRIORITIES.items() for t in ts}
TAG_PRIORITY={t:i for ts in TAG_PRIORITIES.values() for i,t in enumerate(ts)}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input", nargs="+", required=True); p.add_argument("--output", required=True)
 a=p.parse_args()
 allrows=[]
 for ix, path in enumerate(a.input,1):
  print(f"[{ix}/{len(a.input)}] Processing {os.path.basename(path)}", flush=True)
  with zipfile.ZipFile(path) as zf:
   names={n.lower():n for n in zf.namelist()}
   with zf.open(names["sub.txt"]) as f:
    sub=pd.read_csv(f,sep="\t",dtype=str,low_memory=False)
   sub.columns=[c.lower() for c in sub.columns]
   sub=sub[sub["form"].str.upper().isin(["10-K","10-K/A"])].copy()
   if sub.empty: continue
   # Keep submissions from annual forms, with one name per accession.
   sub=sub[["adsh","cik","name","form","filed","period"]].drop_duplicates("adsh")
   adsh_set=set(sub.adsh.astype(str))
   with zf.open(names["num.txt"]) as f:
    chunks=pd.read_csv(f,sep="\t",dtype={"adsh":str,"tag":str,"version":str,"ddate":str,"qtrs":str,"uom":str,"segments":str,"coreg":str,"value":str},chunksize=750000,low_memory=False)
    kept=[]
    for ch in chunks:
     ch=ch[ch.adsh.isin(adsh_set)]
     if ch.empty: continue
     ch["tag_base"]=ch.tag.astype(str).str.split(":").str[-1]
     ch=ch[ch.tag_base.isin(TAG_TO_METRIC)]
     if ch.empty: continue
     ch["qtrs_num"]=pd.to_numeric(ch.qtrs,errors="coerce")
     ch["value_num"]=pd.to_numeric(ch.value,errors="coerce")
     ch["ddate"]=ch.ddate.astype(str)
     # SEC facts should be unsegmented consolidated values in USD.
     if "segments" in ch: ch=ch[ch.segments.fillna("").astype(str).str.strip().isin(["","nan","None"])]
     if "coreg" in ch: ch=ch[ch.coreg.fillna("").astype(str).str.strip().isin(["","nan","None"])]
     ch=ch[ch.uom.eq("USD") & ch.value_num.notna()]
     ch["Metric"]=ch.tag_base.map(TAG_TO_METRIC)
     ch=ch[((ch.Metric.isin(FLOW)) & (ch.qtrs_num==4)) | ((~ch.Metric.isin(FLOW)) & (ch.qtrs_num==0))]
     if not ch.empty: kept.append(ch[["adsh","tag_base","ddate","qtrs_num","value_num","Metric"]])
   if not kept: continue
   facts=pd.concat(kept,ignore_index=True)
   facts=facts.merge(sub[["adsh","cik","name","filed"]],on="adsh",how="inner")
   facts["period"]=pd.to_numeric(facts.ddate.str.slice(0,4),errors="coerce")
   facts["priority"]=facts.tag_base.map(TAG_PRIORITY).fillna(999)
   facts=facts.dropna(subset=["period"])
   # Within each filing/year/metric choose best-priority tag. For debt, sum current/noncurrent components only if both present; simplified to priority-selected one.
   facts=facts.sort_values(["priority","filed"])
   facts=facts.drop_duplicates(["adsh","period","Metric"],keep="first")
   facts=facts.sort_values("filed").drop_duplicates(["cik","period","Metric"],keep="last")
   if not facts.empty:
    allrows.append(facts.rename(columns={"cik":"company_id","name":"company_name","value_num":"Value"})[["company_id","company_name","period","Metric","Value","filed"]])
 if not allrows: raise SystemExit("No annual facts found. Check SEC ZIP formats.")
 facts=pd.concat(allrows,ignore_index=True)
 facts["company_id"]=facts.company_id.astype(str)
 facts=facts.sort_values("filed").drop_duplicates(["company_id","period","Metric"],keep="last")
 wide=facts.pivot_table(index=["company_id","company_name","period"],columns="Metric",values="Value",aggfunc="last").reset_index()
 wide.columns.name=None
 for metric in TAG_PRIORITIES:
  if metric not in wide: wide[metric]=np.nan
 wide["period"]=pd.to_numeric(wide.period,errors="coerce")
 wide=wide.dropna(subset=["period"])
 os.makedirs(os.path.dirname(a.output) or ".",exist_ok=True)
 wide.to_csv(a.output,index=False)
 print(f"Wrote {len(wide):,} company-year rows; companies={wide.company_id.nunique():,}; years={wide.period.min():.0f}-{wide.period.max():.0f}",flush=True)
 print("Metrics available:",{c:int(wide[c].notna().sum()) for c in TAG_PRIORITIES},flush=True)
if __name__=="__main__": main()
