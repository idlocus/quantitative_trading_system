#!/usr/bin/env python3
import subprocess, json, os, shlex

SKILL_DIR = os.path.expanduser(r"C:\Users\hua.guo\.agents\skills\wind-mcp-skill")
CLI = os.path.join(SKILL_DIR, "scripts", "cli.mjs")

def mcp_call(server_type, tool_name, params):
    params_json = json.dumps(params, ensure_ascii=False)
    cmd = ["node", CLI, "call", server_type, tool_name, params_json]
    out = subprocess.check_output(cmd, cwd=SKILL_DIR, stderr=subprocess.STDOUT, timeout=25)
    return json.loads(out)

# 1. K线
raw = mcp_call("index_data", "get_index_kline",
               {"windcode": "000985.SH", "begin_date": "20260515", "end_date": "20260529"})
text = json.loads(raw[0]["text"])
rows = text["data"]["rows"]
dates = [r[9] for r in rows]
closes = [float(r[2]) for r in rows]

print("=== 中证全指 K线 ===")
for d, c in zip(dates, closes):
    print(f"  {d}: {c:.2f}")

changes = [(dates[i], (closes[i] - closes[i-1]) / closes[i-1] * 100) for i in range(1, len(closes))]
print("\n=== 每日涨跌幅 ===")
for d, c in changes:
    print(f"  {d}: {c:+.2f}%")

chg_pos = sum(1 for _, c in changes if c > 0)
breadth_score = 50 + (chg_pos / len(changes) - 0.5) * 50
print(f"\n上涨日 {chg_pos}/{len(changes)} → 宽度分={breadth_score:.0f}")

# 2. 估值分位
val_raw = mcp_call("index_data", "get_index_technicals",
                   {"windcode": "000985.SH", "question": "中证全指PE/PB历史分位"})
val_text = json.loads(val_raw[0]["text"])
val_rows = val_text["data"]["data"][0]["rows"]
pe_pct = float(val_rows[6])
pb_pct = float(val_rows[7])
avg_val_pct = (pe_pct + pb_pct) / 2
val_score = (1 - avg_val_pct) * 100
print(f"\nPE分位={pe_pct*100:.1f}%  PB分位={pb_pct*100:.1f}%  → 估值分={val_score:.0f}")

market_score = int(breadth_score * 0.5 + val_score * 0.5)
pf = max(0.2, min(1.0, market_score / 100))
print(f"\n=== 综合市场评分: {market_score}  建议仓位: {pf:.0%} ===")