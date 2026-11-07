"""Set the team details everywhere in one go (team.json, README, report, docs, eval file).

  python scripts/set_team.py --team_id T07 --team_name "Night Owls" --usns 1XX22CS001,1XX22CS002
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--team_id", required=True)
ap.add_argument("--team_name", required=True)
ap.add_argument("--usns", required=True, help="comma-separated, same order as members in team.json")
a = ap.parse_args()

p = os.path.join(ROOT, "team.json")
team = json.load(open(p))
old = {"id": team["team_id"], "name": team["team_name"], "usns": [m["usn"] for m in team["members"]]}
usns = [u.strip() for u in a.usns.split(",")]
if len(usns) != len(team["members"]):
    sys.exit(f"need {len(team['members'])} USNs, got {len(usns)}")
team["team_id"], team["team_name"] = a.team_id, a.team_name
for m, u in zip(team["members"], usns):
    m["usn"] = u
json.dump(team, open(p, "w"), indent=2)

reps = [(f"eval_P1_{old['id']}", f"eval_P1_{a.team_id}"), (f"`{old['id']}`", f"`{a.team_id}`"),
        (f"{old['id']} · {old['name']}", f"{a.team_id} · {a.team_name}"), (f"`{old['name']}`", f"`{a.team_name}`")]
reps += list(zip(old["usns"], usns))
for f in [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "Hackathon_Solution_Report_P1.md")] + \
        glob.glob(os.path.join(ROOT, "docs", "*.md")):
    s = open(f).read()
    for x, y in reps:
        s = s.replace(x, y)
    open(f, "w").write(s)
subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "write_eval_md.py")], check=True)
print("team details updated:", a.team_id, a.team_name, usns)
