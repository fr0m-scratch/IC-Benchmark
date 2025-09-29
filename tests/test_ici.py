import json, os, subprocess, sys, tempfile

def test_extract_and_score():
    # tiny schema inline
    tools = [{
        "tool_id": "toy.simple",
        "title": "Toy",
        "description": "Simple tool.",
        "input_schema": {
            "type":"object",
            "properties":{
                "x":{"type":"string"},
                "p":{"type":"string","enum":["a","b","c"]},
                "z":{"type":"object","properties":{"k":{"type":"integer","minimum":0}}}
            },
            "required":["x"]
        }
    }]
    with tempfile.TemporaryDirectory() as td:
        tpath = os.path.join(td,"tools.jsonl")
        fpath = os.path.join(td,"feat.jsonl")
        spath = os.path.join(td,"scored.jsonl")
        with open(tpath,"w",encoding="utf-8") as f:
            for r in tools: f.write(json.dumps(r)+"\n")
        subprocess.check_call([sys.executable,"ici/extract.py","--tools",tpath,"--out",fpath])
        subprocess.check_call([sys.executable,"ici/score.py","--feat",fpath,"--out",spath])
        rows = [json.loads(x) for x in open(spath)]
        assert "ici" in rows[0]
        assert rows[0]["num_params"] >= 2
        assert rows[0]["depth"] >= 1
