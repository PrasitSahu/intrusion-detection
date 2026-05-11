import requests

SERVER = "http://127.0.0.1:5000"

r = requests.post(f"{SERVER}/api/send_traffic", json={"category": "r2l"}, timeout=10)
data = r.json()
if r.status_code == 200:
    print(f"[R2L] OK  prediction={data['predicted_label']}  attack_type={data.get('attack_type','')}  confidence={data['confidence']:.4f}")
else:
    print(f"[R2L] FAIL  {data}")
