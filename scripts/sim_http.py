import argparse
import sys
import requests
import concurrent.futures
import time

SERVER = "http://127.0.0.1:5000"


def send(payload):
    try:
        r = requests.post(f"{SERVER}/api/send_traffic", json=payload, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="HTTP traffic simulator for IDS")
    parser.add_argument("-n", type=int, default=1, help="Number of concurrent requests (default: 1)")
    args = parser.parse_args()

    if args.n <= 0:
        print("error: -n must be >= 1")
        sys.exit(1)

    if args.n <= 5:
        for i in range(5):
            data = send({"type": "normal"})
            status, body = data
            if status == 200:
                print(f"[NORMAL] OK  prediction={body['predicted_label']}  confidence={body['confidence']:.4f}")
            else:
                print(f"[NORMAL] FAIL  {body}")
            time.sleep(2)
    else:
        payload = {"type": "attack"}
        print(f"[DoS] launching {args.n} concurrent requests ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.n, 200)) as ex:
            futures = [ex.submit(send, payload) for _ in range(args.n)]
            done = [f.result() for f in concurrent.futures.as_completed(futures)]

        ok = sum(1 for s, _ in done if s == 200)
        print(f"[DoS] {ok}/{len(done)} succeeded")
        for status, body in done[:5]:
            label = body.get("predicted_label", "?") if isinstance(body, dict) else body
            print(f"  -> {status}  {label}")


if __name__ == "__main__":
    main()
