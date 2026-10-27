import requests, time

dates = ['20260808', '20260810', '20260818', '20260901', '20260907']
for ds in dates:
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?response=json&date={ds}&selectType=ALL"
    t0 = time.time()
    try:
        r = requests.get(url, timeout=20)
        elapsed = time.time() - t0
        j = r.json()
        print(f"{ds}: status={r.status_code} stat={j.get('stat')!r} rows={len(j.get('data', []))} elapsed={elapsed:.2f}s", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{ds}: EXCEPTION {type(e).__name__}: {e} elapsed={elapsed:.2f}s", flush=True)
    time.sleep(0.8)
