import urllib.request

for path in ['/', '/predict', '/recommend']:
    try:
        with urllib.request.urlopen('http://127.0.0.1:5000'+path, timeout=8) as resp:
            print(path, resp.status)
    except Exception as ex:
        print(path, 'ERR', type(ex).__name__, ex)
