import urllib.request

def ping(url):
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return r.status
    except Exception as ex:
        return f"ERR {type(ex).__name__}: {ex}"

urls = [
  'http://127.0.0.1:5000/',
  'http://127.0.0.1:5000/predict',
  'http://127.0.0.1:5000/recommend',
  'http://127.0.0.1:5001/',
  'http://127.0.0.1:5001/predict'
]
for u in urls:
    print(u, '->', ping(u))
