import urllib.request, urllib.parse
import re

def post(url, fields):
    data = urllib.parse.urlencode(fields).encode('ascii')
    req = urllib.request.Request(url, data=data, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read(2000).decode('utf-8', errors='ignore')
        m = re.search(r"Recommended\s*Crop\s*:?\s*<[^>]*>([^<]+)</", body, re.I)
        crop = m.group(1).strip() if m else None
        return resp.status, crop or body[:160]

def get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read(200).decode('utf-8', errors='ignore')[:160]

root_status, root_snip = get('http://127.0.0.1:5001/')
print('GET / ->', root_status, root_snip.replace('\n',' '))

url = 'http://127.0.0.1:5001/predict'
examples = [
    {'name':'typical','Nitrogen':'90','Phosporus':'42','Potassium':'43','Temperature':'25.5','Humidity':'65.2','pH':'6.5','Rainfall':'120'},
    {'name':'cool_wet','Nitrogen':'10','Phosporus':'20','Potassium':'30','Temperature':'18.0','Humidity':'80.0','pH':'6.8','Rainfall':'220'},
    {'name':'hot_dry','Nitrogen':'120','Phosporus':'60','Potassium':'40','Temperature':'33.0','Humidity':'40.0','pH':'5.8','Rainfall':'60'},
    {'name':'edge_low','Nitrogen':'0','Phosporus':'0','Potassium':'0','Temperature':'10.0','Humidity':'20.0','pH':'4.5','Rainfall':'0'},
    {'name':'edge_high','Nitrogen':'200','Phosporus':'200','Potassium':'200','Temperature':'45.0','Humidity':'95.0','pH':'9.0','Rainfall':'400'}
]

for e in examples:
    try:
        status, out = post(url, {k:v for k,v in e.items() if k!='name'})
        print(f"POST /predict [{e['name']}] ->", status, out)
    except Exception as ex:
        print(f"POST /predict [{e['name']}] ERR:", type(ex).__name__, ex)
