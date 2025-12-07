import urllib.request, urllib.parse

def post(url, fields):
    data = urllib.parse.urlencode(fields).encode('ascii')
    req = urllib.request.Request(url, data=data, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read(500).decode('utf-8', errors='ignore')
        print(resp.status, body.replace('\n',' ')[:160])

url = 'http://127.0.0.1:5001/predict'
examples = [
    {'Nitrogen':'90','Phosporus':'42','Potassium':'43','Temperature':'25.5','Humidity':'65.2','pH':'6.5','Rainfall':'120'},
    {'Nitrogen':'10','Phosporus':'20','Potassium':'30','Temperature':'18.0','Humidity':'55.0','pH':'7.0','Rainfall':'200'},
    {'Nitrogen':'120','Phosporus':'60','Potassium':'40','Temperature':'30.0','Humidity':'70.0','pH':'5.8','Rainfall':'80'},
]
for i,e in enumerate(examples,1):
    try:
        print('EX', i, end=': ')
        post(url, e)
    except Exception as ex:
        print('ERR', type(ex).__name__, ex)
