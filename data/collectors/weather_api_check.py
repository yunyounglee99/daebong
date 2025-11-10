import requests
import json

url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
params ={'serviceKey' : '17f766b554e3e4bd767ecb2913a2450783b7e243995f928e2d28532c672550f4',
        'pageNo' : '1', 
        'numOfRows' : '10', 
        'dataType' : 'JSON', 
        'dataCd' : 'ASOS', 
        'dateCd' : 'DAY', 
        'startDt' : '20250526', 
        'endDt' : '20250714', 
        'stnIds' : '108' }

response = requests.get(url, params=params)
print(response.content)

file_path = '/Users/nyoung/Desktop/dev/project/daebong/data/collectors/sample.json'
with open(file_path, 'w') as f:
    json.dump(response.content, f)
