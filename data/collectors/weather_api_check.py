"""
기상청 ASOS 일별 기상 데이터 수집 스크립트

공공데이터포털 API를 통해 서울, 안동, 광주 지역의 일별 기상 데이터를 수집합니다.
- 수집 데이터: 일자, 지역정보, 기온, 습도, 강수량, 풍속 등
- 저장 형식: JSON 파일 (data/collectors/weather_data.json)
- API: 기상청 ASOS 일별 기상정보 조회 서비스

사용법:
    python data/collectors/weather_api_check.py
"""
import requests
import json

url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
stn_list = [
    ('108', '서울'),
    ('136', '안동'),
    ('156', '광주')
]

target_keys = [
    'tm',     # 일자
    'stnId',     # 지역번호
    'stnNm',     # 지역이름
    'avgTs',     # 평균지면온도
    'avgRhm',     # 평균상대습도
    'minTa',     # 최저기온
    'maxTa',     # 최고기온
    'avgTa',     # 평균기온
    'sumRn',     # 일강수량
    'avgWs'     # 평균풍속
]

weather_data = []

for stn_id, stn_name in stn_list:
    params ={'serviceKey' : '17f766b554e3e4bd767ecb2913a2450783b7e243995f928e2d28532c672550f4',
            'pageNo' : '1', 
            'numOfRows' : '50', 
            'dataType' : 'JSON', 
            'dataCd' : 'ASOS', 
            'dateCd' : 'DAY', 
            'startDt' : '20250526', 
            'endDt' : '20250714', 
            'stnIds' : stn_id }

    response = requests.get(url, params=params)
    data = response.json()

    items = data['response']['body']['items']['item']
    for item in items:
        filtered = {key: item.get(key, None) for key in target_keys}
        weather_data.append(filtered)

with open('data/collectors/weather_data.json', 'w', encoding = 'utf-8') as f:
    json.dump(weather_data, f, ensure_ascii=False, indent=4)