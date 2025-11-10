"""
중기예보자료 조회 (fct_medm_reg.php)
- 과거 데이터 가능
- 발표시간(tmfc1, tmfc2)과 발효시간(tmef1, tmef2) 모두 필요
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

class MidTermForecastCollector:
    def __init__(self, auth_key: str):
        self.auth_key = auth_key
        self.markets = {
            '서울': '11B00000',
            '전라북도': '11F10000',
            '경상북도': '11G10000'
        }
        
        # ✅ 중기예보자료 조회 (과거 데이터 가능)
        self.api_url = "https://apihub.kma.go.kr/api/typ01/url/fct_medm_reg.php"
    
    def parse_forecast_line(self, line: str):
        """중기예보 라인 파싱"""
        
        # ✅ 올바른 if문: 라인 유효성 확인
        if not line or line.startswith('#'):
            return None
        
        parts = line.replace('=', '').split('#')
        
        # ✅ 올바른 if문: 파트 개수 확인
        if len(parts) < 10:
            return None
        
        try:
            record = {
                'STN': parts[0].strip(),
                'REG_ID': parts[1].strip(),
                'TM_FC': parts[2].strip(),
                'MAN_FC': parts[3].strip(),
                'TM_EF': parts[4].strip(),
                'MODE': parts[5].strip(),
                'WF': parts[6].strip(),
                'SKY': parts[7].strip(),
                'PRE': parts[8].strip(),
                'CONF': parts[9].strip() if len(parts) > 9 else None,
            }
            return record
        except:
            return None
    
    def fetch_forecast(self, region_code: str, tmfc1: str, tmfc2: str, tmef1: str, tmef2: str):
        """중기예보자료 조회 (발표시간 + 발효시간 모두 지정)"""
        
        # ✅ 올바른 파라미터 (4개 시간 모두 필요!)
        params = {
            'tmfc1': tmfc1,    # 발표시간 시작
            'tmfc2': tmfc2,    # 발표시간 종료
            'tmef1': tmef1,    # 발효시간 시작
            'tmef2': tmef2,    # 발효시간 종료
            'reg': region_code,
            'mode': 0,
            'disp': 0,
            'help': 0,
            'authKey': self.auth_key
        }
        
        try:
            print(f"    📥 Fetching...", end=' ')
            
            response = requests.get(self.api_url, params=params, timeout=30)
            
            print(f"HTTP {response.status_code}: ", end='')
            
            # ✅ 올바른 if문: 상태 코드 확인
            if response.status_code != 200:
                print(f"❌")
                return pd.DataFrame()
            
            print(f"✅ ({len(response.content)} bytes)")
            
            text = response.content.decode('euc-kr')
            lines = text.split('\n')
            
            # ✅ 데이터 라인 파싱
            parsed_records = []
            
            for line in lines:
                record = self.parse_forecast_line(line)
                
                # ✅ 올바른 if문: 파싱 성공 확인
                if record:
                    parsed_records.append(record)
            
            print(f"    ✅ Parsed: {len(parsed_records)} records")
            
            # ✅ 올바른 if문: 레코드 확인
            if len(parsed_records) == 0:
                print(f"    ⚠️ No records")
                return pd.DataFrame()
            
            df = pd.DataFrame(parsed_records)
            print(f"    🔑 Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return pd.DataFrame()
    
    def collect_data(self, start_date: str, end_date: str):
        """
        중기예보자료 조회 (과거 데이터 가능)
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
        """
        
        print("="*80)
        print(f"[중기예보자료 조회 - 과거 데이터 가능]")
        print(f"  API: fct_medm_reg.php")
        print(f"  기간: {start_date} ~ {end_date}")
        print("="*80)
        
        # ✅ 발표시간 (과거 날짜 OK)
        start_dt = datetime.strptime(f"{start_date}00", '%Y-%m-%d%H')
        end_dt = datetime.strptime(f"{end_date}23", '%Y-%m-%d%H')
        
        tmfc1 = start_dt.strftime('%Y%m%d%H')
        tmfc2 = end_dt.strftime('%Y%m%d%H')
        
        # ✅ 발효시간 (12시간 뒤부터 조회)
        tmef1 = (start_dt + timedelta(hours=12)).strftime('%Y%m%d%H')
        tmef2 = (end_dt + timedelta(hours=12)).strftime('%Y%m%d%H')
        
        print(f"\n  📅 발표시간: {tmfc1} ~ {tmfc2}")
        print(f"  📅 발효시간: {tmef1} ~ {tmef2}\n")
        
        all_data = []
        
        for market_name, region_code in self.markets.items():
            print(f"[{market_name}] (Region: {region_code})")
            
            df = self.fetch_forecast(
                region_code,
                tmfc1, tmfc2,  # 발표시간
                tmef1, tmef2   # 발효시간
            )
            
            # ✅ 올바른 if문: 데이터 확인
            if len(df) > 0:
                df['market'] = market_name
                df['region_code'] = region_code
                all_data.append(df)
                print(f"  ✅ {len(df)} records\n")
            else:
                print(f"  ⚠️ No data\n")
            
            time.sleep(0.5)
        
        # ✅ 올바른 if문: 전체 데이터 확인
        if len(all_data) == 0:
            print("\n❌ 데이터 수집 실패")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        save_path = f"data/raw/forecast_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}.csv"
        result.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"{'='*80}")
        print(f"✅ 중기예보자료 조회 완료")
        print(f"   File: {save_path}")
        print(f"   Total: {len(result)} 레코드")
        print(f"   Markets: {', '.join(result['market'].unique())}")
        print("="*80)
        
        return result


# 실행
if __name__ == "__main__":
    AUTH_KEY = "SNlh8lEdStiZYfJRHXrY3A"
    
    collector = MidTermForecastCollector(AUTH_KEY)
    
    # ✅ 과거 데이터 조회 (발표시간 + 발효시간 모두 지정!)
    data = collector.collect_data(
        start_date='2025-05-26',
        end_date='2025-07-14'
    )
    
    if len(data) > 0:
        print("\n✅ 데이터 수집 성공!")
        print(f"\nShape: {data.shape}")
        
        print(f"\n⭐ 주요 필드:")
        for col in ['TM_EF', 'WF', 'SKY', 'PRE', 'CONF']:
            if col in data.columns:
                print(f"  ✅ {col}")
        
        print(f"\n[샘플 (처음 5행)]")
        print(data[['market', 'TM_EF', 'WF', 'SKY', 'CONF']].head())
    else:
        print("\n❌ 데이터 없음")
