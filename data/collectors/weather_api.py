"""
기상청 API Hub - 시간자료를 CSV로 변환하여 저장
완전히 수정된 버전 (if문 올바르게 작성)
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

class WeatherDataCollector:
    def __init__(self, auth_key: str):
        self.auth_key = auth_key
        self.markets = {
            '서울_가락': '108',
            '안동': '136',
            '광주': '156'
        }
    
    def fetch_and_parse_hourly(self, station_id: str, date: str):
        """
        시간자료를 텍스트로 받아서 CSV로 파싱
        
        Args:
            station_id: 관측소 코드
            date: 날짜 (YYYYMMDD, HH 없으면 현재시각 기준)
        """
        url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm={date}&stn={station_id}&help=0&authKey={self.auth_key}"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            # EUC-KR 디코딩
            text = response.content.decode('euc-kr')
            
            # 데이터 라인 추출
            lines = text.split('\n')
            data_lines = []
            
            for line in lines:
                # ✅ 올바른 if문: 주석이 아니고 빈 줄이 아니고 헤더도 아닌 줄
                if not line.startswith('#') and line.strip() and not line.startswith('KST'):
                    data_lines.append(line)
            
            # ✅ 올바른 if문: data_lines가 비어있는지 확인
            if len(data_lines) == 0:
                return pd.DataFrame()
            
            # Fixed-width 텍스트를 파싱
            data = []
            for line in data_lines:
                if len(line) < 100:
                    continue
                
                try:
                    # 고정폭으로 각 필드 추출
                    datetime_str = line[0:10].strip()    # YYMMDDHHMI
                    ta = line[63:68].strip()             # TA (기온)
                    hm = line[75:80].strip()             # HM (습도)
                    rn = line[96:102].strip()            # RN (강수량)
                    ws = line[20:24].strip()             # WS (풍속)
                    
                    """
                    if len(data) == 0:
                        print(f"    🔬 First record parsing:")
                        print(f"       datetime_str: '{datetime_str}'")
                        print(f"       ta: '{ta}', hm: '{hm}', rn: '{rn}'")
                    """
                    # ✅ 올바른 if문: datetime_str이 결측값인지 확인
                    if datetime_str == '-9' or not datetime_str:
                        continue
                    
                    # 문자열을 실수로 변환
                    try:
                        ta_val = float(ta) if ta != '-9' and ta else None
                        hm_val = float(hm) if hm != '-9' and hm else None
                        rn_val = float(rn) if rn != '-9' and rn else None
                        ws_val = float(ws) if ws != '-9' and ws else None
                    except ValueError:
                        continue
                    
                    data.append({
                        'datetime': datetime_str,
                        'station_id': station_id,
                        'temperature': ta_val,
                        'humidity': hm_val,
                        'precipitation': rn_val,
                        'wind_speed': ws_val,
                    })
                except Exception:
                    continue
            
            # ✅ 올바른 if문: data가 비어있는지 확인
            if len(data) > 0:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return pd.DataFrame()
    
    def collect_daily_aggregated(self, start_date: str, end_date: str):
        """
        시간자료를 모아서 일별로 집계
        (일자료와 동일한 형식으로 변환)
        """
        print("="*70)
        print(f"[Weather Data Collection - Hourly to Daily Conversion]")
        print(f"  Period: {start_date} ~ {end_date}")
        print("="*70)
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        
        for market_name, station_id in self.markets.items():
            print(f"\n[{market_name}]")
            
            market_daily_data = []
            current_date = start_dt
            
            while current_date <= end_dt:
                date_str = current_date.strftime('%Y%m%d')
                
                print(f"  {date_str}...", end=' ')
                
                # 시간자료 수집
                hourly_df = self.fetch_and_parse_hourly(station_id, date_str)
                
                # ✅ 올바른 if문: hourly_df가 비어있지 않은지 확인
                if len(hourly_df) > 0:
                    # 일별 집계 (평균, 합계 등)
                    daily_record = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'market': market_name,
                        'station_id': station_id,
                        'avg_temp': hourly_df['temperature'].mean(),
                        'max_temp': hourly_df['temperature'].max(),
                        'min_temp': hourly_df['temperature'].min(),
                        'avg_humidity': hourly_df['humidity'].mean(),
                        'precipitation': hourly_df['precipitation'].sum(),
                        'avg_wind_speed': hourly_df['wind_speed'].mean(),
                    }
                    market_daily_data.append(daily_record)
                    print(f"✅ {len(hourly_df)} hourly records")
                else:
                    print(f"⚠️ No data")
                
                current_date += timedelta(days=1)
                time.sleep(0.3)
            
            # ✅ 올바른 if문: market_daily_data가 비어있지 않은지 확인
            if len(market_daily_data) > 0:
                market_df = pd.DataFrame(market_daily_data)
                all_data.append(market_df)
                print(f"  ✅ Aggregated: {len(market_df)} daily records")
        
        # ✅ 올바른 if문: all_data가 비어있지 않은지 확인
        if len(all_data) == 0:
            print("\n❌ No data collected")
            return pd.DataFrame()
        
        # 전체 통합
        result = pd.concat(all_data, ignore_index=True)
        
        # CSV로 저장
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        save_path = f"data/raw/historical_weather_all_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}.csv"
        result.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*70}")
        print(f"✅ DATA SAVED (CSV FORMAT)")
        print(f"   File: {save_path}")
        print(f"   Total: {len(result)} daily records")
        print(f"   Columns: {result.columns.tolist()}")
        print("="*70)
        
        return result


# 실행
if __name__ == "__main__":
    AUTH_KEY = "SNlh8lEdStiZYfJRHXrY3A"
    
    collector = WeatherDataCollector(AUTH_KEY)
    
    # 일별 집계 데이터 수집
    data = collector.collect_daily_aggregated(
        start_date='2025-05-26',
        end_date='2025-07-14'
    )
    
    # ✅ 올바른 if문: data가 비어있지 않은지 확인
    if len(data) > 0:
        print("\n[Sample Data]")
        print(data.head(10))
        print(f"\n[Statistics]")
        print(data.groupby('market')[['avg_temp', 'precipitation']].describe())
    else:
        print("\n❌ No data to display")
