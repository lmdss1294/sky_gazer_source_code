from pymongo import MongoClient
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

import time
service = Service(ChromeDriverManager().install())
import os

download_directory = os.path.join(os.getcwd(), "date")




# 연결 문자열 설정
username = "lmdss1294"
password = "########"
cluster_url = "cluster0.jktknik.mongodb.net"
database_name = "cluster0"
collection_name = "your_collection_name"
# MongoClient를 사용하여 연결
client = MongoClient(f"mongodb+srv://{username}:{password}@{cluster_url}/{database_name}?retryWrites=true&w=majority")

# 데이터베이스 선택
db = client[database_name]
collection = db[collection_name]
collection_name2 = "your_collection_name2"
collection2 = db[collection_name2]
# 연결 확인
print(client.server_info())



key = "##########"

from datetime import datetime, timedelta
import pytz
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import os
import glob
from datetime import datetime
import pytz
import time
with open("random_search.pkl", "rb") as file:
        model2 = pickle.load(file)

while True:
    result = collection.delete_many({})
    result = collection2.delete_many({})
    korea = pytz.timezone('Asia/Seoul')
    datetime_korea = datetime.now(korea) #- timedelta(hours=2)
    base_date = str(datetime_korea.year) +datetime_korea.strftime('%m')  +str(datetime_korea.day)
    base_time =  datetime_korea.strftime('%H') +'00'
    fcstDate = str((datetime.now(korea)+ timedelta(days=5)).year)+(datetime.now(korea)+ timedelta(days=5)).strftime('%m')+(datetime.now(korea)+ timedelta(days=5)).strftime('%d')
    nx = '52'
    ny = '38'
    url = f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?serviceKey={key}&numOfRows=3000&pageNo=1&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}&fcstDate={fcstDate}&fcstDate={base_time}"
    response = requests.get(url)
    xml_data = response.text
    root = ET.fromstring(xml_data)
    # 각 요소를 담을 리스트 초기화
    baseDates = []
    baseTimes = []
    categories = []
    fcstDates = []
    fcstValues = []
    fcstTime = []
    # XML 데이터를 순회하며 정보 추출
    for item in root.findall(".//item"):
        baseDates.append(item.find("baseDate").text)
        baseTimes.append(item.find("baseTime").text)
        categories.append(item.find("category").text)
        fcstDates.append(item.find("fcstDate").text)
        fcstValues.append(item.find("fcstValue").text)
        fcstTime.append(item.find("fcstTime").text)

    # 데이터프레임 생성
    df = pd.DataFrame(
        {
            "baseDate": baseDates,
            "baseTime": baseTimes,
            "category": categories,
            "fcstDate": fcstDates,
            "fsctTime": fcstTime,
            "fcstValue": fcstValues
            
        }
    )
    categories_to_remove = ['SKY', 'WAV', 'TMN', 'TMX', 'REH', 'UUU', 'VVV']
    df = df[~df['category'].isin(categories_to_remove)]
    df = df.reset_index(drop=True)
    df['Datetime'] = pd.to_datetime(df['fcstDate'].astype(str) + df['fsctTime'].astype(str), format='%Y%m%d%H%M')
    L = []
    for i in range(int(len(df)/7)):
        if len(df)%7 != 0:
            print('ERROR')
        else:
            A = df[['fcstValue','category','Datetime']].loc[7*i:7*i+6]
            b = []
            b.append(A['Datetime'].iloc[0])
            for j in range(7):
                b.append(A['fcstValue'].iloc[j])
            
        L.append(b)
    
    df2 = pd.DataFrame(L, columns=['Datetime', '기온(°C)', '풍향(deg)', '풍속(KT)', '강수형태', '강수확률', '강수량(mm)', '적설량(mm)'])
    df2 = df2[['Datetime', '기온(°C)', '풍향(deg)', '풍속(KT)','강수량(mm)', '적설량(mm)']]
    df2['강수량(mm)'] = df2['강수량(mm)'].replace('강수없음', 0)
    df2['적설량(mm)'] = df2['적설량(mm)'].replace('적설없음', 0)
    df2['풍속(KT)'] = pd.to_numeric(df2['풍속(KT)'])
    df2['풍속(KT)'] = 1.95 * df2['풍속(KT)'] 

    df3 = df2[['Datetime','적설량(mm)']]
    df3.name = '적설량(mm)'
    df4 = df2.drop('적설량(mm)',axis=1)
    df4 = df4.reindex(columns=['풍향(deg)','풍속(KT)','기온(°C)','강수량(mm)'])
    df4['풍향(deg)'] = pd.to_numeric(df4['풍향(deg)'])
    df4['풍속(KT)'] = pd.to_numeric(df4['풍속(KT)'])
    df4['기온(°C)'] = pd.to_numeric(df4['기온(°C)'])

    df4['강수량(mm)'] = df4['강수량(mm)'].str.replace(r'mm$', '', regex=True)
    df4['강수량(mm)'] = df4['강수량(mm)'].fillna(0)
    df4['강수량(mm)'] = pd.to_numeric(df4['강수량(mm)'])
    print('*****')
    non_null_values = df4[df4['강수량(mm)'].notnull()]['강수량(mm)']
    print(non_null_values)
    print('*****')
    with open('random_search_model_visibility.pkl', 'rb') as file:
        model = pickle.load(file)
    y = model.predict(df4)
    df4['시정'] = y
    df2 = pd.concat([df4, df3], axis=1)
    print(df2)





    
    chrome_options = Options()
    #chrome_options.add_argument("headless")
    chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_directory
})
    driver = webdriver.Chrome(service = service,options=chrome_options)
    # 제주국제공항 운항정보 페이지로 이동합니다.
    driver.get('https://www.airport.co.kr/jeju/cms/frCon/index.do?MENU_ID=40')
    print("접속중")
    time.sleep(10)
    print("사이트 로딩 대기중")

    # JavaScript 함수 'excelDown()'를 실행합니다.
    driver.execute_script("excelDown();")
    print("다운중")
    time.sleep(10)
    # 작업이 끝나면 웹드라이버를 종료합니다.
    driver.quit()

    
    korea_timezone = pytz.timezone('Asia/Seoul')

    cttime = datetime.now(korea_timezone)

    current_datetime_in_korea = cttime.strftime('%Y%m%d')
    day_of_week = cttime.weekday()

    data_name = 'SCHEDULE_DOM_'+current_datetime_in_korea +'.xlsx'
    df = pd.read_excel(f"date\{data_name}")
    df = df.drop(df.index[0])
    df = df.rename(columns={
        'Unnamed: 8': '1',
        'Unnamed: 9': '2',
        'Unnamed: 10': '3',
        'Unnamed: 11': '4',
        'Unnamed: 12': '5',
        'Unnamed: 13': '6',
        '운항요일': '0'
    })
    df = df.reset_index(drop = True)
    df['출발 시간'] = df['출발 시간'].str.replace(":", "")
    print(df['출발 시간'])
    next_three_days = [str((day_of_week + i) % 7) for i in range(4)]
    L = []
    for i in range(len(df)):
        for j in next_three_days:
            if df[str(j)].loc[i] =="Y":
                if df['운항구분'].loc[i] =='출발':
                    time = (cttime + timedelta(days = (int(j) - day_of_week) % 7)).strftime('%Y%m%d')
                    time = time + df['출발 시간'].loc[i]
                    flight_info = df[['운항구분', '항공사', '편명', '공항', '운항구간']].loc[i].to_list()
                    flight_info.append(time)
                    L.append(flight_info)
    df_new = pd.DataFrame(L)
    df_new.columns = ['출발','비행사','항공기코드','출발','도착','날짜']
    df_new = df_new.reset_index(drop=True)
    print('------')

#with open('random_search_predict.pkl', 'rb') as file:
        #model = pickle.load(file)

    df_new['날짜'] = pd.to_datetime(df_new['날짜'])
    df_new['시정'] = None
    df_new['풍향(deg)'] = None
    df_new['풍속(KT)'] = None
    df_new['기온(°C)'] = None
    df_new['적설량(mm)'] = None
    df_new['강수량(mm)'] = None   
    df_new['결항여부'] = '모름'

    for i in range(len(df_new)):
            input_datetime=    df_new['날짜'].loc[i]
            print(input_datetime)
            previous_datetime = input_datetime.replace(minute=0)
            print(previous_datetime)
            next_datetime = previous_datetime + timedelta(hours=1)
            print(next_datetime)

            ratio = ((input_datetime - previous_datetime).total_seconds()) / ((next_datetime - previous_datetime).total_seconds())
            print(ratio)

            a = df2[df2['Datetime'] == previous_datetime]
            b = df2[df2['Datetime'] == next_datetime]
            
            if a.empty or b.empty:
                continue
            else:
                df_new.at[i, '시정'] = ratio*(b['시정'].values[0]-a['시정'].values[0]) + a['시정'].values[0]
                df_new.at[i, '풍향(deg)'] = ratio*(b['풍향(deg)'].values[0]-a['풍향(deg)'].values[0]) + a['풍향(deg)'].values[0]
                df_new.at[i, '풍속(KT)'] = ratio*(b['풍속(KT)'].values[0]-a['풍속(KT)'].values[0]) + a['풍속(KT)'].values[0]
                df_new.at[i, '기온(°C)'] = ratio*(b['기온(°C)'].values[0]-a['기온(°C)'].values[0]) + a['기온(°C)'].values[0]
                df_new.at[i, '적설량(mm)'] = ratio*(b['적설량(mm)'].values[0]-a['적설량(mm)'].values[0]) + a['적설량(mm)'].values[0]
                df_new.at[i, '강수량(mm)'] = ratio*(b['강수량(mm)'].values[0]-a['강수량(mm)'].values[0]) + a['강수량(mm)'].values[0]
                L = list(df_new[['풍향(deg)','풍속(KT)','시정','기온(°C)','강수량(mm)','적설량(mm)']].loc[i].values)
                print(L)
                y = model2.predict([L])
                df_new.at[i,'결항여부'] = str(y[0])

    df_new = df_new.sort_values('날짜')
                



    records = df_new.to_dict(orient='records')
    collection2.insert_many(records)
    print(df_new)

    # 업로드 확인
    print("데이터프레임이 몽고DB에 성공적으로 업로드되었습니다.")
    

    records1 = df2.to_dict(orient='records')
    collection.insert_many(records1)
    print("데이터프레임이 몽고DB에 성공적으로 업로드되었습니다.")
    import time

    time.sleep(7200)



print(df2)
            


