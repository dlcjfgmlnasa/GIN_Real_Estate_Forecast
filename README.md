# GIN Real Estate Forecast

작성중...

지인 부동산 아파트 실거래가 시세 예측

<br>

## Requirements
- **language** : Python 3.6 (may work with other versions, but I used 3.6) 
- **Database** : Mysql 5.7
- **python library**
  - scikit-learn 0.21.3
  - pandas 0.25.0
  - numpy 1.17.0
  - ...

<br>

## Feature Information

### 1. 매물정보를 활용한 feature

|  No   |                   feature                   |                         Explanation                          |
| :---: | :-----------------------------------------: | :----------------------------------------------------------: |
| **1** |            sale_price_with_floor            | 예측하고자 하는 아파트의 `같은 층`에 있는 과거 매물가격의 평균 |
| **2** |        sale_price_with_floor_recent         | 예측하고자 하는 아파트의 `같은 층`에 있는 과거 매물가격 중 최근 가격 평균 |
| **3** |         sale_price_with_floor_group         | 예측하고자 하는 아파트의 `같은 그룹으로 묶은 층`에 있는 과거 매물가격의 평균 |
| **4** |     sale_price_with_floor_group_recent      | 예측하고자 하는 아파트의 `같은 그룹으로 묶은 층`에 있는 과거 매물 중 최근 가격의 평균 |
| **5** |        sale_price_with_complex_group        | 예측하고자 하는 아파트의 `같은 단지`에 있는 과거 매물가격의 평균 (층 고려) |
| **6** |    sale_price_with_complex_group_recent     | 예측하고자 하는 아파트의 `같은 단지`에 있는 과거 매물 중 최근 가격의 평균 (층 고려) |
| **7** |    sale_price_with_similarity_apt_group     | 예측하고자 하는 아파트의 `비슷한 아파트`에 있는 과거 매물가격의 평균 (층 고려) |
| **8** | sale_price_with_similarity_apt_group_recent | 예측하고자 하는 아파트의 `비슷한 아파트`에 있는 과거 매물 중 최근 가격의 평균 (층 고려) |



### 2. 매매정보를 활용한 feature

|  No   |                   feature                    |                         Explanation                          |
| :---: | :------------------------------------------: | :----------------------------------------------------------: |
| **1** |            trade_price_with_floor            | 예측하고자 하는 아파트의 `같은 층`에 있는 과거 매매가격의 평균 |
| **2** |        trade_price_with_floor_recent         | 예측하고자 하는 아파트의 `같은 층`에 있는 과거 매매가격 중 최근 가격 평균 |
| **3** |         trade_price_with_floor_group         | 예측하고자 하는 아파트의 `같은 그룹으로 묶은 층`에 있는 과거 매매가격의 평균 |
| **4** |     trade_price_with_floor_group_recent      | 예측하고자 하는 아파트의 `같은 그룹으로 묶은 층`에 있는 과거 매매 중 최근 가격의 평균 |
| **5** |        trade_price_with_complex_group        | 예측하고자 하는 아파트의 `같은 단지`에 있는 과거 매매가격의 평균 (층 고려) |
| **6** |    trade_price_with_complex_group_recent     | 예측하고자 하는 아파트의 `같은 단지`에 있는 과거 매매 중 최근 가격의 평균 (층 고려) |
| **7** |    trade_price_with_similarity_apt_group     | 예측하고자 하는 아파트의 `비슷한 아파트`에 있는 과거 매매가격의 평균 (층 고려) |
| **8** | trade_price_with_similarity_apt_group_recent | 예측하고자 하는 아파트의 `비슷한 아파트`에 있는 과거 매매 중 최근 가격의 평균 (층 고려) |



### 3. 거래량 정보를 활용한 feature

|  No   |          feature           |                         Explanation                          |
| :---: | :------------------------: | :----------------------------------------------------------: |
| **1** | trade_volume_standard_area |    지역의 면적별 거래량과 10년 치 면적별 기준 거래량 비율    |
| **2** | trade_volume_standard_year | 지역의 건축년도별 거래량과 10년 치 기준 건축년도 거래량 비율 |

<br>

## Traning Model

- [x] Linear Regression Model 
- [x] Support Vector Model (SVM)
- [ ] Random forest regression 
- [x] Deep Neural Network (DNN)
- [ ] Recurrent Neural Network (RNN)

<br>

## Result

### 1. 10-fold Cross Validation

#### - (매매정보 + 매물정보 + 거래량 정보 feature) 를 활용한 모델 - full.model

|     No      |      MAPE       | success percent (%) | error percent (%) |
| :---------: | :-------------: | :-----------------: | :---------------: |
|    **1**    |   3.936291126   |     80.08234689     |    19.91765311    |
|    **2**    |   3.97111797    |     79.81214617     |    20.18785383    |
|    **3**    |   3.92099672    |     80.55841482     |    19.44158518    |
|    **4**    |   3.841401839   |     81.00874936     |    18.99125064    |
|    **5**    |   3.918456153   |     80.48121462     |    19.51878538    |
|    **6**    |   3.913764077   |     80.2624807      |    19.7375193     |
|    **7**    |   3.982324762   |     80.09521359     |    19.90478641    |
|    **8**    |   3.867020916   |     81.03204221     |    18.96795779    |
|    **9**    |   3.932863318   |     80.0540471      |    19.9459529     |
|   **10**    |   4.004123346   |     79.8738901      |    20.1261099     |
| **average** | **3.928836023** |   **80.32605455**   |  **19.67394545**  |

<br>

#### - (매물정보 + 거래량 정보 feature) 를 활용한 모델 - sale.model

|     No      |      MAPE       | success percent (%) | error percent (%) |
| :---------: | :-------------: | :-----------------: | :---------------: |
|    **1**    |   4.23777369    |     78.99728997     |    21.00271003    |
|    **2**    |   4.364677929   |     78.86178862     |    21.13821138    |
|    **3**    |   4.392633108   |     80.08130081     |    19.91869919    |
|    **4**    |   4.661950785   |     76.82926829     |    23.17073171    |
|    **5**    |   4.377170144   |     77.37127371     |    22.62872629    |
|    **6**    |   4.645748809   |     78.59078591     |    21.40921409    |
|    **7**    |   4.589261073   |     78.15468114     |    21.84531886    |
|    **8**    |   4.433719078   |     78.69742198     |    21.30257802    |
|    **9**    |   4.535485842   |     77.6119403      |    22.3880597     |
|   **10**    |   4.483332198   |     79.64721845     |    20.35278155    |
| **average** | **4.472175266** |   **78.48429692**   |  **21.51570308**  |

<br>

#### - (매매정보 + 거래량 정보 feature) 를 활용한 모델 - trade.model

|     No      |    MAPE     | success percent (%) | error percent (%) |
| :---------: | :---------: | :-----------------: | :---------------: |
|    **1**    | 5.072981459 |     71.29850614     |    28.70149386    |
|    **2**    | 5.075177423 |     71.26204367     |    28.73795633    |
|    **3**    | 5.07349034  |     71.39131972     |    28.60868028    |
|    **4**    | 5.059204558 |     71.39573941     |    28.60426059    |
|    **5**    | 5.082268545 |     71.26314859     |    28.73685141    |
|    **6**    | 5.032228301 |     71.6111995      |    28.3888005     |
|    **7**    | 5.059755917 |     71.52059577     |    28.47940423    |
|    **8**    | 5.073535406 |     71.28414214     |    28.71585786    |
|    **9**    | 5.093585477 |     71.27056562     |    28.72943438    |
|   **10**    | 5.058044208 |     71.37663945     |    28.62336055    |
| **average** | 5.068027163 |      71.36739       |     28.63261      |

<br>



### 2. Visualization

- **[apt_detail_pk 1 Apt]** predicate (with  **visualization**)

  ![pk1 predicate](https://github.com/dlcjfgmlnasa/GIN_Real_Estate_Forecast/blob/master/result/img/1.png?raw=true)

- **[apt_detail_pk 2 Apt]** predicate (with  **visualization**)

![pk2 predicate](https://github.com/dlcjfgmlnasa/GIN_Real_Estate_Forecast/blob/master/result/img/2.png?raw=true)

- **[apt_detail_pk 12 Apt]** predicate (with  **visualization**)

![pk12 predicate](https://github.com/dlcjfgmlnasa/GIN_Real_Estate_Forecast/blob/master/result/img/12.png?raw=true)

- **[apt_detail_pk 13 Apt]** predicate (with  **visualization**)

![pk13 predicate](https://github.com/dlcjfgmlnasa/GIN_Real_Estate_Forecast/blob/master/result/img/13.png?raw=true)

- **[apt_detail_pk 1024 Apt]** predicate (with  **visualization**)

![pk1024 predicate](https://github.com/dlcjfgmlnasa/GIN_Real_Estate_Forecast/blob/master/result/img/1024.png?raw=true)

<br>

## How to Using

### train.py

- machine learning model traning



---

<br>

### test.py

- machine learning model testing

```cmd

```



---

<br>

### data_helper.py

```cmd
usage: data_helper.py [-h] [--calc_similarity_apt] [--make_dataset]
                      [--correlation] [--features FEATURES]
                      [--sale_month_size SALE_MONTH_SIZE]
                      [--sale_recent_month_size SALE_RECENT_MONTH_SIZE]
                      [--trade_month_size TRADE_MONTH_SIZE]
                      [--trade_recent_month_size TRADE_RECENT_MONTH_SIZE]
                      [--trade_cd {t,c}] [--similarity_size SIMILARITY_SIZE]
                      [--save_path SAVE_PATH]
                      [--correlation_path CORRELATION_PATH]
                      [--label_name LABEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --calc_similarity_apt
                        APT similarity 계산
  --make_dataset        데이터셋 생성
  --correlation         correlation analysis
  --features FEATURES   예측에 필요한 feature (default: setting.py에 있는 features 참조)
  --sale_month_size SALE_MONTH_SIZE
                        예측시 사용될 매물 데이터 크기 (default: setting.py에 있는
                        sale_month_size 참조)
  --sale_recent_month_size SALE_RECENT_MONTH_SIZE
                        예측시 사용될 매물 데이터 크기 (default: setting.py에 있는
                        sale_month_size 참조)
  --trade_month_size TRADE_MONTH_SIZE
                        예측시 사용될 매매 데이터 크기 (default: setting.py에 있는
                        trade_month_size 참조)
  --trade_recent_month_size TRADE_RECENT_MONTH_SIZE
                        예측시 사용될 최근 매매 데이터 크기 (default: setting.py에 있는
                        trade_recent_month_size 참조)
  --trade_cd {t,c}      t : 아파트 매매가격 추정 / r: 아파트 전월세가격 추정
  --similarity_size SIMILARITY_SIZE
                        비슷한 아파트 리스트 출력 갯수 (default: setting.py에 있는
                        similarity_size 참조
  --save_path SAVE_PATH
                        DATASET PATH (default: setting.py에 있는 save_path 참조)
  --correlation_path CORRELATION_PATH
                        correlation analysis result DATA PATH (default:
                        setting.py에 있는 correlation_path 참조)
  --label_name LABEL_NAME
                        DATASET label name (default: setting.py에 있는 label_name
                        참조)
```



#### 1. Train dataset 생성

```cmd
=> python data_helper.py --make_dataset
```



#### 2. Correlation

```cmd
=> python data_helper.py --make_dataset
```



#### 3. Calculation Similarity APT

```cmd
=> python data_helper.py --calc_similarity_apt
```





---

<br>

### predicate.py

```cmd
python predicate --help
```

```cmd
usage: predicate.py [-h] [--full_pk] [--full_date] [--db_inject]
                    [--evaluation] [--evaluation_plot]
                    [--apt_detail_pk APT_DETAIL_PK] [--date DATE]
                    [--feature_list FEATURE_LIST]
                    [--sale_month_size SALE_MONTH_SIZE]
                    [--sale_recent_month_size SALE_RECENT_MONTH_SIZE]
                    [--trade_month_size TRADE_MONTH_SIZE]
                    [--trade_recent_month_size TRADE_RECENT_MONTH_SIZE]
                    [--model_info MODEL_INFO]
                    [--previous_month_size PREVIOUS_MONTH_SIZE]
                    [--feature_engine {default,optimizer}] [--trade_cd {t,r}]

optional arguments:
  -h, --help            show this help message and exit
  --full_pk             대상 아파트 전체 예측
  --full_date           2006년도부터 현재까지 예측
  --db_inject           mysql database injection
  --evaluation          대상 아파트 정확도 평가
  --evaluation_plot     대상 아파트 정확도 시각화 및 저장 (setting.py에 있는 image_path 값을 참조하여
                        저장)
  --apt_detail_pk APT_DETAIL_PK
                        예측 대상 아파트 pk
  --date DATE           예측하고 싶은 날짜 ex) 2018-01-01 (default : 현재 날짜)
  --feature_list FEATURE_LIST
                        예측에 필요한 feature (default: setting.py에 있는 features 참조)
  --sale_month_size SALE_MONTH_SIZE
                        예측시 사용될 매물 데이터 크기 (default: setting.py에 있는
                        sale_month_size 참조)
  --sale_recent_month_size SALE_RECENT_MONTH_SIZE
                        예측시 사용될 최근 매물 데이터 크기 (default: setting.py에 있는
                        sale_recent_month_size 참조
  --trade_month_size TRADE_MONTH_SIZE
                        예측시 사용될 매매 데이터 크기 (default: setting.py에 있는
                        trade_month_size 참조)
  --trade_recent_month_size TRADE_RECENT_MONTH_SIZE
                        예측시 사용될 최근 매매 데이터 크기 (default: setting.py에 있는
                        trade_recent_month_size 참조)
  --model_info MODEL_INFO
                        모델 위치 정보 (default: setting.py에 있는 model_info)
  --previous_month_size PREVIOUS_MONTH_SIZE
                        예측시 사용하는 과거 매물&매매 사이즈 (default: setting.py에 있는
                        predicate_previous_month_size
  --feature_engine {default,optimizer}
                        feature engineering 을 하는 방법
  --trade_cd {t,r}      t : 아파트 매매가격 추정 / r: 아파트 전월세가격 추정
```



#### 1. 2006년도 부터 현재까지 전체 아파트 시세 예측

```cmd
=> python predicate.py --full_pk --full_date
```

- **option** : *--db_inject* 를 추가하면 mysql에 예측된 결과 저장

  

#### 2. 지정된 날짜에 대해서 전체 아파트 시세 예측

```cmd
=> python predicate.py --full_pk
=> python predicate.py --full_pk --date={날짜}
```

- **example**
  
  1. `python predicate.py --full_pk`  : 현재 날짜에 예측 대상 아파트 시세 예측
2. `python predicate.py --full_pk --date=2018-01-01 ` : 2018-01-01 예측 대상 아파트 시세 예측

  

- **option** : *--db_inject* 를 추가하면 mysql에 예측된 결과 저장



#### 3. 2006년도 부터 현재까지 [apt_detail_pk] 시세 예측

```cmd
=> python predicate.py --full_date --apt_detail_pk={아파트 pk}
```

- **example**

  1. `python predicate.py --full_date --apt_detail_pk=1`  : 2006년도 부터 현재까지 [아파트 pk 1] 시세 예측

  

- **option** : *--db_inject* 를 추가하면 mysql에 예측된 결과 저장



#### 4. 지정된 날짜에 대해서 [apt_detail_pk] 시세 예측

```cmd
=> python predicate.py --apt_detail_pk={아파트 pk}
=> python predicate.py --apt_detail_pk={아파트 pk} --date={날짜}
```

- **example**
  1. `python predicate.py --apt_detail_pk=1` : [아파트 pk 1]  
  2. `python predicate.py --apt_detail_pk=1 --date=2018-01-01`: [아파트 pk 1]  예측 정확도 측정 및 그래프 출력



#### 5. [apt_detail_pk] 정확도 평가

```cmd
=> python predicate.py --evaluation --apt_detail_pk={아파트 pk}
=> python predicate.py --evaluation --evaluation_plot --apt_detail_pk={아파트 pk}
```

- **example**
  1. `python predicate.py --evaluation --apt_detail_pk=1` : [아파트 pk 1]  예측 정확도 측정
  2. `python predicate.py --evaluation --evaluation_plot --apt_detail_pk=1` : [아파트 pk 1]  예측 정확도 측정 및 그래프 출력

