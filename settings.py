# -*- coding:utf-8 -*-
import os
import datetime
import mysql.connector
from dotenv import load_dotenv

# env path
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# db connections
cnx = mysql.connector.MySQLConnection(
    user=os.getenv('db_user'),
    password=os.getenv('db_password'),
    host=os.getenv('db_host'),
    port=os.getenv('db_port'),
    database=os.getenv('db_name')
)
db_cursor = cnx.cursor(buffered=True)

# Parameter
sale_features = [
    # 매물 데이터를 이용한 feature
    'sale_price_with_floor',
    'sale_price_with_floor_recent',
    'sale_price_with_floor_group',
    'sale_price_with_floor_group_recent',
    'sale_price_with_complex_group',
    'sale_price_with_complex_group_recent',
    # 'sale_price_with_similarity_apt_group',
    # 'sale_price_with_similarity_apt_group_recent'
]

trade_features = [
    # 매매 데이터를 이용한 feature
    'trade_price_with_floor',
    'trade_price_with_floor_recent',
    'trade_price_with_floor_group',
    'trade_price_with_floor_group_recent',
    'trade_price_with_complex_group',
    'trade_price_with_complex_group_recent'
    # 'trade_price_with_similarity_apt_group',
    # 'trade_price_with_similarity_apt_group_recent'
]

training_volume_feature = [
    # 거래량을 사용하는 feature
    'training_volume_standard_area',
    'training_volume_standard_year'
]

# total_feature
features = sale_features + trade_features + training_volume_feature

dataset_pk_size = 10000

sale_month_size = 6
sale_recent_month_size = 2
trade_month_size = 6
trade_recent_month_size = 2

similarity_size = 10

# 2. model information parameter
save_path = os.path.join('./dataset', 'apt_dataset.csv')
model_path = os.path.join('./model', 'store', 'linear_regression.model')
model_type = 'linear_regression'
trade_cd = 't'
label_name = 'price'

# 3. Test Parameter
n_fold = 10
plot_flag = False
test_result_path = os.path.join('./result', 'linear_regression', 'test01.xlsx')

# 4. Train Parameter

# 5. Predicate Parameter
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
predicate_previous_month_size = 5

# 6. Correlation
correlation_path = os.path.join('./result', 'correlation.csv')
