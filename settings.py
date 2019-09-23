# -*- coding:utf-8 -*-
import os
from dotenv import load_dotenv
import mysql.connector

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
