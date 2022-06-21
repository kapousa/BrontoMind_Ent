#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import numpy
import pandas as pd

from bm.db_helper.DBConnector import DBConnector

cc = DBConnector()

crsour = cc.create_mysql_connection('us-cdbr-east-04.cleardb.com', 'bb28d2412194b2', '96616834', 'heroku_f01b6802cd615be')
#s_q_l = "SELECT table_name  FROM INFORMATION_SCHEMA.TABLES WHERE (TABLE_SCHEMA = 'heroku_f01b6802cd615be')"
s_q_l_C = "SELECT column_name  FROM INFORMATION_SCHEMA.COLUMNS WHERE (TABLE_NAME = 'diabetes')"
s_q_l = "SELECT *  FROM diabetes"
mycursor = crsour.cursor()
mycursor.execute(s_q_l_C)
column_names = numpy.array(mycursor.fetchall()).flatten()
mycursor.execute(s_q_l)
df = pd.DataFrame(mycursor.fetchall(), columns=column_names)
df.to_csv (r'exported_data.csv', index = False)
mycursor.close()

#myresult = numpy.array(mycursor.fetchall())

#myresult = myresult.flatten()

#mycursor.close()

#for x in range (len(myresult)):
 # print(myresult[x])