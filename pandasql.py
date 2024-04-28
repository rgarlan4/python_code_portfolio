import pandas as pd 
from pandasql import sqldf 
from pandasql import load_births

births = load_births()

print(sqldf("select * from births where births > 25000000 limit 5;",locals()))

q = """ select date(date) as DOB, 
            sum(births) as "Total births"
            from births group by date limit 10;
            """
print(sqldf(q,locals()))

print(sqldf(q, locals()))

print(sqldf(q,globals()))

def pysqldf(q):

    return sqldf(q,globals())

print(pysqldf(q))



