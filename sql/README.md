# SQL

`python train.py` writes two files here using the real column names of the dataset:

- `schema.sql` - `CREATE TABLE` statement for the solar + weather data
- `analysis_queries.sql` - hourly / monthly averages, top-10 records, producing-vs-idle comparison, daily totals

Load the CSV into SQLite and run them, for example:

```python
import sqlite3, pandas as pd
con = sqlite3.connect("data/solar.db")
con.executescript(open("sql/schema.sql").read())
pd.read_csv("data/your_file.csv").to_sql("solar_weather", con, if_exists="append", index=False)
print(pd.read_sql(open("sql/analysis_queries.sql").read().split(";")[1], con))
```
