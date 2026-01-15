import sqlite3
import csv

DB_PATH = r"onto_db/sample_db.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS ;")  
conn.commit()
conn.close()