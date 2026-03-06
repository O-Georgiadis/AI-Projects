import sqlite3
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "prices.db")

def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)'
        )
        cursor.execute('INSERT OR REPLACE INTO prices (city, price) VALUES (?, ?)', (city.lower(), price))
        conn.commit()


def price_setter():
    ticket_prices = {"london": 799, "paris": 899, "tokyo": 1400, "berlin": 499}
    for city, price in ticket_prices.items():
        set_ticket_price(city, price)

