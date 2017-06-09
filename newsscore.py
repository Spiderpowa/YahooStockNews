import csv
from collections import defaultdict
from datetime import datetime, timedelta


scores = defaultdict()


def post_score(post):
    if len(scores) == 0:
        build_score()
    date = datetime.strptime(post['date'], '%Y-%m-%d %H:%M:%S')
    for _ in range(10):
        date = date + timedelta(days=1)
        day = date.date().strftime('%Y-%m-%d')
        if day in scores:
            return scores[day]
    return 0


def build_score():
    print("Building Score...")
    histories = []
    with open('2330.TW.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[4] == 'null' or row[1] == 'null':
                continue
            histories.append({
                'Date': row[0],
                'Open': row[1],
                'High': row[2],
                'Low': row[3],
                'Close': row[4],
                'AdjClose': row[5],
                'Volume': row[6]
            })
    for history in histories:
        scores[history['Date']] = \
            1 if history['Close'] > history['Open'] else 0
    print("Building Score Finished")
