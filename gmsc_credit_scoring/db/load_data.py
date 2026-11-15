import psycopg
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def load_csv_to_postgres():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed.csv"))
    df = df.rename(columns={
        "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
        "total_late_payments":                  "total_late_payments",
        "DebtRatio":                            "debt_ratio",
        "age":                                  "age",
        "NumberOfOpenCreditLinesAndLoans":       "number_of_open_credits",
        "MonthlyIncome":                         "monthly_income",
        "income_per_dependent":                  "income_per_dependent",
        "NumberOfTimes90DaysLate":               "times_90_days_late",
        "SeriousDlqin2yrs":                      "target",
    })

    cols = [
        "revolving_utilization", "total_late_payments", "debt_ratio",
        "age", "number_of_open_credits", "monthly_income",
        "income_per_dependent", "times_90_days_late", "target"
    ]
    df = df[cols].dropna()

    conn = psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", 5432))
    )

    with conn.cursor() as cur:
        # Очищаем таблицу если уже есть данные
        cur.execute("TRUNCATE TABLE customers RESTART IDENTITY")

        # Батчевая вставка — быстро
        rows = [tuple(row) for row in df.itertuples(index=False)]
        cur.executemany(
            f"INSERT INTO customers ({', '.join(cols)}) VALUES "
            f"({', '.join(['%s'] * len(cols))})",
            rows
        )

    conn.commit()
    conn.close()
    print(f"Загружено {len(df)} строк в таблицу customers")

if __name__ == "__main__":
    load_csv_to_postgres()