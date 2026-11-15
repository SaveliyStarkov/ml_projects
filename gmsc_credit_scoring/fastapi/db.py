import psycopg
import pandas as pd
import os


def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "credit_scoring"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        port=int(os.getenv("DB_PORT", 5432))
    )


def load_training_data() -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    revolving_utilization,
                    total_late_payments,
                    debt_ratio,
                    age,
                    number_of_open_credits,
                    monthly_income,
                    income_per_dependent,
                    times_90_days_late,
                    target
                FROM customers
            """)
            rows = cur.fetchall()

    columns = [
        "RevolvingUtilizationOfUnsecuredLines",
        "total_late_payments",
        "DebtRatio",
        "age",
        "NumberOfOpenCreditLinesAndLoans",
        "MonthlyIncome",
        "income_per_dependent",
        "NumberOfTimes90DaysLate",
        "target"
    ]
    return pd.DataFrame(rows, columns=columns)


def save_prediction(prediction: int, probability: float, model_version: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (prediction, probability, model_version)
                VALUES (%s, %s, %s)
            """, (prediction, probability, model_version))
        conn.commit()


def save_training_run(model_version: str, roc_auc: float,
                      pr_auc: float, train_rows: int, status: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_runs
                    (model_version, roc_auc, pr_auc, train_rows, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (model_version, roc_auc, pr_auc, train_rows, status))
        conn.commit()