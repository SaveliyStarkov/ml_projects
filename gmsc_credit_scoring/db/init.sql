-- Таблица данных для обучения
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    revolving_utilization   FLOAT NOT NULL,
    total_late_payments     INT   NOT NULL,
    debt_ratio              FLOAT NOT NULL,
    age                     INT   NOT NULL,
    number_of_open_credits  INT   NOT NULL,
    monthly_income          FLOAT NOT NULL,
    income_per_dependent    FLOAT NOT NULL,
    times_90_days_late      INT   NOT NULL,
    target                  INT   NOT NULL
);

-- Таблица результатов предиктов
CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    prediction    INT     NOT NULL,
    probability   FLOAT   NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Таблица истории обучений
CREATE TABLE IF NOT EXISTS training_runs (
    id            SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    roc_auc       FLOAT,
    pr_auc        FLOAT,
    train_rows    INT,
    status        VARCHAR(20) NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);