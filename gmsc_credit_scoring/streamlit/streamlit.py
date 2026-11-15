import streamlit as st      
import requests             
import matplotlib.pyplot as plt  
import shap                 
import numpy as np          
import os                  
import pandas as pd         


st.set_page_config(
    page_title='Кредитный скоринг',
    layout='centered'
)
st.title('Кредитный скоринг - оценка риска дефолта')


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")



with st.form("prediction_form"):
    st.subheader("Данные клиента")


    RevolvingUtilizationOfUnsecuredLines = st.number_input(
        'Превышение кредитного лимита (доля)',
        min_value=0.0, max_value=2.0,
        value=0.33,   
        step=0.01,
        help="0 = лимит не используется, 1.0 = лимит полностью исчерпан, >1 = превышен"
    )
    total_late_payments = st.number_input(
        'Общее количество просроченных платежей',
        min_value=0, max_value=30,
        value=0,
        help="Сумма просрочек 30-59 дней + 60-89 дней + 90+ дней"
    )
    DebtRatio = st.number_input(
        'Соотношение долга к зарплате (доля)',
        min_value=0.0, max_value=10.0,
        value=0.52,
        step=0.01,
        help="Ежемесячные выплаты по долгу / Ежемесячный доход"
    )
    age = st.number_input(
        'Возраст',
        min_value=21, max_value=104,
        value=40
    )
    NumberOfOpenCreditLinesAndLoans = st.number_input(
        'Количество открытых кредитов',
        min_value=0, max_value=85,
        value=5
    )
    MonthlyIncome = st.number_input(
        'Месячная зарплата ($)',
        min_value=0.0, max_value=8_000_000.0,
        value=7000.0,
        step=100.0
    )
    income_per_dependent = st.number_input(
        'Зарплата на члена семьи ($)',
        min_value=0.0, max_value=4_200_000.0,
        value=4700.0,
        step=100.0,
        help="MonthlyIncome / NumberOfDependents"
    )
    NumberOfTimes90DaysLate = st.number_input(
        'Просрочек более 90 дней',
        min_value=0, max_value=13,
        value=0
    )

    
    submitted = st.form_submit_button('Оценить риск', use_container_width=True)



if submitted:
    data = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "total_late_payments": int(total_late_payments),
        "DebtRatio": DebtRatio,
        "age": int(age),
        "NumberOfOpenCreditLinesAndLoans": int(NumberOfOpenCreditLinesAndLoans),
        "MonthlyIncome": MonthlyIncome,
        "income_per_dependent": income_per_dependent,
        "NumberOfTimes90DaysLate": int(NumberOfTimes90DaysLate),
    }

    with st.spinner("Модель считает..."):
        try:
            response = requests.post(API_URL, json=data, timeout=10)

            if response.status_code == 200:
                result = response.json()

                prediction = result['prediction']
                probability = result['probability']

                
                st.divider()

                if prediction == 1:
                    st.error(f"Высокий риск дефолта: **{probability:.1%}**")
                else:
                    st.success(f"Низкий риск дефолта: **{probability:.1%}**")

                
                st.progress(probability)
                st.subheader("Факторы влияния на решение")
                st.caption("Красный - увеличивает риск дефолта | Синий - снижает риск")

                shap_values = np.array(result['shap_values'])    
                expected_value = result['expected_value']          
                feature_names = result['feature_names']            
                feature_values = np.array(result['feature_values'])

               
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=expected_value,
                    data=feature_values,
                    feature_names=feature_names
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(explanation, show=False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.subheader("Детальный анализ признаков")
                details = pd.DataFrame({
                    'Признак': feature_names,
                    'Значение клиента': feature_values,
                    'Вклад SHAP': [round(v, 4) for v in shap_values]
                })

                details = details.sort_values('Вклад SHAP', key=abs, ascending=False)

                details['Влияние'] = details['Вклад SHAP'].apply(
                    lambda x: 'Увеличивает риск' if x > 0 else 'Снижает риск'
                )

                st.dataframe(details, use_container_width=True, hide_index=True)

            else:
                st.error(f"Ошибка API: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Не удалось подключиться к серверу.\n\n"
                f"Проверь что FastAPI запущен на {API_URL}"
            )
        except requests.exceptions.Timeout:
            st.error("Сервер не ответил")
        except Exception as e:
            st.error(f"Неизвестная ошибка: {e}")

st.divider()
st.subheader("Управление моделью")

TRAIN_URL = os.getenv("API_URL", "http://localhost:8000").replace("/predict", "") + "/train"

if st.button("Обучить модель на данных из БД", type="secondary"):
    with st.spinner("Модель обучается"):
        try:
            r = requests.post(TRAIN_URL, timeout=120)
            if r.status_code == 200:
                result = r.json()
                st.success("Обучение завершено!")
                col1, col2, col3 = st.columns(3)
                col1.metric("ROC-AUC",    result["roc_auc"])
                col2.metric("PR-AUC",     result["pr_auc"])
                col3.metric("Train rows", result["train_rows"])
            else:
                st.error(f"Ошибка обучения: {r.text}")
        except requests.exceptions.Timeout:
            st.error("Таймаут - обучение заняло больше 2 минут")
        except Exception as e:
            st.error(f"Ошибка: {e}")