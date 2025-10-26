import streamlit as st
from agents.expense_tracker_agent import DB
import pandas as pd
import datetime

st.title(f"{st.session_state.username}的账本")

current_year = datetime.datetime.now().year
year = st.selectbox(
    "选择需要查看的年份",
    range(current_year, 1970, -1)
)

y = []
monthly_consumption = {
    "value": [0,]
}
for m in range(1, 13):
    month_name = str(year) + "-" + str(m).zfill(2)
    y.append(month_name)
    value = DB.get_month_consumption(year, m)
    monthly_consumption["value"].append(value)
st.bar_chart(pd.DataFrame(monthly_consumption), x_label="月份", y_label="金额（元）", stack=False)

ledger_data = {
    "名称": [],
    "金额": [],
    # "分类": [],
    "日期": [],
    "时间": []
}

for record in DB.get_all_transactions():
    ledger_data["名称"].append(record[3])
    ledger_data["金额"].append(int(record[4])/100)
    # ledger_data["分类"].append(record[6])
    date, time = record[2].split("T")
    ledger_data["日期"].append(date)
    ledger_data["时间"].append(time)

ledger_table = st.table(
    ledger_data,
    border="horizontal",
)




