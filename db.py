import pandas as pd

FILE = "accounts.xlsx"

def load_data():
    return pd.read_excel(FILE)

def save_data(df):
    df.to_excel(FILE, index=False)

def get_user(account_number):
    df = load_data()
    user = df[df["account_number"] == account_number]
    return user, df

def update_balance(df, account_number, new_balance):
    df.loc[df["account_number"] == account_number, "balance"] = new_balance
    save_data(df)