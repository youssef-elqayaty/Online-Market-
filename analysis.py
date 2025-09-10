import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------ Paths ------------
data_path = "data/train.csv"
out_plots = "outputs/plots"
out_data = "outputs/data"
os.makedirs(out_plots, exist_ok=True)
os.makedirs(out_data, exist_ok=True)

# ------------ Read CSV or Excel ------------
if data_path.lower().endswith(".csv"):
    df = pd.read_csv(data_path)
else:
    df = pd.read_excel(data_path)

# ------------ Detect columns ------------
cols = [c for c in df.columns]
order_col = next((c for c in cols if "order" in c.lower() and "date" in c.lower()), None)
if order_col is None:
    order_col = next((c for c in cols if "date" in c.lower()), None)
state_col = next((c for c in cols if c.lower() == "state"), next((c for c in cols if "state" in c.lower()), None))
sales_col = next((c for c in cols if c.lower() == "sales"), next((c for c in cols if "sale" in c.lower()), None))
subcat_col = next((c for c in cols if "sub" in c.lower() and "cat" in c.lower()), next((c for c in cols if "sub" in c.lower()), None))

if order_col is None or state_col is None or sales_col is None:
    raise Exception("Dataset must have columns for Order Date, State, and Sales.")

df["Order Date"] = pd.to_datetime(df[order_col], dayfirst=True, errors='coerce')
df["State"] = df[state_col]
df["Sales"] = pd.to_numeric(df[sales_col], errors='coerce')
if subcat_col:
    df["Sub-Category"] = df[subcat_col]

df = df.dropna(subset=["Order Date", "State", "Sales"]).drop_duplicates()
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# ------------ Top 5 States ------------
state_totals = df.groupby("State", as_index=False)["Sales"].sum().sort_values(by="Sales", ascending=False)
top5_states = state_totals.head(5)["State"].tolist()
print("Top 5 states:", top5_states)
state_totals.to_csv(os.path.join(out_data, "state_totals.csv"), index=False)

state_year_sales = (
    df[df["State"].isin(top5_states)]
    .groupby(["Year", "State"], as_index=False)["Sales"]
    .sum()
)
state_year_sales.to_csv(os.path.join(out_data, "top5_state_year.csv"), index=False)

plt.figure(figsize=(12,6))
sns.barplot(data=state_year_sales, x="Year", y="Sales", hue="State")
plt.title("Top 5 States Sales by Year")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(out_plots, "top5_states_by_year.png"))
plt.clf()

# ------------ California Sub-Categories ------------
target_state = "California"
if "Sub-Category" in df.columns:
    df_ca = df[df["State"] == target_state]
    if not df_ca.empty:
        ca_subcat = df_ca.groupby(["Year", "Sub-Category"], as_index=False)["Sales"].sum()
        ca_subcat.to_csv(os.path.join(out_data, "california_subcat_year.csv"), index=False)
        total_year = df_ca.groupby("Year", as_index=False)["Sales"].sum().rename(columns={"Sales":"TotalYearSales"})
        ca_share = ca_subcat.merge(total_year, on="Year")
        ca_share["Pct"] = (ca_share["Sales"] / ca_share["TotalYearSales"]) * 100
        ca_share.to_csv(os.path.join(out_data, "california_subcat_year_pct.csv"), index=False)

        plt.figure(figsize=(14,7))
        sns.barplot(data=ca_subcat, x="Year", y="Sales", hue="Sub-Category")
        plt.title(f"{target_state} Sub-Category Sales by Year")
        plt.xlabel("Year")
        plt.ylabel("Total Sales")
        plt.legend(title="Sub-Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_plots, "california_subcat_by_year.png"))
        plt.clf()
