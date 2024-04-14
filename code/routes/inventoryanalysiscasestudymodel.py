#Import libraries
import numpy as np         #Mathematical Computation
import pandas as pd        # Data Manipulation
import matplotlib.pyplot as plt        # Data Visualization
import seaborn as sns                  # Data Visualization
from sqlalchemy import create_engine
from configparser import ConfigParser
import logging
import warnings
warnings.filterwarnings('ignore')
import psycopg2

# Replace 'your_username', 'your_password', 'your_host', 'your_port', 'your_database_name' with your actual PostgreSQL credentials
DATABASE_URL = "postgresql://postgres:Sathvika@localhost:5432/Stock"

# Actual table names
table_name = 'purchase_prices'
table_name1 = 'beg_inv'
table_name2 = 'end_inv'
table_name3 = 'purchases'
table_name4 = 'invoice_purchases'
table_name5 = 'sales'

# Create a connection to the database
conn = psycopg2.connect(DATABASE_URL)

# Use pandas to read data from PostgreSQL into a DataFrame
df_purchaseprice = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
df_beginv = pd.read_sql_query(f"SELECT * FROM {table_name1}", con=conn)
df_endinv = pd.read_sql_query(f"SELECT * FROM {table_name2}", con=conn)
df_invoicepurchases = pd.read_sql_query(f"SELECT * FROM {table_name4}", con=conn)
df_purchases = pd.read_sql_query(f"SELECT * FROM {table_name3}", con=conn)
df_sales = pd.read_sql_query(f"SELECT * FROM {table_name5}", con=conn)

# Close the connection
conn.close()
df_purchaseprice.head()
df_beginv.head()
df_endinv.head()
df_purchases.head()
df_invoicepurchases.head()
df_sales.head()
print(df_purchaseprice.shape)
print(df_beginv.shape)
print(df_endinv.shape)
print(df_purchases.shape)
print(df_invoicepurchases.shape)
print(df_sales.shape)

print("purchase_prices Columns:")
print(df_purchaseprice.info())

print("\nbeg_inv Columns:")
print(df_beginv.info())

print("\nend_inv Columns:")
print(df_endinv.info())

print("\ninvoice_purchases Columns:")
print(df_purchases.info())

print("\npurchases Columns:")
print(df_invoicepurchases.info())

print("\nsales Columns:")
print(df_sales.info())

print('Purchase Price Description')
print('\n')
print(df_purchaseprice.describe(include='all'))
print('Beginning Inventor Description')
print('\n')
print(df_beginv.describe(include='all'))
print('End Inventor Description')
print('\n')
print(df_endinv.describe(include='all'))
print('Purchases Description')
print('\n')
print(df_purchases.describe(include='all'))
print('Invoice Purchases Description')
print('\n')
print(df_invoicepurchases.describe(include='all'))
print('Sales Description')
print('\n')
print(df_sales.describe(include='all'))

# # Data Processings
# Checking for missing data in each dataset
datasets = [df_purchaseprice, df_beginv, df_endinv, df_invoicepurchases, df_purchases, df_sales]
dataset_names = ["purchase_prices", "beg_inv", "end_inv", "invoice_purchases", "purchases", "sales"]

for name, data in zip(dataset_names, datasets):
    missing_values = data.isnull().sum()
    non_zero_missing_values = missing_values[missing_values > 0]
    
    if not non_zero_missing_values.empty:
        print(f"\nMissing values in {name}:")
        print(non_zero_missing_values)

# Handling missing values for purchase_prices dataset
columns = ['Description', 'Size', 'Volume']
for col in columns:
    df_purchaseprice = df_purchaseprice[df_purchaseprice[col].notna()]


# Handling missing values for end_inv dataset
if df_endinv['Store'].nunique() == df_endinv['City'].nunique():
    city_store_mapping = df_endinv[['Store', 'City']].drop_duplicates().set_index('Store').to_dict()['City']
    df_endinv['City'] = df_endinv['City'].fillna(df_endinv['Store'].map(city_store_mapping))
else:
    df_endinv['City'].fillna('Unknown', inplace=True)



# Handling missing values for invoice_purchases dataset
df_invoicepurchases['Approval'].fillna('Pending', inplace=True)

# Handling missing values for purchases dataset
df_purchases = df_purchases[df_purchases['Size'].notna()]

datasets = [df_purchaseprice, df_beginv, df_endinv, df_invoicepurchases, df_purchases, df_sales]
dataset_names = ["purchase_prices", "beg_inv", "end_inv", "invoice_purchases", "purchases", "sales"]

for name, data in zip(dataset_names, datasets):
    missing_values = data.isnull().sum()
    non_zero_missing_values = missing_values[missing_values > 0]
    
    if not non_zero_missing_values.empty:
        print(f"\nMissing values in {name}:")
        print(non_zero_missing_values)
    else:
        print(f"\nNo missing values in {name}.")


# # Real Time inventory Monitoring

from scipy.optimize import minimize
def economic_order_quantity(demand, ordering_cost, holding_cost, unit_cost):
            """
            Calculate Economic Order Quantity (EOQ) and visualize the cost components.

            Parameters:
            - demand: Average demand for the product.
            - ordering_cost: Cost to place one order.
            - holding_cost: Cost to hold one unit in inventory for a year.
            - unit_cost: Cost per unit of the product.

            Returns:
            - optimal_order_quantity: Optimal order quantity that minimizes total costs.
            """

            # Cost function for EOQ analysis
            def total_cost(q, demand, ordering_cost, holding_cost, unit_cost):
                return ordering_cost + (demand / q) * holding_cost + unit_cost * demand

            # Optimize the order quantity (EOQ)
            result = minimize(total_cost, x0=100, args=(demand, ordering_cost, holding_cost, unit_cost), bounds=[(1, None)])

            optimal_order_quantity = result.x[0]

            # Visualization of cost components
            q_values = np.linspace(1, 2 * optimal_order_quantity, 1000)
            cost_values = total_cost(q_values, demand, ordering_cost, holding_cost, unit_cost)

            plt.figure(figsize=(10, 6))

            # Plotting the total cost curve
            plt.plot(q_values, cost_values, label='Total Cost', color='blue')

            # Marking the optimal order quantity
            plt.scatter(optimal_order_quantity, total_cost(optimal_order_quantity, demand, ordering_cost, holding_cost, unit_cost),
                        color='red', label=f'Optimal EOQ: {optimal_order_quantity:.2f}', zorder=5)

            # Annotating the optimal order quantity
            plt.annotate(f'Optimal EOQ: {optimal_order_quantity:.2f}', 
                         xy=(optimal_order_quantity, total_cost(optimal_order_quantity, demand, ordering_cost, holding_cost, unit_cost)),
                         xytext=(optimal_order_quantity + 300, total_cost(optimal_order_quantity, demand, ordering_cost, holding_cost, unit_cost) + 300),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         fontsize=10)

            plt.title('Economic Order Quantity (EOQ) Analysis')
            plt.xlabel('Order Quantity (Q)')
            plt.ylabel('Total Cost')
            plt.legend()
            plt.grid(True)
            plt.show()

            return optimal_order_quantity


# !pip3 install --upgrade tensorflow
# !pip3 uninstall tensorflow
# !pip3 install tensorflow

# import tensorflow as tf
# print(tf.__version__)
# !pip list


from sklearn.preprocessing import  StandardScaler
from keras.models import Sequential

from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Grouping by Brand,price, startdate and Description and summarize inventory for beginning of the year
beg_summary = df_beginv.groupby(['Brand', 'Description','Price','startDate'])['onHand'].sum().sort_values(ascending=False)

# Grouping by Brand, price, enddate and Description and summarize inventory for end of the year
end_summary = df_endinv.groupby(['Brand', 'Description','Price','endDate'])['onHand'].sum().sort_values(ascending=False)

# Identifying top 5 products at the beginning and end of the year
top5_beg = beg_summary.head(5)
top5_end = end_summary.head(5)

top5_beg.plot(kind='pie',figsize=(10,10),autopct='%1.1f%%',fontsize=20)
plt.show()
top5_end.plot(kind='pie',figsize=(10,10),autopct='%1.1f%%',fontsize=20)
plt.show()

#Real-time overall Inventory Monitoring
current_stock = df_beginv['onHand'].sum()
print(f"Current stock level: {current_stock}")

##Current stock at the beginning of the year in every city
df_beginv.groupby(['City'])['onHand'].sum().plot(kind='barh',figsize=(20,15))
plt.show()
##Current stock at the end of the year in every city
df_endinv.groupby(['City'])['onHand'].sum().plot(kind='barh',figsize=(20,15))
plt.show()

# 2. Increased carrying costs due to frequent stock-outs and excess inventory:
average_demand = df_purchases['Quantity'].mean() 
# ordering_cost = 100   
# holding_cost = 2      
# unit_cost = 10 

# Assuming a constant ordering cost for all products
ordering_cost_per_order = 50  # Placeholder for ordering cost

# Calculate total ordering cost for the subset
total_ordering_cost = len(df_purchases) * ordering_cost_per_order

# Assuming a constant holding cost per unit for all products
holding_cost_per_unit = 5  # Placeholder for holding cost 

# Calculate total holding cost for the subset
total_holding_cost = (df_purchases['Quantity'] / 2).sum()  # Assuming average inventory is half the order quantity

# Calculate total cost for the subset
total_cost = total_ordering_cost + total_holding_cost + df_purchases['PurchasePrice'].sum()

# Calculate unit cost
unit_cost = total_cost / df_purchases['Quantity'].sum()

print(f"Average Demand: {average_demand:.2f}")
print(f"Total Ordering Cost: ${total_ordering_cost:.2f}")
print(f"Total Holding Cost: ${total_holding_cost:.2f}")
print(f"Total Cost: ${total_cost:.2f}")
print(f"Unit Cost: ${unit_cost:.2f}")
optimal_order_quantity = economic_order_quantity(average_demand, total_ordering_cost, total_holding_cost, unit_cost)
print(f"Optimal Order Quantity (EOQ): {optimal_order_quantity}")

#ABC Analysis:
total_sales = df_sales['SalesDollars'].sum()

df_sales['Contribution'] = df_sales['SalesDollars'] / total_sales
df_sales['ABC_Classification'] = pd.qcut(df_sales['Contribution'], q=[0, 0.2, 0.8, 1], labels=['A', 'B', 'C'])

print(df_sales[['InventoryId', 'ABC_Classification']])
# plt.style.use('seaborn-notebook')
df_sales[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30).plot(kind='barh',fontsize=20,
                                                                                             figsize=(20,15),stacked=True,color='green')

# Assuming df_sales is your DataFrame
df_sales[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30).plot(kind='barh', fontsize=20, figsize=(20,15), stacked=True, color='green')

# Calculate percentages
total_count = len(df_sales)
percentages = df_sales[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30) / total_count * 100

# Annotate the bars with percentages
for index, value in enumerate(df_sales[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30)):
    plt.text(value, index, f'{value} ({percentages.iloc[index]:.2f}%)', va='center', fontsize=12, color='black')

reduced_purchases = df_purchases[['VendorName', 'PurchasePrice']]
top_vendors = reduced_purchases.groupby('VendorName').sum()['PurchasePrice'].nlargest(10)

# top_vendors = reduced_purchases.groupby('VendorName').sum()['PurchasePrice'].nlargest(10)

plt.figure(figsize=(10, 6))
top_vendors.plot(kind='barh', color='Green', fontsize=15)
plt.title('Top 10 Vendors by Purchase Cost', fontsize=15)
plt.ylabel('Vendor Name')
plt.xlabel('Purchase Cost')
plt.xticks(rotation=45, ha='right')

# Add percentage annotations to the bars
total_purchase = top_vendors.sum()
for index, value in enumerate(top_vendors):
    percentage = (value / total_purchase) * 100
    plt.text(value, index, f'{percentage:.2f}%', va='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()

#  Convert 'SalesDate' to datetime type
df_sales['SalesDate'] = pd.to_datetime(df_sales['SalesDate'])
start_date = df_sales['SalesDate'].min()
end_date = df_sales['SalesDate'].max()
total_days = (end_date - start_date).days ## Number of total days

## Datetime conversion
df_purchases['ReceivingDate'] = pd.to_datetime(df_purchases['ReceivingDate'])
df_purchases['PODate'] = pd.to_datetime(df_purchases['PODate'])
##Calculate lead time
df_purchases['Lead_Time'] = (df_purchases['ReceivingDate'] - df_purchases['PODate']).dt.days

# Calculating Sales Velocity for each product, to measure how quickly we're selling our products 
sales_velocity = df_sales.groupby(['Brand', 'Description']).agg(Total_Sales=('SalesQuantity', 'sum')).reset_index()
sales_velocity['Sales_Per_Day'] = sales_velocity['Total_Sales'] / total_days
df_purchases.loc[:, 'Lead_Time'] = (df_purchases['ReceivingDate'] - df_purchases['PODate']).dt.days
lead_times = df_purchases.groupby(['Brand', 'Description']).agg(Avg_Lead_Time=('Lead_Time', 'mean')).reset_index()

# Merging the data
merge_df= pd.merge(sales_velocity, lead_times, on=['Brand', 'Description'], how='left')

# Calculating Optimal Stock Level
merge_df['Optimal_Stock_Level'] = merge_df['Sales_Per_Day'] * merge_df['Avg_Lead_Time']

# Calculating Safety Stock using maximum sales for each product
max_sales = df_sales.groupby(['Brand', 'Description']).agg(Max_Daily_Sales=('SalesQuantity', 'max')).reset_index()
merge_df = pd.merge(merge_df, max_sales, on=['Brand', 'Description'], how='left')

merge_df['Safety_Stock'] = merge_df['Max_Daily_Sales'] - merge_df['Sales_Per_Day']
merge_df['Recommended_Stock_Level'] = merge_df['Optimal_Stock_Level'] + merge_df['Safety_Stock']
# Updating Max_Daily_Sales for problematic products
merge_df.loc[merge_df['Sales_Per_Day'] > merge_df['Max_Daily_Sales'], 'Max_Daily_Sales'] = merge_df['Sales_Per_Day']

# Updating Safety Stock and Recommended Stock Level after modifying Max_Daily_Sales
merge_df['Safety_Stock'] = merge_df['Max_Daily_Sales'] - merge_df['Sales_Per_Day']
merge_df['Recommended_Stock_Level'] = merge_df['Optimal_Stock_Level'] + merge_df['Safety_Stock']

# Sorting the data by Recommended_Stock_Level for better visualization
sorted_data = merge_df.sort_values(by='Recommended_Stock_Level', ascending=False)

# Plotting
plt.figure(figsize=(15, 10))
ax = sns.barplot(x='Recommended_Stock_Level', y='Description', data=sorted_data.head(20), palette='magma')  
plt.xlabel('Recommended Stock Level')
plt.ylabel('Product Description')
plt.title('Recommended Stock Levels for Top 20 Products')

# Add percentage annotations to the bars
total_recstock = sorted_data['Recommended_Stock_Level'].head(20).sum()
for index, value in enumerate(sorted_data['Recommended_Stock_Level'].head(20)):
    percentage = (value / total_recstock) * 100
    ax.text(value, index, f'{percentage:.2f}%', va='center', fontsize=12, color='blue')  
plt.tight_layout()
plt.show()

# Sorting the data by Sales_Per_Day in descending order to get top products
top_products = sales_velocity.sort_values(by='Sales_Per_Day', ascending=False)

# Creating the bar plot
plt.figure(figsize=(15, 10))
ax = sns.barplot(y='Description', x='Sales_Per_Day', data=top_products.head(20), palette='magma')  
plt.xticks(rotation=50, ha='right')
plt.title('Daily Sales Velocity by Product')
plt.xlabel('Product')
plt.ylabel('Sales Per Day')

# Add percentage annotations to the bars
total_sales = top_products['Sales_Per_Day'].head(20).sum()
for index, value in enumerate(top_products['Sales_Per_Day'].head(20)):
    percentage = (value / total_sales) * 100
    ax.text(value, index, f'{percentage:.2f}%', va='center', fontsize=12, color='blue')  
plt.tight_layout()
plt.show()


#optimal stcok level
df_endinv['endDate'] = pd.to_datetime(df_endinv['endDate'])
latest_inventory_date = df_endinv['endDate'].max()
current_inventory = df_endinv[df_endinv['endDate'] == latest_inventory_date]

# Summarizing the current stock levels by product.
current_stock_levels = current_inventory.groupby(['Brand', 'Description']).agg(Current_Stock=('onHand', 'sum')).reset_index()

# Merging the current stock levels with the previously calculated data.
final_data = pd.merge(merge_df, current_stock_levels, on=['Brand', 'Description'], how='left')

# Assume zero current stock for any products not present in the current inventory.
final_data['Current_Stock'] = final_data['Current_Stock'].fillna(0)

# Calculating how much of each product needs to be ordered if current stock is below recommended levels.
final_data['Order_Quantity'] = final_data['Recommended_Stock_Level'] - final_data['Current_Stock']
final_data['Order_Quantity'] = final_data['Order_Quantity'].clip(lower=0)  # Setting negative order quantities to zero.

# Reporting the results.
# print(final_data[['Brand', 'Description', 'Current_Stock', 'Recommended_Stock_Level', 'Order_Quantity']])

# sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.barplot(x='Order_Quantity', y='Description', data=final_data.sort_values('Order_Quantity', ascending=False).head(10))
plt.title('Top 10 Products to Reorder')
plt.xlabel('Quantity to Order')
plt.ylabel('Product Description')
plt.show()
 
# Filtering the data to show the top 10 products where the ordering quantity is highest
top_products_to_order = final_data.nlargest(10, 'Order_Quantity')
# Plotting the bars
fig, ax = plt.subplots(figsize=(12, 8))
# Indexing for the bars
ind = np.arange(len(top_products_to_order))
# Width of the bars
bar_width = 0.4
# Plotting current stock and recommended stock side by side
ax.barh(ind, top_products_to_order['Current_Stock'], bar_width, color='blue', label='Current Stock')
ax.barh([i + bar_width for i in ind], top_products_to_order['Recommended_Stock_Level'], bar_width, color='Red', label='Recommended Stock')
# Setting the y-axis labels to product descriptions
ax.set(yticks=[i + bar_width for i in ind], yticklabels=top_products_to_order['Description'], ylim=[2 * bar_width - 1, len(ind)])
# Adding the legend
ax.legend()
# Adding labels and title
ax.set_xlabel('Quantity')
ax.set_title('Top 10 Products(by Order Quantity): Current vs Recommended Stock Levels')
plt.tight_layout()
plt.show()

df_sales.InventoryId.nunique()
# df_sales.shape

# Reorder Points
historical_sales = df_sales[['SalesQuantity', 'SalesDate', 'InventoryId','SalesPrice']]

df_sales.head(2)

historical_sales['SalesDate'] = pd.to_datetime(historical_sales['SalesDate'])
historical_sales['SalesDate_timestamp'] = historical_sales['SalesDate'].apply(lambda x: x.timestamp())

# Set the 'SalesDate' column as the index
historical_sales.set_index('SalesDate', inplace=True)

# # Resample the DataFrame based on the Daily frequency ('D') and calculate the mean
df_daily = historical_sales[['SalesQuantity', 'SalesPrice']].resample('D').mean()
# df_daily = historical_sales[['InventoryId', 'Brand', 'SalesQuantity', 'SalesPrice']].resample('D').mean()
# df_daily = historical_sales.groupby('InventoryId').resample('D').mean()
# Print the first few rows of the resampled DataFrame
print(df_daily.head())

df_daily.plot(figsize=(14,6))
plt.show()
from sklearn.preprocessing import  StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Data preprocessing
scaler = StandardScaler()
df_daily_scaled = scaler.fit_transform(df_daily[['SalesPrice']])

# Create sequences for LSTM
sequence_length = 10  # Choose an appropriate sequence length
X, y = [], []
for i in range(len(df_daily_scaled) - sequence_length):
    X.append(df_daily_scaled[i:i+sequence_length, 0])
    y.append(df_daily_scaled[i+sequence_length, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM input shape (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Forecast future values
forecast_steps = 10  # Adjust the number of steps as needed
forecast_input = df_daily_scaled[-sequence_length:, 0]
forecast_input = np.reshape(forecast_input, (1, sequence_length, 1))
forecast_scaled = []

for _ in range(forecast_steps):
    forecast_value = model.predict(forecast_input, verbose=0)
    forecast_scaled.append(forecast_value[0, 0])
    forecast_input = np.roll(forecast_input, -1)
    forecast_input[0, -1, 0] = forecast_value[0, 0]

# Inverse transform the forecasted values
forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
forecast = scaler.inverse_transform(forecast_scaled)

# Plot the original time series and the forecast
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['SalesPrice'], label='Actual SalesPrice', marker='o')
plt.plot(pd.date_range(df_daily.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq=df_daily.index.freq),
         forecast, color='red', label='Forecasted SalesPrice')
plt.title('LSTM Model Forecast')
plt.xlabel('SalesDate')
plt.ylabel('SalesPrice')
plt.legend()
plt.show()

from keras.models import save_model
save_model(model, "lstm_model_SalesPrice.keras")

from keras.models import load_model
loaded_model = load_model("lstm_model_SalesPrice.keras")

# Make predictions using the loaded model
forecast_scaled = []

for _ in range(forecast_steps):
    forecast_value = loaded_model.predict(forecast_input, verbose=0)
    forecast_scaled.append(forecast_value[0, 0])
    forecast_input = np.roll(forecast_input, -1)
    forecast_input[0, -1, 0] = forecast_value[0, 0]

# Inverse transform the forecasted values
forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
forecast = scaler.inverse_transform(forecast_scaled)


# Data preprocessing
scaler = StandardScaler()
df_daily_scaled = scaler.fit_transform(df_daily[['SalesQuantity']])

# Create sequences for LSTM
sequence_length = 10  # Choose an appropriate sequence length
X, y = [], []
for i in range(len(df_daily_scaled) - sequence_length):
    X.append(df_daily_scaled[i:i+sequence_length, 0])
    y.append(df_daily_scaled[i+sequence_length, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM input shape (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model with more parameters
model1 = Sequential()
model1.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='relu'))
model1.add(LSTM(units=100, activation='relu'))
model1.add(Dense(units=1, activation='linear'))  # Using linear activation for regression tasks

model1.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Forecast future values
forecast_steps1 = 10  # Adjust the number of steps as needed
forecast_input1 = df_daily_scaled[-sequence_length:, 0]
forecast_input1 = np.reshape(forecast_input1, (1, sequence_length, 1))
forecast_scaled1 = []

for _ in range(forecast_steps):
    forecast_value1 = model.predict(forecast_input1, verbose=0)
    # forecast_scaled1.append(forecast_value1[0, 0])

    forecast_scaled1 = np.concatenate((forecast_scaled1, forecast_value1[0, 0]), axis=None)
    forecast_input1 = np.roll(forecast_input1, -1)
    forecast_input1[0, -1, 0] = forecast_value1[0, 0]



# Inverse transform the forecasted values
forecast_scaled1 = np.array(forecast_scaled1).reshape(-1, 1)
forecast1 = scaler.inverse_transform(forecast_scaled1)

# Create a DataFrame with the forecasted values
forecast_dates = pd.date_range(df_daily.index[-1] + pd.DateOffset(1), periods=forecast_steps1, freq=df_daily.index.freq)
forecast_df = pd.DataFrame({'SalesDate': forecast_dates, 'ForecastedSalesQuantity': forecast1.flatten()})

# Merge the forecast DataFrame with the original DataFrame on 'SalesDate'
df_daily_reset = df_daily.reset_index()
merged_df = pd.merge(df_daily_reset[['SalesDate',  'SalesQuantity']], forecast_df, on='SalesDate', how='outer')

# Plot the original time series and the forecast
plt.figure(figsize=(12, 6))
plt.plot(merged_df['SalesDate'], merged_df['SalesQuantity'], label='Actual SalesQuantity', marker='o')
plt.plot(merged_df['SalesDate'], merged_df['ForecastedSalesQuantity'], color='red', label='Forecasted SalesQuantity')
plt.title('LSTM Model Forecast')
plt.xlabel('SalesDate')
plt.ylabel('SalesQuantity')
plt.legend()
plt.show()


df_beginv['startDate'] = pd.to_datetime(df_beginv['startDate'])  # Assuming 'startDate' column contains the date
df_sales['SalesDate'] = pd.to_datetime(df_sales['SalesDate'])  # Assuming 'SalesDate' column contains the date
# Group sales by date and sum the quantities sold
daily_sales = df_sales.groupby('SalesDate')['SalesQuantity'].sum().reset_index()

# Merge with beginning inventory data
daily_inventory = pd.merge_asof(daily_sales, df_beginv, left_on='SalesDate', right_on='startDate')

# Calculate current stock for each day
daily_inventory['CurrentStock'] = daily_inventory['onHand'].cumsum()

# Plot current stock over time
plt.figure(figsize=(10, 6))
plt.plot(daily_inventory['SalesDate'], daily_inventory['CurrentStock'], marker='o')
plt.xlabel('Date')
plt.ylabel('Current Stock')
plt.title('Current Stock Over Time')
plt.grid(True)
plt.show()

# Calculate reorder point based on current stock
current_stock = daily_inventory['CurrentStock'].iloc[-1]  # Current stock for the last day

# Calculate reorder point for each forecasted day
reorder_points = []
for index, row in forecast_df.iterrows():
    forecasted_demand = row['ForecastedSalesQuantity']
    reorder_point = max(current_stock - forecasted_demand, 0)
    reorder_points.append(reorder_point)

# Add reorder points to the forecast DataFrame
forecast_df['ReorderPoint'] = reorder_points

# Print the result
# print(forecast_df)
print(f"Current Stock: {current_stock}")
print(f"Forecasted Demand: {forecast_df['ForecastedSalesQuantity'].values}")
print(f"Reorder Point: {reorder_point}")

save_model(model1, "lstm_model_SalesQuantity.keras")

loaded_model1 = load_model("lstm_model_SalesQuantity.keras")

# for _ in range(forecast_steps):
#     forecast_value1 = loaded_model1.predict(forecast_input1, verbose=0)
#     # forecast_scaled1.append(forecast_value1[0, 0])
#     forecast_scaled1 = np.concatenate((forecast_scaled1, forecast_value1[0, 0]), axis=None)
#     forecast_input1 = np.roll(forecast_input1, -1)
#     forecast_input1[0, -1, 0] = forecast_value1[0, 0]

# # Inverse transform the forecasted values
# forecast_scaled1 = np.array(forecast_scaled1).reshape(-1, 1)
# forecast1 = scaler.inverse_transform(forecast_scaled1)

# # Create a DataFrame with the forecasted values
# forecast_dates = pd.date_range(df_daily.index[-1] + pd.DateOffset(1), periods=forecast_steps1, freq=df_daily.index.freq)
# forecast_df = pd.DataFrame({'SalesDate': forecast_dates, 'ForecastedSalesQuantity': forecast1.flatten()})
# forecast_df

