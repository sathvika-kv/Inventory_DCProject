# - **First Set of Variables:**
# 
#     Brand: Represents the brand of the product.
# 
#     Description: Describes the product.
# 
#     Price: Indicates the price of the product.
# 
#     Size: Specifies the size of the product.
# 
#     Volume: Represents the volume of the product.
# 
#     Classification: Classifies the product.
# 
#     PurchasePrice: Indicates the purchase price of the product.
# 
#     VendorNumber: Represents the vendor's identification number.
# 
#     VendorName: Specifies the name of the vendor.
#     
# 
# - **Second Set of Variables:**
#     
#     InventoryId: Identifies the inventory item.
#     
#     Store: Represents the store where the inventory item is located.
#     
#     City: Specifies the city associated with the inventory item.
#     
#     Brand: Represents the brand of the product.
#     
#     Description: Describes the product.
#     
#     Size: Specifies the size of the product.
#     
#     onHand: Indicates the quantity of the item currently in stock.
#     
#     Price: Indicates the price of the product.
#     
#     startDate: Represents the start date associated with the inventory record.
# 
# 
# - **Third Set of Variables:**
#     
#     InventoryId: Identifies the inventory item.
#     
#     Store: Represents the store where the inventory item is located.
#     
#     City: Specifies the city associated with the inventory item.
#     
#     Brand: Represents the brand of the product.
#     
#     Description: Describes the product.
#     
#     Size: Specifies the size of the product.
#     
#     onHand: Indicates the quantity of the item currently in stock.
#     
#     Price: Indicates the price of the product.
#     
#     endDate: Represents the end date associated with the inventory record.
# 
# 
# - **Fourth Set of Variables:**
#     
#     VendorNumber: Represents the vendor's identification number.
#     
#     VendorName: Specifies the name of the vendor.
#     
#     InvoiceDate: Indicates the date of the invoice.
#     
#     PONumber: Represents the purchase order number.
#     
#     PODate: Indicates the date of the purchase order.
#     
#     PayDate: Represents the date of payment.
#     
#     Quantity: Specifies the quantity of items.
#     
#     Dollars: Represents the monetary value.
#     
#     Freight: Indicates the cost of freight.
#     
#     Approval: Indicates whether the transaction is approved.
# 
# 
# - **Fifth Set of Variables:**
#     
#     InventoryId: Identifies the inventory item.
#     
#     Store: Represents the store where the inventory item is located.
#     
#     Brand: Represents the brand of the product.
#     
#     Description: Describes the product.
#     
#     Size: Specifies the size of the product.
#     
#     VendorNumber: Represents the vendor's identification number.
#     
#     VendorName: Specifies the name of the vendor.
#     
#     PONumber: Represents the purchase order number.
#     
#     PODate: Indicates the date of the purchase order.
#     
#     ReceivingDate: Represents the date of receiving the inventory.
#     
#     InvoiceDate: Indicates the date of the invoice.
#     
#     PayDate: Represents the date of payment.
#     
#     PurchasePrice: Indicates the purchase price of the product.
#     
#     Quantity: Specifies the quantity of items.
#     
#     Dollars: Represents the monetary value.
#     
#     Classification: Classifies the product.
# 
# 
# - **Sixth Set of Variables:**
#     
#     InventoryId: Identifies the inventory item.
#     
#     Store: Represents the store where the inventory item is located.
#     
#     Brand: Represents the brand of the product.
#     
#     Description: Describes the product.
#     
#     Size: Specifies the size of the product.
#     
#     SalesQuantity: Indicates the quantity of items sold.
#     
#     SalesDollars: Represents the monetary value of sales.
#     
#     SalesPrice: Indicates the price at which the product was sold.
#     
#     SalesDate: Represents the date of the sale.
#     
#     Volume: Represents the volume of the product.
#     
#     Classification: Classifies the product.
#     
#     ExciseTax: Indicates any excise tax associated with the product.
#     
#     VendorNo: Represents the vendor's identification number.
#     
#     VendorName: Specifies the name of the vendor.
# 
# These variables provide detailed information about the products, vendors, purchases, and sales, which will be crucial for the inventory analysis tasks outlined in the case study.
# 
# 
# ## Objectives/pain address:
# 1. Inaccurate stock levels leading to stock-outs and excess inventory.
#     - o	Solution: Implement machine learning algorithms for demand forecasting to determine optimal inventory levels for raw materials, work-in-progress, and finished goods.
#     
# 2. Increased carrying costs due to frequent stock-outs and excess inventory.
#     - Solution: Utilize AI-driven reorder point analysis and economic order quantity (EOQ) analysis for effective replenishment and cost minimization.
# 3. Lack of insights into inventory turnover and associated carrying costs.
#     - Solution: Apply deep learning models to analyze historical data, calculating inventory turnover ratios and identifying areas for cost reduction.
# 4. Product unavailability affecting customer satisfaction.
#     - Solution: Implement AI-driven demand forecasting and inventory optimization to ensure product availability, thereby improving customer satisfaction.
# To address the objectives and pain points mentioned, you can apply various methods and strategies on the provided variables. Here are some suggestions:
# 1. Inaccurate stock levels leading to stock-outs and excess inventory:
# 
#     Implement Real-time Inventory Monitoring:
#         Use the onHand variable to monitor current stock levels in real-time.
#         Utilize automated systems to update stock levels regularly.
# 
#     Set Reorder Points:
#         Analyze historical sales data (SalesQuantity, SalesDate, InventoryId) to establish optimal reorder points.
#         Implement alerts or automatic reorder processes when stock levels reach the specified thresholds.
# 
#     Forecasting Models:
#         Utilize historical sales data to create forecasting models (SalesQuantity, SalesDate, InventoryId) for future demand predictions.
#         Adjust inventory levels based on forecasted demand.
# 
# 2. Increased carrying costs due to frequent stock-outs and excess inventory:
# 
#     Optimize Order Quantities:
#         Analyze purchase data (Quantity, Dollars, InvoiceDate, PONumber) to optimize order quantities.
#         Balance the costs associated with carrying excess inventory against potential stock-out costs.
# 
#     Just-In-Time Inventory (JIT):
#         Implement JIT inventory practices to minimize excess stock.
#         Use historical data (SalesDate, SalesQuantity, InventoryId) to align deliveries with actual demand.
# 
#     Carrying Cost Analysis:
#         Evaluate carrying costs based on historical data (InventoryId, onHand, Price).
#         Identify and reduce unnecessary holding costs.
# 
# 3. Lack of insights into inventory turnover and associated carrying costs:
# 
#     Inventory Turnover Ratio:
#         Calculate inventory turnover ratio using SalesQuantity and onHand.
#         Regularly monitor and analyze the turnover ratio to identify trends.
# 
#     Carrying Cost Analysis:
#         Analyze historical data (InventoryId, onHand, Price) to calculate carrying costs.
#         Identify high-cost items or slow-moving inventory for optimization.
# 
#     ABC Analysis:
#         Classify products based on their importance and contribution to revenue using variables like SalesDollars and SalesQuantity.
#         Allocate resources and attention accordingly.
# 
# 4. Product unavailability affecting customer satisfaction:
# 
#     Stock Level Alerts:
#         Implement automatic alerts when stock levels (onHand) fall below a certain threshold.
#         Ensure timely replenishment to prevent stock-outs.
# 
#     Backorder System:
#         Implement a backorder system for temporarily out-of-stock items.
#         Keep customers informed about restocking dates.
# 
#     Customer Feedback Analysis:
#         Analyze customer feedback and complaints related to product availability.
#         Use insights to improve inventory management strategies.
# 
# These methods can help address the specified objectives and alleviate the pain points related to inventory management. It's essential to continually monitor and adjust these strategies based on evolving business needs and market conditions.

#Import libraries
import numpy as np         #Mathematical Computation
import pandas as pd        # Data Manipulation
import psycopg2
from IPython.display import display
from IPython.display import Javascript
from sqlalchemy import create_engine
from sqlalchemy import text

#Purchase Price
df= pd.read_csv('2017PurchasePricesDec.csv')
df.head()
df.describe(include='all')

#Inventory Begning Final
df1= pd.read_csv('BegInvFINAL12312016.csv')
df1.head()
df1.City.nunique()
df1.shape

#Inventory Ending Final
df2= pd.read_csv('EndInvFINAL12312016.csv')
df2.head()
df2.shape

#Invoice Purchases
df3= pd.read_csv('InvoicePurchases12312016.csv')
df3.head()

#Purchase Data
df4= pd.read_csv('PurchasesFINAL12312016.csv')
df4.head()

## Final Sales Data
df5= pd.read_csv('SalesFINAL12312016.csv')
df5.head()

print(df.shape)
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
print(df5.shape)

df.shape[0]+df1.shape[0]+df2.shape[0]+df3.shape[0]+df4.shape[0]+df5.shape[0]

display(Javascript('IPython.OutputArea.auto_scroll_threshold = 40000000;'))

pd.set_option('display.max_rows', 40000000)
pd.set_option('display.max_columns', None)

# Replace with your PostgreSQL database connection details
DATABASE_URL = "postgresql://postgres:12345678@localhost:5432/Stock"

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Define the table names
purchase_prices_table_name = 'purchase_prices'
beg_inv_table_name = 'beg_inv'
end_inv_table_name = 'end_inv'
invoice_purchases_table_name = 'invoice_purchases'
purchases_table_name = 'purchases'
sales_table_name = 'sales'

# Create tables in PostgreSQL
with engine.connect() as connection:
    # Purchase Prices
    connection.execute(text(f'''
        CREATE TABLE IF NOT EXISTS {purchase_prices_table_name} (
            Brand INT,
            Description VARCHAR(255),
            Price FLOAT,
            Size VARCHAR(255),
            Volume VARCHAR(255),
            Classification INT,
            PurchasePrice FLOAT,
            VendorNumber INT,
            VendorName VARCHAR(255)
        );
    '''))

    # Beginning Inventory
    connection.execute(text(f'''
        CREATE TABLE IF NOT EXISTS {beg_inv_table_name} (
            InventoryId VARCHAR(255),
            Store INT,
            City VARCHAR(255),
            Brand INT,
            Description VARCHAR(255),
            Size VARCHAR(255),
            onHand INT,
            Price FLOAT,
            startDate DATE
        );
    '''))

    # Repeat the above lines for other tables
    # End Inventory
    connection.execute(text(f'''
        CREATE TABLE IF NOT EXISTS {end_inv_table_name} (
            InventoryId VARCHAR(255),
            Store INT,
            City VARCHAR(255),
            Brand INT,
            Description VARCHAR(255),
            Size VARCHAR(255),
            onHand INT,
            Price FLOAT,
            endDate DATE
        );
    '''))

    # Invoice Purchases
    connection.execute(text(f'''
        CREATE TABLE IF NOT EXISTS {invoice_purchases_table_name} (
            VendorNumber INT,
            VendorName VARCHAR(255),
            InvoiceDate DATE,
            PONumber INT,
            PODate DATE,
            PayDate DATE,
            Quantity INT,
            Dollars FLOAT,
            Freight FLOAT,
            Approval VARCHAR(255)
        );
    '''))

    # Purchases
    connection.execute(text(f'''
        CREATE TABLE IF NOT EXISTS {purchases_table_name} (
            InventoryId VARCHAR(255),
            Store INT,
            Brand INT,
            Description VARCHAR(255),
            Size VARCHAR(255),
            VendorNumber INT,
            VendorName VARCHAR(255),
            PONumber INT,
            PODate DATE,
            ReceivingDate DATE,
            InvoiceDate DATE,
            PayDate DATE,
            PurchasePrice FLOAT,
            Quantity INT,
            Dollars FLOAT,
            Classification INT
        );
    '''))

    # Sales
    connection.execute(text(f'''
        CREATE TABLE IF NOT EXISTS {sales_table_name} (
            InventoryId VARCHAR(255),
            Store INT,
            Brand INT,
            Description VARCHAR(255),
            Size VARCHAR(255),
            SalesQuantity INT,
            SalesDollars FLOAT,
            SalesPrice FLOAT,
            SalesDate DATE,
            Volume INT,
            Classification INT,
            ExciseTax FLOAT,
            VendorNo INT,
            VendorName VARCHAR(255)
        );
    '''))

# Dispose the engine to close the database connection
engine.dispose()

# Assuming you have DataFrames named df_beg_inv, df_end_inv, df_invoice_purchases, df_purchases, df_sales
df.to_sql(purchase_prices_table_name, con=engine, if_exists='replace', index=False)
df1.to_sql(beg_inv_table_name, con=engine, if_exists='replace', index=False)
df2.to_sql(end_inv_table_name, con=engine, if_exists='replace', index=False)
df3.to_sql(invoice_purchases_table_name, con=engine, if_exists='replace', index=False)
df4.to_sql(purchases_table_name, con=engine, if_exists='replace', index=False)
df5.to_sql(sales_table_name, con=engine, if_exists='replace', index=False)

# Confirm the table creation and data import
with engine.connect() as connection:
    result = connection.execute(text(f"SELECT * FROM {purchase_prices_table_name}"))
    for row in result:
        print(row)

    result = connection.execute(text(f"SELECT * FROM {beg_inv_table_name}"))
    for row in result:
        print(row)

# Establish connection to PostgreSQL database
conn = psycopg2.connect(
    dbname="Stock",
    user="postgres",
    password="122345678",
    host="localhost",
    port="5432"
)

# Create a cursor object
cursor = conn.cursor()

# Create the users table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    fullname VARCHAR(255),
    user_activity DOUBLE PRECISION
)
"""
cursor.execute(create_table_query)
conn.commit()

# Close cursor and connection
cursor.close()
conn.close()
