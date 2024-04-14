# In[1]:
# !pip3 install matplotlib
# In[2]:
# !pip3 install seaborn
# In[5]:
#Import libraries
import numpy as np         #Mathematical Computation
import pandas as pd    # Data Manipulation
import psycopg2
# In[6]:
#Purchase Price
df= pd.read_csv('2017PurchasePricesDec.csv')
df.head()


# In[7]:


df.describe(include='all')


# In[8]:


#Inventory Begning Final
df1= pd.read_csv('BegInvFINAL12312016.csv')
df1.head()


# In[9]:


df1.City.nunique()


# In[9]:


df1.shape


# In[10]:


#Inventory Ending Final
df2= pd.read_csv('EndInvFINAL12312016.csv')
df2.head()


# In[11]:


df2.shape


# In[12]:


#Invoice Purchases
df3= pd.read_csv('InvoicePurchases12312016.csv')
df3.head()


# In[13]:


#Purchase Data
df4= pd.read_csv('PurchasesFINAL12312016.csv')
df4.head()


# In[14]:


## Final Sales Data.
df5= pd.read_csv('SalesFINAL12312016.csv')
df5.head()


# In[15]:


print(df.shape)
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
print(df5.shape)


# In[16]:


df.shape[0]+df1.shape[0]+df2.shape[0]+df3.shape[0]+df4.shape[0]+df5.shape[0]


# In[17]:


from IPython.display import display
from IPython.display import Javascript

display(Javascript('IPython.OutputArea.auto_scroll_threshold = 40000000;'))


# In[18]:


pd.set_option('display.max_rows', 40000000)
pd.set_option('display.max_columns', None)

# c.NotebookApp.iopub_msg_rate_limit = 10000000.0  # Set your desired limit here
# Set the IOPub message rate limit
# %config NotebookApp.iopub_msg_rate_limit=10000000.0


# In[15]:


# df


# In[22]:


from sqlalchemy import create_engine
from sqlalchemy import text

# Replace with your PostgreSQL database connection details
DATABASE_URL = "postgresql://postgres:Antim%40311997@localhost:5432/inventory"


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



# In[25]:

# session.rollback()

# Assuming you have DataFrames named df_beg_inv, df_end_inv, df_invoice_purchases, df_purchases, df_sales
df.to_sql(purchase_prices_table_name, con=engine, if_exists='replace', index=False)
df1.to_sql(beg_inv_table_name, con=engine, if_exists='replace', index=False)
df2.to_sql(end_inv_table_name, con=engine, if_exists='replace', index=False)
df3.to_sql(invoice_purchases_table_name, con=engine, if_exists='replace', index=False)
df4.to_sql(purchases_table_name, con=engine, if_exists='replace', index=False)
df5.to_sql(sales_table_name, con=engine, if_exists='replace', index=False)


# In[26]:


# Confirm the table creation and data import
with engine.connect() as connection:
    result = connection.execute(text(f"SELECT * FROM {purchase_prices_table_name}"))
    for row in result:
        print(row)

    result = connection.execute(text(f"SELECT * FROM {beg_inv_table_name}"))
    for row in result:
        print(row)


# In[ ]:


# Establish connection to PostgreSQL database
conn = psycopg2.connect(
    dbname="inventory",
    user="postgres",
    password="Antim@311997",
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




# In[ ]:




