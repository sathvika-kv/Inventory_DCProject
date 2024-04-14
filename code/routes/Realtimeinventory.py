import os
import tempfile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import Request, UploadFile, File, HTTPException, APIRouter, Depends, Response
import io
from routes.jwt_bearer import jwtBearer
from sklearn.impute import SimpleImputer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage
import base64
import tempfile

router = APIRouter()


def handle_missing_values(df):
    # Check for missing values
    missing_values = df.isnull().sum()

    # Check if there are any missing values
    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print("Missing values found. Handling missing values...")
        
        # Separate numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Impute missing values for numerical columns
        if not numeric_cols.empty:
            numeric_imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

        # Impute missing values for categorical columns
        if not categorical_cols.empty:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

        print("Missing values handled.")

        # Convert numerical columns to appropriate data types
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Convert categorical columns to appropriate data types
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.as_ordered()

        print("Data type conversion completed.")
    
    return df

@router.post('/realtimeinventory')
async def RealtimeinventoryAnalysis(file: UploadFile = File(...)):
    # Check if the file is an Excel file
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading Excel file: " + str(e))

    df = handle_missing_values(df)

    # Filter relevant columns
    temp = df[['SalesDate', 'PODate', 'InvoiceDate', 'SalesQuantity', 'Brand', 'Description', 'SalesPrice','ReceivingDate',
               'startDate', 'endDate', 'begonHand', 'endonHand','City', 'VendorName', 'PurchasePrice']]

    # Grouping and summarizing inventory for beginning and end of the year
    beg_summary = temp.groupby(['Brand', 'Description', 'SalesPrice', 'startDate'])['begonHand'].sum().sort_values(ascending=False)
    end_summary = temp.groupby(['Brand', 'Description', 'SalesPrice', 'endDate'])['endonHand'].sum().sort_values(ascending=False)

    # Identifying top 5 products at the beginning and end of the year
    top5_beg = beg_summary.head(5)
    top5_end = end_summary.head(5)

    # Plotting top 5 products at the beginning and end of the year
    plt.figure(figsize=(6, 6))
    top5_beg.plot(kind='pie', autopct='%1.1f%%', fontsize=20)
    beg_plot_bytes = io.BytesIO()
    plt.savefig(beg_plot_bytes, format='png')
    plt.close()

    plt.figure(figsize=(6,6))
    top5_end.plot(kind='pie', autopct='%1.1f%%', fontsize=20)
    end_plot_bytes = io.BytesIO()
    plt.savefig(end_plot_bytes, format='png')
    plt.close()

    # Real-time overall Inventory Monitoring
    current_stock = temp['begonHand'].sum()
    print(f"Current stock level: {current_stock}")

    # Current stock at the beginning and end of the year in every city
    plt.figure(figsize=(6, 6))
    temp.groupby(['City'])['begonHand'].sum().plot(kind='barh')
    city_beg_plot_bytes = io.BytesIO()
    plt.savefig(city_beg_plot_bytes, format='png')
    plt.close()

    plt.figure(figsize=(6,6))
    temp.groupby(['City'])['endonHand'].sum().plot(kind='barh')
    city_end_plot_bytes = io.BytesIO()
    plt.savefig(city_end_plot_bytes, format='png')
    plt.close()

    # Plotting Top 10 Vendors by Purchase Cost
    reduced_purchases = temp[['VendorName', 'PurchasePrice']]
    top_vendors = reduced_purchases.groupby('VendorName').sum()['PurchasePrice'].nlargest(10)
    plt.figure(figsize=(6, 6))
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
    vendor_plot_bytes = io.BytesIO()
    plt.savefig(vendor_plot_bytes, format='png')
    plt.close()

    #  Convert 'SalesDate' to datetime type
    temp['SalesDate'] = pd.to_datetime(temp['SalesDate'])
    start_date = temp['SalesDate'].min()
    end_date = temp['SalesDate'].max()
    total_days = (end_date - start_date).days ## Number of total days

    ## Datetime conversion
    temp['ReceivingDate'] = pd.to_datetime(temp['ReceivingDate'])
    temp['PODate'] = pd.to_datetime(temp['PODate'],format='%Y-%m-%d')
    ##Calculate lead time
    temp['LeadTime'] = (temp['ReceivingDate'] - temp['PODate']).dt.days

    # Calculating Sales Velocity for each product
    sales_velocity = temp.groupby(['Brand', 'Description']).agg(Total_Sales=('SalesQuantity', 'sum')).reset_index()
    sales_velocity['Sales_Per_Day'] = sales_velocity['Total_Sales'] / total_days
    lead_times = temp.groupby(['Brand', 'Description']).agg(Avg_Lead_Time=('LeadTime', 'mean')).reset_index()

    # Merging the data
    merge_df = pd.merge(sales_velocity, lead_times, on=['Brand', 'Description'], how='left')

    # Calculating Optimal Stock Level
    merge_df['Optimal_Stock_Level'] = merge_df['Sales_Per_Day'] * merge_df['Avg_Lead_Time']

    # Calculating Safety Stock using maximum sales for each product
    max_sales = temp.groupby(['Brand', 'Description']).agg(Max_Daily_Sales=('SalesQuantity', 'max')).reset_index()
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
    plt.figure(figsize=(6, 6))
    sns.barplot(x='Recommended_Stock_Level', y='Description', data=sorted_data.head(20), palette='magma')

    # Add percentage annotations to the bars
    total_recstock = sorted_data['Recommended_Stock_Level'].head(20).sum()
    for index, value in enumerate(sorted_data['Recommended_Stock_Level'].head(20)):
        percentage = (value / total_recstock) * 100
        plt.text(value, index, f'{percentage:.2f}%', va='center', fontsize=12, color='blue')

    plt.tight_layout()
    recstock_plot_bytes = io.BytesIO()
    plt.savefig(recstock_plot_bytes, format='png')
    plt.close()

    # Sorting the data by Sales_Per_Day in descending order to get top products
    top_products = sales_velocity.sort_values(by='Sales_Per_Day', ascending=False)

    # Creating the bar plot
    plt.figure(figsize=(6, 6))
    sns.barplot(y='Description', x='Sales_Per_Day', data=top_products.head(20), palette='magma')

    # Add percentage annotations to the bars
    total_sales = top_products['Sales_Per_Day'].head(20).sum()
    for index, value in enumerate(top_products['Sales_Per_Day'].head(20)):
        percentage = (value / total_sales) * 100
        plt.text(value, index, f'{percentage:.2f}%', va='center', fontsize=12, color='blue')

    plt.tight_layout()
    salesvel_plot_bytes = io.BytesIO()
    plt.savefig(salesvel_plot_bytes, format='png')
    plt.close()

    # Optimal stock level
    temp['endDate'] = pd.to_datetime(temp['endDate'])
    latest_inventory_date = temp['endDate'].max()
    current_inventory = temp[temp['endDate'] == latest_inventory_date]

    # Summarizing the current stock levels by product
    current_stock_levels = current_inventory.groupby(['Brand', 'Description']).agg(Current_Stock=('begonHand', 'sum')).reset_index()

    # Merging the current stock levels with the previously calculated data
    final_data = pd.merge(merge_df, current_stock_levels, on=['Brand', 'Description'], how='left')

    # Assume zero current stock for any products not present in the current inventory
    final_data['Current_Stock'] = final_data['Current_Stock'].fillna(0)

    # Calculating how much of each product needs to be ordered if current stock is below recommended levels
    final_data['Order_Quantity'] = final_data['Recommended_Stock_Level'] - final_data['Current_Stock']
    final_data['Order_Quantity'] = final_data['Order_Quantity'].clip(lower=0)  # Setting negative order quantities to zero

    # Reporting the results
    print(final_data[['Brand', 'Description', 'Current_Stock', 'Recommended_Stock_Level', 'Order_Quantity']])

    # Plotting top 10 products to reorder
    plt.figure(figsize=(6, 6))
    sns.barplot(x='Order_Quantity', y='Description', data=final_data.sort_values('Order_Quantity', ascending=False).head(10))
    top10_reorder_plot_bytes = io.BytesIO()
    plt.savefig(top10_reorder_plot_bytes, format='png')
    plt.close()
 
    with tempfile.TemporaryDirectory() as temp_dir:
        output_pdf_path = os.path.join(temp_dir, 'inventory_analysis.pdf')
        doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Plotting and adding plots to the PDF
        for i, plot_data in enumerate([
            (top5_beg, 'Top 5 Products at Beginning of Year', 'pie'),
            (top5_end, 'Top 5 Products at End of Year', 'pie'),
            (temp.groupby(['City'])['begonHand'].sum(), 'Current Stock by City (Beginning)', 'barh'),
            (temp.groupby(['City'])['endonHand'].sum(), 'Current Stock by City (End)', 'barh'),
            (top_vendors, 'Top 10 Vendors by Purchase Cost', 'barh'),
            (sorted_data.head(20), 'Recommended Stock Level', 'bar'),
            (top_products.head(20), 'Sales Velocity', 'bar'),
            (final_data.sort_values('Order_Quantity', ascending=False).head(10), 'Top 10 Products to Reorder', 'bar')
        ]):
            plt.figure(figsize=(6, 6))
            plot_data[0].plot(kind=plot_data[2], fontsize=15)
            plt.title(plot_data[1], fontsize=15)
            plt.tight_layout()
            plot_path = os.path.join(temp_dir, f'plot_{i}.png')
            plt.savefig(plot_path)
            plt.close()

            story.append(Image(plot_path, width=500, height=500))
            story.append(PageBreak())
        
        doc.build(story)

        # Return the PDF file as a response
        with open(output_pdf_path, "rb") as file:
            pdf_bytes = file.read()

        # Inside your RealtimeinventoryAnalysis function
        response = Response(pdf_bytes, media_type='application/pdf')
        response.headers['Content-Disposition'] = 'attachment; filename=inventory_analysis.pdf'
        return response
