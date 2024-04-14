from fastapi.templating import Jinja2Templates
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import io
from fastapi import APIRouter, Depends, File, Request, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sklearn.impute import SimpleImputer
from routes.jwt_bearer import jwtBearer
import time

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
    
    return df  # Return the modified DataFrame

@router.post('/ABC')
async def ABC(file: UploadFile = File(...)):
    # Check if the file is an Excel file
    print("auth------")
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(io.BytesIO(await file.read()))
    df = handle_missing_values(df)

    # Perform your calculations    
    temp = df[['InventoryId', 'SalesPrice']]
    total_sales = temp['SalesPrice'].sum()
    temp.loc[:, 'Contribution'] = temp['SalesPrice'] / total_sales
    temp.loc[:, 'ABC_Classification'] = pd.qcut(temp['Contribution'], q=[0, 0.2, 0.8, 1], labels=['A', 'B', 'C'])


    # Generate the bar plot
    temp_plot = temp[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30)
    ax = temp_plot.plot(kind='barh', fontsize=15, figsize=(25,25), stacked=True, color='green')



    # Calculate percentages
    total_count = len(temp)
    percentages = temp[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30) / total_count * 100
    
    # Annotate the bars with percentages
    for index, value in enumerate(temp[['InventoryId', 'ABC_Classification']].value_counts().sort_values(ascending=False).head(30)):
        plt.text(value, index, f'{value} ({percentages.iloc[index]:.2f}%)', va='center', fontsize=12, color='black')

    # Save the plot as a PDF using matplotlib
    plt.savefig('plot.pdf', format='pdf')
    plt.close()

    # Write results to a PDF using pandas
    results_pdf_bytes = io.BytesIO()
    temp[['InventoryId', 'ABC_Classification']].to_csv(results_pdf_bytes, index=False)
    results_pdf_bytes.seek(0)

    # Combine the plot and results PDFs into one file
    # combined_pdf_bytes = io.BytesIO()

    combined_pdf_path = 'combined_report.pdf'
    with open('plot.pdf', 'rb') as plot_file, open(combined_pdf_path, 'wb') as combined_file:
        combined_file.write(plot_file.read())
        combined_file.write(results_pdf_bytes.read())

    # Return the combined PDF file
    return FileResponse(combined_pdf_path, filename='ABC_Report.pdf')

