import base64
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from .jwt_bearer import jwtBearer
from fastapi import APIRouter, Request, Response, UploadFile, File, Depends, HTTPException
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors

from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.impute import SimpleImputer
from pydantic import BaseModel
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import io
import json

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
        df[col] = df[col].astype('category')

    print("Data type conversion completed.")
    
    return df



# Modify the path to your trained LSTM model
lstm_model_sales_quantity = load_model("lstm_model_SalesQuantity.keras")

@router.post('/SalesQuantityForecast', dependencies=[Depends(jwtBearer())])
async def salesQuantitypred(file: UploadFile = File(...)):
    # Check if the file is an Excel file
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading Excel file: " + str(e))

    # Extract relevant columns and process data
    temp = df[['SalesDate', 'startDate', 'begonHand', 'SalesQuantity']]
    temp['SalesDate'] = pd.to_datetime(temp['SalesDate'])
    temp['startDate'] = pd.to_datetime(temp['startDate'])

    # Resample to daily frequency
    temp = temp.set_index('SalesDate').resample('D').mean()

    # Data preprocessing
    scaler = StandardScaler()
    temp_scaled = scaler.fit_transform(temp[['SalesQuantity']])

    # Create sequences for LSTM
    sequence_length = 10  # Choose an appropriate sequence length
    X, y = [], []
    for i in range(len(temp_scaled) - sequence_length):
        X.append(temp_scaled[i:i+sequence_length, 0])
        y.append(temp_scaled[i+sequence_length, 0])

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM input shape (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Forecast future values
    forecast_steps = 10  # Adjust the number of steps as needed
    forecast_input = temp_scaled[-sequence_length:, 0]
    forecast_input = np.reshape(forecast_input, (1, sequence_length, 1))
    forecast_scaled = []

    # Make predictions using the loaded model
    for _ in range(forecast_steps):
        forecast_value = lstm_model_sales_quantity.predict(forecast_input, verbose=0)
        forecast_scaled.append(forecast_value[0, 0])
        forecast_input = np.roll(forecast_input, -1)
        forecast_input[0, -1, 0] = forecast_value[0, 0]

    # Inverse transform the forecasted values
    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled)

    # Create a DataFrame with the forecasted values
    forecast_dates = pd.date_range(temp.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq=temp.index.freq)
    forecast_df = pd.DataFrame({'SalesDate': forecast_dates, 'ForecastedSalesQuantity': forecast.flatten()})

    # Calculate current stock for each day
    temp['CurrentStock'] = temp['begonHand'].cumsum()
    temp_reset = temp.reset_index()
    merged_df = pd.merge(temp_reset[['SalesDate', 'SalesQuantity']], forecast_df, on='SalesDate', how='outer')

    # Plot the original time series and the forecast
    plt.figure(figsize=(6, 6))
    plt.plot(merged_df['SalesDate'], merged_df['SalesQuantity'], label='Actual SalesQuantity', marker='o')
    plt.plot(merged_df['SalesDate'], merged_df['ForecastedSalesQuantity'], color='red', label='Forecasted SalesQuantity')
    plt.title('LSTM Model Forecast')
    plt.xlabel('SalesDate')
    plt.ylabel('SalesQuantity')
    plt.legend()

    forecast_plt_bytes = io.BytesIO()
    plt.savefig(forecast_plt_bytes, format='png')
    plt.close()

    # Plot current stock over time
    plt.figure(figsize=(6, 6))
    plt.plot(temp.index, temp['CurrentStock'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Current Stock')
    plt.title('Current Stock Over Time')
    plt.grid(True)
    plt.show()
    currentstock_plt_bytes = io.BytesIO()
    plt.savefig(currentstock_plt_bytes, format='png')
    plt.close()

    # Calculate reorder point based on current stock
    current_stock = temp['CurrentStock'].iloc[-1]  # Current stock for the last day


    # Calculate reorder point for each forecasted day
    reorder_points = []
    for index, row in forecast_df.iterrows():
        forecasted_demand = row['ForecastedSalesQuantity']
        if pd.notna(forecasted_demand):  # Check for missing values
            reorder_point = max(current_stock - forecasted_demand, 0)
            reorder_points.append(reorder_point)
        else:
            reorder_points.append(0)  # Handle missing values


    # Add reorder points to the forecast DataFrame
    forecast_df['ReorderPoint'] = reorder_points

    # Inside your FastAPI endpoint function
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as forecast_plot_image, \
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as reorder_point_plot_image:

        # Plot the original time series and the forecast
        plt.figure(figsize=(6, 6))
        plt.plot(temp.index, temp['SalesQuantity'], label='Actual SalesQuantity', marker='o')
        plt.plot(forecast_df['SalesDate'], forecast_df['ForecastedSalesQuantity'], color='red', label='Forecasted SalesQuantity')
        plt.title('LSTM Model Forecast')
        plt.xlabel('SalesDate')
        plt.ylabel('SalesQuantity')
        plt.legend()
        plt.savefig(forecast_plot_image.name, format='png')
        plt.close()

        # Plot reorder points over time
        plt.figure(figsize=(6, 6))
        plt.plot(forecast_df['SalesDate'], forecast_df['ReorderPoint'], marker='o')
        plt.xlabel('Date')
        plt.ylabel('Reorder Point')
        plt.title('Reorder Point Over Time')
        plt.savefig(reorder_point_plot_image.name, format='png')
        plt.close()

        # Generate PDF
        output_pdf = io.BytesIO()
        doc = SimpleDocTemplate(output_pdf, pagesize=letter)
        styles = getSampleStyleSheet()

        elements = []

        # Add forecast plot image to PDF
        elements.append(Paragraph("Forecast Plot", styles['Title']))
        forecast_plot_image.seek(0)
        forecast_plot_image_data = Image(forecast_plot_image.name)
        elements.append(forecast_plot_image_data)
        elements.append(PageBreak())

        # Add reorder point plot image to PDF
        elements.append(Paragraph("Reorder Point Plot", styles['Title']))
        reorder_point_plot_image.seek(0)
        reorder_point_plot_image_data = Image(reorder_point_plot_image.name)
        elements.append(reorder_point_plot_image_data)
        elements.append(PageBreak())

        # Add forecasted values table to PDF
        elements.append(Paragraph("Forecasted Values", styles['Title']))

        # Convert forecast DataFrame to list of lists
        forecast_data = [['SalesDate', 'ForecastedSalesQuantity']]
        for index, row in forecast_df.iterrows():
            forecast_data.append([row['SalesDate'], row['ForecastedSalesQuantity']])

        # Create a table for forecasted values
        forecast_table = Table(forecast_data)
        forecast_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

        # Add the forecast table to PDF
        elements.append(forecast_table)
        elements.append(Spacer(1, 12))  # Add some space after the table


        doc.build(elements)

        output_pdf.seek(0)

        # Set response headers for PDF file
        response = Response(content=output_pdf.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=SalesQuantityForecast.pdf"
        response.headers["Content-Type"] = "application/pdf"

        return response
