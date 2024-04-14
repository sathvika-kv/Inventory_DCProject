from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import List
from fastapi import APIRouter, Request, UploadFile, File, Depends, HTTPException, Response
from routes.jwt_bearer import jwtBearer
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import io
import pdfkit
import tempfile
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.DEBUG)


class SalesPriceInput(BaseModel):
    SalesDate: str
    SalesPrice: float

class SalesPriceOutput(BaseModel):
    forecast: List[float]
    plot_image: bytes

lstm_model_sales_price = load_model("lstm_model_SalesPrice.keras")

@router.post('/salesprice', response_model=SalesPriceOutput)
async def salesPricepred(file: UploadFile = File(...)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    df = pd.read_excel(io.BytesIO(await file.read()))
    df['SalesDate'] = pd.to_datetime(df['SalesDate'])
    df.set_index('SalesDate', inplace=True)
    print(df.resample('D'))
    #df = df.resample('D').mean()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['SalesPrice']])

    sequence_length = 10
    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i+sequence_length, 0])
        y.append(df_scaled[i+sequence_length, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    forecast_steps = 10
    forecast_input = df_scaled[-sequence_length:, 0]
    forecast_input = np.reshape(forecast_input, (1, sequence_length, 1))
    forecast_scaled = []

    for _ in range(forecast_steps):
        forecast_value = lstm_model_sales_price.predict(forecast_input, verbose=0)
        forecast_scaled.append(forecast_value[0, 0])
        forecast_input = np.roll(forecast_input, -1)
        forecast_input[0, -1, 0] = forecast_value[0, 0]

    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled)

        # Plot original data
    plt.plot(df.index[:len(y)], y, marker='o', color='blue', label='Actual SalesPrice')

    # Plot forecast values
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1)[1:]  # Generate forecast dates
    plt.plot(forecast_dates, forecast.flatten(), marker='o', color='red', label='Forecasted SalesPrice')

    plt.xlabel('Sales Date')
    plt.ylabel('SalesPrice')
    plt.title('Actual and Forecasted SalesPrice')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed


    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_plot_image:
        plot_image_path = temp_plot_image.name
        plt.savefig(plot_image_path, format='png')

    # Generate PDF
    output_pdf = io.BytesIO()
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Add text to PDF
    elements.append(Paragraph(f"<b>Forecasted SalesPrice Over 10 Steps</b>", styles['Heading1']))
    elements.append(Spacer(1, 15))

    # Add plot image to PDF
    plot_image = Image(plot_image_path)
    plot_image.drawHeight = 400
    plot_image.drawWidth = 600
    elements.append(plot_image)

    # Add forecast table to PDF
    forecast_table_data = [['Forecast Date', 'Forecasted SalesPrice']]
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1)[1:]  # Generate forecast dates

    for date, price in zip(forecast_dates, [df['SalesPrice'].iloc[-1] + i * 10 for i in range(1, forecast_steps + 1)]):
        forecast_table_data.append([date.strftime('%Y-%m-%d'), price])

    forecast_table = Table(forecast_table_data)
    forecast_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(forecast_table)
    doc.build(elements)

    output_pdf.seek(0)

    # Set response headers for PDF file
    response = Response(content=output_pdf.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=SalesPriceForecast.pdf"
    response.headers["Content-Type"] = "application/pdf"

    return response
