from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import io
from scipy.optimize import minimize
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Form, File, Request, Response
from routes.jwt_bearer import jwtBearer
from sklearn.impute import SimpleImputer
from typing import List
import pdfkit
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfFileReader
import base64
from reportlab.lib.utils import ImageReader
import os

router = APIRouter()

class InventoryModel(BaseModel):
    Number_of_Purchases: int
    PurchasePrice: float
    Quantity: int
    ordering_cost: float
    holding_cost: float
    unit_cost: float
    ordering_cost_per_order: float

class EOQOutput(BaseModel):
    optimal_order_quantity: float
    plot_image: str  # Change the type to string for base64 encoding

def economic_order_quantity(demand, ordering_cost, holding_cost, unit_cost):
    # Calculate Economic Order Quantity (EOQ)
    optimal_order_quantity = np.sqrt((2 * demand * ordering_cost) / holding_cost)
    
    return optimal_order_quantity
@router.post('/EOQ', response_model=None, dependencies=[Depends(jwtBearer())])
async def EOQ(input: InventoryModel):
    demand = input.Quantity
    ordering_cost = input.ordering_cost_per_order
    holding_cost = input.holding_cost
    unit_cost = input.unit_cost

    # Calculate Economic Order Quantity (EOQ)
    optimal_order_quantity = economic_order_quantity(demand, ordering_cost, holding_cost, unit_cost)

    # Create plot for visualization (optional)
    q_values = np.linspace(1, 2 * optimal_order_quantity, 1000)
    cost_values = ordering_cost + (demand / q_values) * holding_cost + unit_cost * demand

    plt.figure(figsize=(6, 6))
    plt.plot(q_values, cost_values, label='Total Cost', color='blue')
    plt.scatter(optimal_order_quantity, ordering_cost + (demand / optimal_order_quantity) * holding_cost + unit_cost * demand,
                color='red', label=f'Optimal EOQ: {optimal_order_quantity:.2f}', zorder=5)
    plt.title('Economic Order Quantity (EOQ) Analysis')
    plt.xlabel('Order Quantity (Q)')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plot_image_path = "/tmp/plot_image.png"  # Temporary file path
    plt.savefig(plot_image_path, format='png')
    plt.close()

    # Generate PDF
    output_pdf = io.BytesIO()
    c = canvas.Canvas(output_pdf, pagesize=letter)
    c.drawString(100, 750, f"Optimal Order Quantity (EOQ): {optimal_order_quantity:.2f}")
    c.drawString(100, 730, f"Demand: {demand}")
    c.drawString(100, 710, f"Ordering Cost: ${ordering_cost:.2f}")
    c.drawString(100, 690, f"Holding Cost: ${holding_cost:.2f}")
    c.drawString(100, 670, f"Unit Cost: ${unit_cost:.2f}")
    c.drawImage(plot_image_path, 20, 50)  # Adjust the coordinates for image position
    c.save()

    # Set response headers for PDF file
    response = Response(content=output_pdf.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=EOQ_analysis.pdf"
    response.headers["Content-Type"] = "application/pdf"

    # Remove temporary plot image file
    os.remove(plot_image_path)

    return response
