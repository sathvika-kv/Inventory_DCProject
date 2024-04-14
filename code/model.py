from pydantic import BaseModel, Field, EmailStr
import time

class UserSchema(BaseModel):
    fullname: str = Field(...)
    email: EmailStr = Field(...)
    password: str = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "fullname": "Username",
                "email": "user@example.com",
                "password": "password",
                
            }
        }

class UserLoginSchema(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "password"
            }
        }