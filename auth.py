# This file is responsible for signing,encoding,decoding and return JWT token.
# This file is responsible for signing,encoding,decoding and return JWT token.
import time
import jwt
from fastapi import  Depends, HTTPException

# from decouple import config
import pandas as pd

# JWT_SECRET = config("secret")
# JWT_ALGORITHM = config("algorithm")

JWT_SECRET = 'ekQuJRWOLc2g93nV18mSTF94apf4QaIjURNJEpL4hpg'
JWT_ALGORITHM = 'HS256'

def token_response(token: str):
    return {
        "access_token": token
    }

def signJWT(userID: str, email: str):
    payload = {
        "userID": userID,
        "email": email,
        "expiry": time.time() + 3000  # Example expiry time, adjust as needed
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token_response(token)



# Function
def decodeJWT(token: str):
    try:
        decode_token = jwt.decode(token, JWT_SECRET, algorithms=JWT_ALGORITHM)
        df = pd.read_csv("user.csv")
        for i in range(0, len(df)):
            if df['email'][i] == decode_token["userID"]:
                last_activity_user = df['user_activity'][i]
                if (time.time() - last_activity_user) <= 600:
                    df['user_activity'][i] = time.time()
        df.to_csv("user.csv", index=False, header=True)
        return decode_token if (time.time() - last_activity_user) <= 600 else None
        #return decode_token if decode_token["expiry"] >= time.time() else None
    except:
        return {}



# def decodeJWT(token: str):
#     try:
#         decode_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#         return decode_token if decode_token["expiry"] >= time.time() else None
#     except jwt.ExpiredSignatureError:
#         return None
#     except jwt.InvalidTokenError:
#         return None

async def get_current_user_email(token: str = Depends(decodeJWT)):
    if token:
        return token.get('email')
    else:
        raise HTTPException(status_code=401, detail="Invalid or expired token")