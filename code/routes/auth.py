import time
import jwt
import pandas as pd
from fastapi import HTTPException

JWT_SECRET = 'ekQuJRWOLc2g93nV18mSTF94apf4QaIjURNJEpL4hpg'
JWT_ALGORITHM = 'HS256'

def token_response(token: str):
    return {
        "access_token": token
    }

def signJWT(email: str):
    session_id = f"{email}_{int(time.time())}"  # Generating a unique session ID
    payload = {
        "email": email,
        "session_id": session_id,  # Include session ID in payload
        "expiry": time.time() + 6000  # Expiry set to 24 hours from the current time
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token_response(token)

def decodeJWT(token: str):
    try:
        print("decode1")
        current_timestamp = time.time()
        sub1 = current_timestamp-current_timestamp
        print(sub1)
        print(current_timestamp)
        decode_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        df = pd.read_csv("user.csv")
        print(decode_token["email"])
        for i in range(len(df)):
            if df['email'][i] == decode_token["email"]:
                last_activity_user = df['user_activity'][i]
                if (last_activity_user):
                    df.at[i, 'user_activity'] = time.time()
                else:
                    raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
                break
        else:
            raise HTTPException(status_code=401, detail="User not found.")
        
        df.to_csv("user.csv", index=False, header=True)
        return decode_token
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
