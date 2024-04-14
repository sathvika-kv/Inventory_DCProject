from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from routes.auth import decodeJWT

class jwtBearer(HTTPBearer):
    def __init__(self, auto_Error: bool = True):
        super(jwtBearer, self).__init__(auto_error=auto_Error)
        print("in jwt bearer")

    async def __call__(self, request: Request):
        print("in jwt bearer2")
        print(request)
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        print("credit")
        if credentials:
            print(credentials.credentials)
            if not credentials.scheme == "Bearer":
                print("step 1")
                raise HTTPException(status_code=403, detail="Invalid or Expired Token")
            if not self.verify_jwt(credentials.credentials):
                print("step 2")
                raise HTTPException(status_code=403, detail="Invalid token or expired token.")
            return credentials.credentials
        else:
            print("step 3")
            raise HTTPException(status_code=403, detail="Invalid or Expired Token")

    def verify_jwt(self, jwtoken: str):
        isTokenValid: bool = False
        try:
            print("payload")
            payload = decodeJWT(jwtoken)
            print(payload)
            # Check if the payload contains the required fields (email and session_id)
            if "email" in payload and "session_id" in payload:
                # Here you can perform additional checks if needed
                isTokenValid = True
            print("nothing")
        except:
            payload = None
        return isTokenValid
