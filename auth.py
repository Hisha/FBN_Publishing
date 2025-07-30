import os
import bcrypt
from starlette.middleware.sessions import SessionMiddleware
from fastapi import Request, HTTPException, status

ADMIN_HASH = os.getenv("ADMIN_PASSWORD_HASH")

def verify_password(password: str) -> bool:
    if not ADMIN_HASH:
        return False
    return bcrypt.checkpw(password.encode(), ADMIN_HASH.encode())

def is_authenticated(request: Request) -> bool:
    return request.session.get("logged_in", False)

def require_login(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/fbnp/login"})
