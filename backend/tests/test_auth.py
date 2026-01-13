import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from jose import jwt

import auth

# Ensure secret key is consistent for testing
TEST_SECRET_KEY = "testsecret"
auth.SECRET_KEY = TEST_SECRET_KEY

# ───────────────────────────────
# hash_password and verify_password
# ───────────────────────────────

def test_hash_and_verify_password():
    pw = "mypass123"
    hashed = auth.hash_password(pw)
    assert isinstance(hashed, str)
    assert auth.verify_password(pw, hashed)
    assert not auth.verify_password("wrongpass", hashed)

# ───────────────────────────────
# authenticate_user
# ───────────────────────────────

@patch("auth.users_collection.find_one")
@patch("auth.verify_password")
def test_authenticate_user_success(mock_verify, mock_find_one):
    user_data = {"username": "testuser", "password": "hashedpw"}
    mock_find_one.return_value = user_data
    mock_verify.return_value = True

    result = auth.authenticate_user("testuser", "plainpw")

    assert result == user_data
    mock_find_one.assert_called_once_with({"username": "testuser"})
    mock_verify.assert_called_once_with("plainpw", "hashedpw")


@patch("auth.users_collection.find_one")
@patch("auth.verify_password")
def test_authenticate_user_invalid_password(mock_verify, mock_find_one):
    user_data = {"username": "testuser", "password": "hashedpw"}
    mock_find_one.return_value = user_data
    mock_verify.return_value = False

    result = auth.authenticate_user("testuser", "wrongpw")

    assert result is None


@patch("auth.users_collection.find_one")
def test_authenticate_user_user_not_found(mock_find_one):
    mock_find_one.return_value = None
    result = auth.authenticate_user("nosuchuser", "whatever")
    assert result is None

# ───────────────────────────────
# create_access_token
# ───────────────────────────────

def test_create_access_token_encodes_payload():
    data = {"sub": "testuser"}
    token = auth.create_access_token(data, expires_delta=timedelta(minutes=10))
    decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=[auth.ALGORITHM])
    assert decoded["sub"] == "testuser"
    assert "exp" in decoded

# ───────────────────────────────
# get_current_user (async)
# ───────────────────────────────

@patch("auth.users_collection.find_one")
@pytest.mark.asyncio
async def test_get_current_user_success(mock_find_one):
    token_data = {"sub": "testuser"}
    token = jwt.encode(token_data, TEST_SECRET_KEY, algorithm=auth.ALGORITHM)
    
    mock_find_one.return_value = {"username": "testuser"}

    user = await auth.get_current_user(token=token)
    assert user == {"username": "testuser"}


@patch("auth.users_collection.find_one")
@pytest.mark.asyncio
async def test_get_current_user_user_not_found(mock_find_one):
    token_data = {"sub": "testuser"}
    token = jwt.encode(token_data, TEST_SECRET_KEY, algorithm=auth.ALGORITHM)
    
    mock_find_one.return_value = None

    with pytest.raises(auth.HTTPException) as exc:
        await auth.get_current_user(token=token)

    assert exc.value.status_code == 401
    assert "User not found" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    token = "invalidtoken"

    with pytest.raises(auth.HTTPException) as exc:
        await auth.get_current_user(token=token)

    assert exc.value.status_code == 401
    assert "Invalid token" in str(exc.value.detail)
