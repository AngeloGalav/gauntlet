@echo off

:: Start the Python server
start "" python.exe webapp/server.py

:: Navigate to the frontend directory
cd webapp\frontend

:: Install npm packages
start "" cmd /k "npm install && npm start"
