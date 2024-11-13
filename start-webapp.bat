@echo off

:: Navigate to the webapp directory
cd webapp

:: Start the Python server
start "" python.exe server.py

:: Navigate to the frontend directory
cd frontend

:: Install npm packages
npm install

:: Start the frontend
npm start
