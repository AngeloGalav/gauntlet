@echo off

:: Navigate to the webapp directory

:: Start the Python server
start "" python.exe webapp/server.py

:: Navigate to the frontend directory
cd webapp
cd frontend

:: Install npm packages
npm install

:: Start the frontend
npm start
