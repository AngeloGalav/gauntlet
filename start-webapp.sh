#!/usr/bin/env bash
cd webapp

# start server first
python server.py

# start frontend second
cd frontend
npm install
npm start