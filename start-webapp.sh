#!/usr/bin/env bash

# start server first
python webapp/server.py

# start frontend second
cd webapp
cd frontend
npm install
npm start