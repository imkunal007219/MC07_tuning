#!/bin/bash

# Drone PID Tuning Dashboard Startup Script

set -e

echo "===================================="
echo "🚁 Drone PID Tuning Dashboard"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running in project directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo -e "${YELLOW}Select startup mode:${NC}"
echo "1) Docker Compose (Recommended)"
echo "2) Manual (Backend + Frontend separately)"
echo "3) Backend only"
echo "4) Frontend only"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "${GREEN}Starting with Docker Compose...${NC}"
        echo ""

        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Docker is not installed${NC}"
            echo "Please install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi

        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}Docker Compose is not installed${NC}"
            echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
            exit 1
        fi

        # Build and start containers
        echo "Building containers..."
        docker-compose build

        echo "Starting services..."
        docker-compose up -d

        echo ""
        echo -e "${GREEN}✅ Dashboard started successfully!${NC}"
        echo ""
        echo "Services:"
        echo "  Frontend:  http://localhost:3000"
        echo "  Backend:   http://localhost:8000"
        echo "  API Docs:  http://localhost:8000/docs"
        echo ""
        echo "View logs:"
        echo "  docker-compose logs -f"
        echo ""
        echo "Stop services:"
        echo "  docker-compose down"
        ;;

    2)
        echo -e "${GREEN}Starting Manual Mode...${NC}"
        echo ""

        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}Python 3 is not installed${NC}"
            exit 1
        fi

        # Check Node.js
        if ! command -v node &> /dev/null; then
            echo -e "${RED}Node.js is not installed${NC}"
            exit 1
        fi

        # Start Backend
        echo "Starting Backend..."
        cd backend
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
        fi

        source venv/bin/activate
        pip install -q -r requirements.txt

        echo "Backend starting on http://localhost:8000"
        python api_server.py &
        BACKEND_PID=$!
        cd ..

        sleep 3

        # Start Frontend
        echo "Starting Frontend..."
        cd frontend
        if [ ! -d "node_modules" ]; then
            echo "Installing npm dependencies..."
            npm install
        fi

        echo "Frontend starting on http://localhost:3000"
        npm start &
        FRONTEND_PID=$!
        cd ..

        echo ""
        echo -e "${GREEN}✅ Dashboard started successfully!${NC}"
        echo ""
        echo "Services:"
        echo "  Frontend:  http://localhost:3000 (PID: $FRONTEND_PID)"
        echo "  Backend:   http://localhost:8000 (PID: $BACKEND_PID)"
        echo ""
        echo "Stop services:"
        echo "  kill $BACKEND_PID $FRONTEND_PID"
        ;;

    3)
        echo -e "${GREEN}Starting Backend only...${NC}"
        cd backend

        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi

        source venv/bin/activate
        pip install -q -r requirements.txt
        python api_server.py
        ;;

    4)
        echo -e "${GREEN}Starting Frontend only...${NC}"
        cd frontend

        if [ ! -d "node_modules" ]; then
            npm install
        fi

        npm start
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
