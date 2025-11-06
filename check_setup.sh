#!/bin/bash

# Installation Verification Script for Drone Tuning Dashboard
# This script checks all prerequisites and guides you through setup

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Drone PID Tuning Dashboard - Setup Checker      ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo ""

# Function to check command
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed ($(command -v $1))"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        return 1
    fi
}

# Function to check version
check_version() {
    local cmd=$1
    local version=$2
    echo -e "${BLUE}  Version:${NC} $version"
}

MISSING_DEPS=0

echo -e "${YELLOW}Checking Prerequisites...${NC}"
echo ""

# Check Python
echo "1. Python 3.10+"
if check_command python3; then
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    check_version python3 "$PY_VERSION"
else
    ((MISSING_DEPS++))
fi
echo ""

# Check pip
echo "2. pip"
if check_command pip3; then
    PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
    check_version pip "$PIP_VERSION"
else
    ((MISSING_DEPS++))
fi
echo ""

# Check Node.js
echo "3. Node.js 18+"
if check_command node; then
    NODE_VERSION=$(node --version 2>&1)
    check_version node "$NODE_VERSION"
else
    ((MISSING_DEPS++))
fi
echo ""

# Check npm
echo "4. npm"
if check_command npm; then
    NPM_VERSION=$(npm --version 2>&1)
    check_version npm "$NPM_VERSION"
else
    ((MISSING_DEPS++))
fi
echo ""

# Check Docker (optional)
echo "5. Docker (Optional - for containerized deployment)"
if check_command docker; then
    DOCKER_VERSION=$(docker --version 2>&1)
    check_version docker "$DOCKER_VERSION"
else
    echo -e "${YELLOW}  Docker not installed (optional)${NC}"
fi
echo ""

# Check Docker Compose (optional)
echo "6. Docker Compose (Optional)"
if check_command docker-compose; then
    DC_VERSION=$(docker-compose --version 2>&1)
    check_version docker-compose "$DC_VERSION"
else
    echo -e "${YELLOW}  Docker Compose not installed (optional)${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""

if [ $MISSING_DEPS -eq 0 ]; then
    echo -e "${GREEN}✓ All required dependencies are installed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Install backend dependencies:  cd backend && pip3 install -r requirements.txt"
    echo "  2. Install frontend dependencies: cd frontend && npm install"
    echo "  3. Run the startup script:        ./start_dashboard.sh"
    echo ""
else
    echo -e "${RED}✗ Missing $MISSING_DEPS required dependencies${NC}"
    echo ""
    echo -e "${YELLOW}Installation instructions:${NC}"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install python3 python3-pip nodejs npm"
    echo ""
    echo "macOS:"
    echo "  brew install python3 node"
    echo ""
    echo "Fedora/RHEL:"
    echo "  sudo dnf install python3 python3-pip nodejs npm"
    echo ""
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Warning: docker-compose.yml not found in current directory${NC}"
    echo "Please run this script from the MC07_tuning project root"
fi
