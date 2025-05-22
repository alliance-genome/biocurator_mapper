#!/bin/bash

# Biocurator Mapper - Local Development Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 Biocurator Mapper - Local Development Setup${NC}"
echo "==============================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env file with your API keys:${NC}"
    echo "   - OPENAI_API_KEY: Get from https://platform.openai.com/account/api-keys"
    echo "   - ADMIN_API_KEY: Set to a secure secret"
    echo ""
    read -p "Press Enter after updating .env file..."
fi

# Load environment variables
if [ -f .env ]; then
    echo -e "${BLUE}📖 Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo -e "${RED}❌ OPENAI_API_KEY not set properly in .env file${NC}"
    echo "   Please set a valid OpenAI API key and restart."
    exit 1
fi

if [ -z "$ADMIN_API_KEY" ] || [ "$ADMIN_API_KEY" = "your_admin_secret_key_here" ]; then
    echo -e "${YELLOW}⚠️  Using default ADMIN_API_KEY. Please set a secure one in .env${NC}"
    export ADMIN_API_KEY="dev-admin-key-$(date +%s)"
fi

echo -e "${GREEN}✅ Environment variables loaded${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Clean up any existing containers
echo -e "${BLUE}🧹 Cleaning up existing containers...${NC}"
docker-compose down 2>/dev/null || true

# Start services
echo -e "${BLUE}🚀 Starting Biocurator Mapper services...${NC}"
echo "This may take a few minutes on first run (downloading images, building containers)"
echo ""

docker-compose up --build

echo -e "${GREEN}🎉 Services started successfully!${NC}"
echo ""
echo "Access your services:"
echo -e "  📊 ${BLUE}Streamlit UI:${NC}     http://localhost:8501"
echo -e "  🔌 ${BLUE}FastAPI Backend:${NC}  http://localhost:8000"
echo -e "  📚 ${BLUE}API Documentation:${NC} http://localhost:8000/docs"
echo -e "  🔍 ${BLUE}Weaviate DB:${NC}     http://localhost:8080"
echo ""
echo -e "${YELLOW}💡 Tip: Use Ctrl+C to stop all services${NC}"