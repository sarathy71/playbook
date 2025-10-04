#!/bin/bash

# Topic Browser - Google Cloud Deployment Script
# This script provides options for deploying to Google Cloud Run or App Engine

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
SERVICE_NAME="topic-browser"
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Topic Browser Deployment Script${NC}"
echo "=================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated with gcloud. Please run: gcloud auth login${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}📋 Setting project to: ${PROJECT_ID}${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable appengine.googleapis.com

# Function to deploy to Cloud Run
deploy_cloud_run() {
    echo -e "${GREEN}🚀 Deploying to Google Cloud Run...${NC}"
    
    # Build and deploy
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 512Mi \
        --cpu 1 \
        --timeout 300 \
        --max-instances 10 \
        --set-env-vars FLASK_ENV=production
    
    echo -e "${GREEN}✅ Deployment complete!${NC}"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
    echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
}

# Function to deploy to App Engine
deploy_app_engine() {
    echo -e "${GREEN}🚀 Deploying to Google App Engine...${NC}"
    
    gcloud app deploy app.yaml --quiet
    
    echo -e "${GREEN}✅ Deployment complete!${NC}"
    
    # Get service URL
    SERVICE_URL=$(gcloud app browse --no-launch-browser)
    echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
}

# Function to build Docker image locally
build_local() {
    echo -e "${GREEN}🔨 Building Docker image locally...${NC}"
    
    docker build -t topic-browser:latest .
    
    echo -e "${GREEN}✅ Docker image built successfully!${NC}"
    echo -e "${YELLOW}To run locally: docker run -p 8080:8080 topic-browser:latest${NC}"
}

# Function to test locally
test_local() {
    echo -e "${GREEN}🧪 Testing locally...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Set environment variables
    export FLASK_ENV=development
    export PORT=8080
    
    # Run the app
    echo -e "${GREEN}Starting local server...${NC}"
    echo -e "${YELLOW}Visit: http://localhost:8080${NC}"
    python main.py
}

# Main menu
echo ""
echo "Select deployment option:"
echo "1) Deploy to Google Cloud Run (Recommended)"
echo "2) Deploy to Google App Engine"
echo "3) Build Docker image locally"
echo "4) Test locally"
echo "5) Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        deploy_cloud_run
        ;;
    2)
        deploy_app_engine
        ;;
    3)
        build_local
        ;;
    4)
        test_local
        ;;
    5)
        echo -e "${GREEN}👋 Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Done!${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Set your OpenAI API key in Google Cloud Console"
echo "2. Optionally set YouTube API key for video suggestions"
echo "3. Configure environment variables in Cloud Run/App Engine"
echo ""
echo -e "${YELLOW}🔧 Environment Variables to Set:${NC}"
echo "- OPENAI_API_KEY=your-openai-api-key"
echo "- YOUTUBE_API_KEY=your-youtube-api-key (optional)"
echo ""

