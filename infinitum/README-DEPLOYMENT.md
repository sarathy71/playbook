# Topic Browser - Google Cloud Deployment Guide

This guide will help you deploy the Topic Browser Flask application to Google Cloud Platform.

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Make sure you have gcloud CLI installed and authenticated
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Run the deployment script
./deploy.sh
```

### Option 2: Manual Deployment

#### Deploy to Google Cloud Run

```bash
# Build and deploy
gcloud run deploy topic-browser \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10
```

#### Deploy to Google App Engine

```bash
# Deploy using app.yaml
gcloud app deploy app.yaml
```

## 📋 Prerequisites

1. **Google Cloud Account**: Sign up at [cloud.google.com](https://cloud.google.com)
2. **Google Cloud CLI**: Install from [cloud.google.com/sdk](https://cloud.google.com/sdk)
3. **Docker** (for local testing): Install from [docker.com](https://docker.com)
4. **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com)

## 🔧 Environment Variables

Set these in Google Cloud Console or via gcloud CLI:

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for LLM functionality

### Optional
- `YOUTUBE_API_KEY`: YouTube API key for video suggestions
- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Port number (default: 8080)

### Setting Environment Variables

#### For Cloud Run:
```bash
gcloud run services update topic-browser \
    --set-env-vars OPENAI_API_KEY=your-key-here \
    --set-env-vars YOUTUBE_API_KEY=your-youtube-key
```

#### For App Engine:
Add to `app.yaml`:
```yaml
env_variables:
  OPENAI_API_KEY: "your-key-here"
  YOUTUBE_API_KEY: "your-youtube-key"
```

## 🏗️ Architecture

### Dockerfile Features
- **Python 3.11**: Latest stable Python version
- **Multi-stage optimization**: Efficient layer caching
- **Security**: Non-root user execution
- **Health checks**: Built-in health monitoring
- **Production ready**: Gunicorn WSGI server

### Application Structure
```
/
├── main.py              # Flask application
├── templates/
│   └── index.html       # Frontend with all features
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── app.yaml            # App Engine configuration
├── deploy.sh           # Deployment script
└── .dockerignore       # Docker build optimization
```

## 🧪 Local Testing

### Using Docker
```bash
# Build image
docker build -t topic-browser .

# Run container
docker run -p 8080:8080 \
    -e OPENAI_API_KEY=your-key \
    -e YOUTUBE_API_KEY=your-youtube-key \
    topic-browser
```

### Using Python directly
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-key
export YOUTUBE_API_KEY=your-youtube-key

# Run application
python main.py
```

## 📊 Monitoring & Logs

### View Logs
```bash
# Cloud Run logs
gcloud logs read --service=topic-browser

# App Engine logs
gcloud app logs tail
```

### Monitor Performance
- **Cloud Run**: Use Cloud Console monitoring dashboard
- **App Engine**: Use App Engine dashboard in Cloud Console

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use Google Cloud Secret Manager for sensitive data
3. **HTTPS**: Both Cloud Run and App Engine provide HTTPS by default
4. **Authentication**: Consider adding authentication for production use

## 💰 Cost Optimization

### Cloud Run
- **Pay per request**: Only pay when the app is used
- **Automatic scaling**: Scales to zero when not in use
- **Resource limits**: Configured for optimal cost/performance

### App Engine
- **Automatic scaling**: Scales based on traffic
- **Free tier**: Includes generous free usage limits

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check Docker build locally
   docker build -t topic-browser .
   ```

2. **Environment Variables Not Set**
   ```bash
   # Verify environment variables
   gcloud run services describe topic-browser --region us-central1
   ```

3. **API Key Issues**
   - Verify API key is valid and has sufficient credits
   - Check OpenAI API key permissions

4. **Memory Issues**
   ```bash
   # Increase memory allocation
   gcloud run services update topic-browser --memory 1Gi
   ```

### Debug Mode
For debugging, you can run locally with debug mode:
```bash
export FLASK_ENV=development
python main.py
```

## 🔄 Updates & Maintenance

### Updating the Application
1. Make your changes to the code
2. Test locally using Docker or Python
3. Deploy using the deployment script: `./deploy.sh`

### Scaling
- **Cloud Run**: Automatically scales based on traffic
- **App Engine**: Configure scaling in `app.yaml`

## 📞 Support

If you encounter issues:
1. Check the logs using `gcloud logs`
2. Verify environment variables are set correctly
3. Test locally to isolate issues
4. Check Google Cloud status page for service issues

## 🎯 Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Configure proper environment variables
- [ ] Set up monitoring and alerting
- [ ] Configure custom domain (optional)
- [ ] Set up backup strategy for notebooks
- [ ] Review security settings
- [ ] Test all functionality in production environment
- [ ] Set up CI/CD pipeline (optional)

---

**Happy Deploying! 🚀**

Your Topic Browser is now ready to help users explore and learn from any topic with AI-powered deep dives and interactive notebooks!
