#!/bin/bash
# ============================================================
# Cloud Run Deployment Script for SSAI Course Apps
# Deploys weeks 4, 5, 6 backends to Google Cloud Run
# ============================================================

# Source gcloud PATH (needed when running as a script on macOS)
if [ -f "$HOME/Documents/google-cloud-sdk/path.bash.inc" ]; then
    source "$HOME/Documents/google-cloud-sdk/path.bash.inc"
elif [ -f "$HOME/google-cloud-sdk/path.bash.inc" ]; then
    source "$HOME/google-cloud-sdk/path.bash.inc"
elif [ -f "$HOME/.config/google-cloud-sdk/path.bash.inc" ]; then
    source "$HOME/.config/google-cloud-sdk/path.bash.inc"
fi
#
# Prerequisites:
#   1. gcloud CLI installed (https://cloud.google.com/sdk)
#   2. Authenticated: gcloud auth login
#   3. Project set: gcloud config set project YOUR_PROJECT_ID
#   4. APIs enabled:
#        gcloud services enable run.googleapis.com cloudbuild.googleapis.com
#
# Usage:
#   ./deploy-cloudrun.sh                    # Deploy all weeks
#   ./deploy-cloudrun.sh week4              # Deploy only week 4
#   ./deploy-cloudrun.sh week5 week6        # Deploy weeks 5 and 6
#
# After deployment, update your Railway frontend HTML files
# with the Cloud Run URLs printed at the end.
# ============================================================

set -e

# Configuration - EDIT THESE
REGION="${GCP_REGION:-us-central1}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"

# If project is empty, try to detect it
if [ -z "$PROJECT" ]; then
    PROJECT=$(gcloud config list --format='value(core.project)' 2>/dev/null || true)
fi

if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project detected."
    echo "Please run this script with your project ID:"
    echo "  GCP_PROJECT=corded-palisade-136523 ./deploy-cloudrun.sh"
    echo ""
    echo "Or set it first:"
    echo "  gcloud config set project corded-palisade-136523"
    exit 1
fi

# Cloud Run settings per week
WEEK3_LANDER_SERVICE="week3-lander-api"
WEEK4_SERVICE="week4-nn-api"
WEEK5_SERVICE="week5-cnn-api"
WEEK6_SERVICE="week6-transformer-api"

# Resource settings
CPU=4
MEMORY="8Gi"
MAX_INSTANCES=50
MIN_INSTANCES=0

echo "============================================"
echo "  SSAI Cloud Run Deployment"
echo "============================================"
echo "Project:  $PROJECT"
echo "Region:   $REGION"
echo ""

if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project set."
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Determine which weeks to deploy
WEEKS_TO_DEPLOY=("${@}")
if [ ${#WEEKS_TO_DEPLOY[@]} -eq 0 ]; then
    WEEKS_TO_DEPLOY=("week3-lander" "week4" "week5" "week6")
fi

deploy_week3_lander() {
    echo ""
    echo "--- Deploying Week 3: Lunar Lander RL API ---"
    cd "$SCRIPT_DIR/week3-lander-app"

    gcloud run deploy "$WEEK3_LANDER_SERVICE" \
        --source . \
        --region "$REGION" \
        --cpu "$CPU" \
        --memory "$MEMORY" \
        --timeout 3600 \
        --concurrency 3 \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --allow-unauthenticated \
        --set-env-vars="PYTHONUNBUFFERED=1" \
        --quiet

    WEEK3_LANDER_URL=$(gcloud run services describe "$WEEK3_LANDER_SERVICE" --region "$REGION" --format='value(status.url)')
    echo "Week 3 Lander API deployed: $WEEK3_LANDER_URL"
}

deploy_week4() {
    echo ""
    echo "--- Deploying Week 4: Neural Networks API ---"
    cd "$SCRIPT_DIR/week4-app"

    gcloud run deploy "$WEEK4_SERVICE" \
        --source . \
        --region "$REGION" \
        --cpu "$CPU" \
        --memory "$MEMORY" \
        --timeout 300 \
        --concurrency 3 \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --allow-unauthenticated \
        --set-env-vars="PYTHONUNBUFFERED=1" \
        --quiet

    WEEK4_URL=$(gcloud run services describe "$WEEK4_SERVICE" --region "$REGION" --format='value(status.url)')
    echo "Week 4 API deployed: $WEEK4_URL"
}

deploy_week5() {
    echo ""
    echo "--- Deploying Week 5: CNN Training API ---"
    cd "$SCRIPT_DIR/week5-app"

    gcloud run deploy "$WEEK5_SERVICE" \
        --source . \
        --region "$REGION" \
        --cpu "$CPU" \
        --memory "$MEMORY" \
        --timeout 600 \
        --concurrency 3 \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --allow-unauthenticated \
        --set-env-vars="PYTHONUNBUFFERED=1" \
        --quiet

    WEEK5_URL=$(gcloud run services describe "$WEEK5_SERVICE" --region "$REGION" --format='value(status.url)')
    echo "Week 5 API deployed: $WEEK5_URL"
}

deploy_week6() {
    echo ""
    echo "--- Deploying Week 6: Transformer API ---"
    cd "$SCRIPT_DIR/week6-app"

    gcloud run deploy "$WEEK6_SERVICE" \
        --source . \
        --region "$REGION" \
        --cpu "$CPU" \
        --memory "$MEMORY" \
        --timeout 900 \
        --concurrency 3 \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --allow-unauthenticated \
        --set-env-vars="PYTHONUNBUFFERED=1" \
        --quiet

    WEEK6_URL=$(gcloud run services describe "$WEEK6_SERVICE" --region "$REGION" --format='value(status.url)')
    echo "Week 6 API deployed: $WEEK6_URL"
}

# Deploy requested weeks
for week in "${WEEKS_TO_DEPLOY[@]}"; do
    case "$week" in
        week3-lander) deploy_week3_lander ;;
        week4) deploy_week4 ;;
        week5) deploy_week5 ;;
        week6) deploy_week6 ;;
        *) echo "Unknown week: $week (expected week3-lander, week4, week5, or week6)" ;;
    esac
done

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "Update your Railway frontend HTML files with the Cloud Run URLs."
echo "In each index.html, set the API base URL at the top of the <script> tag:"
echo ""

for week in "${WEEKS_TO_DEPLOY[@]}"; do
    case "$week" in
        week3-lander)
            echo "  Week 3 Lander (index.html): Update API_BASE_URL:"
            echo "    ${WEEK3_LANDER_URL:-https://YOUR-WEEK3-LANDER-URL.run.app}"
            echo ""
            ;;
        week4)
            echo "  Week 4 (index.html): Add before </head>:"
            echo "    <script>window.API_BASE = '${WEEK4_URL:-https://YOUR-WEEK4-URL.run.app}';</script>"
            echo ""
            ;;
        week5)
            echo "  Week 5 (index.html): Add before </head>:"
            echo "    <script>window.API_BASE_URL = '${WEEK5_URL:-https://YOUR-WEEK5-URL.run.app}';</script>"
            echo ""
            ;;
        week6)
            echo "  Week 6 (index.html): Add before </head>:"
            echo "    <script>window.API_BASE = '${WEEK6_URL:-https://YOUR-WEEK6-URL.run.app}';</script>"
            echo ""
            ;;
    esac
done

echo "Or set them dynamically in Railway environment variables"
echo "and inject via a small server-side template."
echo ""
echo "To check service status:"
echo "  gcloud run services list --region $REGION"
echo ""
echo "To view logs:"
echo "  gcloud run services logs read SERVICE_NAME --region $REGION"
echo ""
echo "To delete a service:"
echo "  gcloud run services delete SERVICE_NAME --region $REGION"
