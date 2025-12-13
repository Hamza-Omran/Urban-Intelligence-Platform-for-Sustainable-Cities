#!/bin/bash

# ============================================
# Big Data Project - Quick Setup Script
# ============================================

echo "Big Data Project - MinIO Complete Setup"
echo "==========================================="
echo ""

# Step 1: Start MinIO
echo "Step 1: Starting MinIO container..."
docker compose up -d
echo "MinIO started"
echo ""

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
sleep 5
echo ""

# Step 2: Create buckets
echo "Step 2: Creating MinIO buckets..."
python3 Scripts/create_buckets.py
echo ""

# Step 3: Upload raw data to bronze
echo "Step 3: Uploading raw data to bronze bucket..."
python3 Scripts/upload_to_minio.py
echo ""

# Step 4: Upload cleaned data to silver
echo "Step 4: Uploading cleaned data to silver bucket..."
python3 Scripts/upload_to_silver.py
echo ""

# Done
echo "==========================================="
echo "Setup Complete!"
echo ""
echo "MinIO Web Console: http://localhost:9003"
echo "Username: bdprojectfcds4"
echo "Password: bdprojectfcds4"
echo ""
echo "Buckets:"
echo "   - bronze: Raw CSV data"
echo "   - silver: Cleaned Parquet data"
echo "   - gold: Results (ready for your work)"
echo ""
echo "You can now proceed with:"
echo "   - Phase 4: Dataset Merging"
echo "   - Phase 5: Monte Carlo Simulation"
echo "   - Phase 6: Factor Analysis"
echo "==========================================="
