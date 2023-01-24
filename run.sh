docker build -t local/mb-2-builder:latest .
docker run -it local/mb-2-builder:latest bash
docker run -e S3_BASE_PATH=s3://prefix/<modelid> local/mb-2-builder:latest
