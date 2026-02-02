#!/bin/bash
# Check Docker build status

echo "Checking Docker build progress..."
echo ""

if pgrep -f "docker build" > /dev/null; then
    echo "✓ Docker build is running"
    echo ""
    echo "Latest build output:"
    echo "===================="
    tail -30 /tmp/docker_build.log
    echo ""
    echo "To monitor continuously: tail -f /tmp/docker_build.log"
else
    echo "✗ Docker build is not running"
    echo ""
    if [ -f /tmp/docker_build.log ]; then
        echo "Last build output:"
        echo "=================="
        tail -50 /tmp/docker_build.log
    fi
    
    # Check if image was built successfully
    if sudo docker images | grep -q "pixel2mesh"; then
        echo ""
        echo "✓ Docker image 'pixel2mesh' exists!"
        sudo docker images | grep pixel2mesh
    fi
fi
