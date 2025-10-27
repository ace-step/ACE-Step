# ACE-Step Docker Setup

Complete Docker containerization for ACE-Step music generation server.

## Prerequisites

### 1. Docker & Docker Compose
```bash
# Arch Linux
sudo pacman -S docker docker-compose

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

### 2. NVIDIA Container Toolkit (for GPU support)
```bash
# Arch Linux
sudo pacman -S nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Build and Run
```bash
# Build the Docker image
./docker-manage.sh build

# Start the container
./docker-manage.sh start

# Access the web interface
# Local:    http://localhost:7866
# Network:  http://192.168.4.72:7866
```

### Management Commands
```bash
./docker-manage.sh status      # Check if running
./docker-manage.sh logs        # View logs (live)
./docker-manage.sh stop        # Stop container
./docker-manage.sh restart     # Restart container
./docker-manage.sh shell       # Open shell in container
./docker-manage.sh clean       # Remove container
./docker-manage.sh rebuild     # Full rebuild from scratch
```

## Directory Structure

```
ACE-Step/
├── Dockerfile.scorpy           # Docker build instructions
├── docker-compose.scorpy.yml   # Container orchestration
├── docker-manage.sh            # Management script
├── .dockerignore              # Files to exclude from build
│
├── checkpoints/               # Model files (persistent)
├── outputs/                   # Generated audio files
└── logs/                      # Application logs
```

## Configuration

### Default Settings
The container runs with these optimized settings:
- **Port:** 7866 (mapped to host)
- **GPU:** Full access to all NVIDIA GPUs
- **Memory:** CPU offload enabled (~0.4GB VRAM)
- **Quality:** BFloat16 precision
- **Decoding:** Overlapped decode enabled

### Custom Configuration
Edit `docker-compose.scorpy.yml` to change settings:

```yaml
# Change port
ports:
  - "8080:7866"  # Access on port 8080

# Modify ACE-Step arguments
command: >
  acestep
  --server_name 0.0.0.0
  --port 7866
  --bf16 false           # Disable BFloat16
  --cpu_offload false    # Use more VRAM for quality
  --overlapped_decode true
```

## Volume Mounts

### Persistent Data
- `./checkpoints` → `/app/checkpoints` - Model files
- `./outputs` → `/app/outputs` - Generated audio
- `./logs` → `/app/logs` - Application logs
- `ace-step-cache` → `/root/.cache/ace-step` - Downloaded models

### Accessing Outputs
Generated audio files are available at:
```bash
ls -la outputs/
```

## Network Access

### Firewall Configuration
The container exposes port 7866. Ensure your firewall allows it:
```bash
# Check UFW status
sudo ufw status

# Allow port 7866
sudo ufw allow 7866/tcp comment 'ACE-Step Docker'
```

### Access Points
- **Local:** http://localhost:7866
- **LAN:** http://192.168.4.72:7866
- **Tailscale:** http://100.66.73.8:7866 (when active)

## Troubleshooting

### Check Container Status
```bash
./docker-manage.sh status
docker logs ace-step-server
```

### GPU Not Detected
```bash
# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker exec ace-step-server nvidia-smi
```

### Container Won't Start
```bash
# Check logs
./docker-manage.sh logs

# Rebuild from scratch
./docker-manage.sh rebuild
```

### Permission Issues
```bash
# Ensure user is in docker group
groups | grep docker

# If not, add and re-login
sudo usermod -aG docker $USER
```

## Performance

### Resource Usage
- **Build Time:** ~10-15 minutes (first time)
- **Image Size:** ~8-10 GB
- **Runtime VRAM:** ~0.4GB (with cpu_offload)
- **Runtime VRAM:** ~8-12GB (without cpu_offload, full quality)

### Generation Speed (RTX 5090)
- **27 steps:** ~9 seconds for 10s audio
- **60 steps:** ~20 seconds for 10s audio
- **RTF:** ~6 iterations/second

## Advanced Usage

### Custom Build
```bash
# Build with specific tag
docker compose -f docker-compose.scorpy.yml build --tag ace-step:v1.0

# Build without cache
docker compose -f docker-compose.scorpy.yml build --no-cache
```

### Environment Variables
```bash
# Set custom output directory
docker compose -f docker-compose.scorpy.yml run \
  -e ACE_OUTPUT_DIR=/custom/path \
  ace-step
```

### Interactive Shell
```bash
# Access container shell
./docker-manage.sh shell

# Or manually
docker exec -it ace-step-server /bin/bash
```

## Integration with asay

The `asay` tool works seamlessly with the Docker container:

```bash
# From any machine in your fleet
asay "Hello from Docker container"

# The tool auto-detects the containerized server
# at http://192.168.4.72:7866
```

## Backup & Restore

### Backup Model Checkpoints
```bash
tar -czf ace-step-checkpoints-$(date +%Y%m%d).tar.gz checkpoints/
```

### Restore
```bash
tar -xzf ace-step-checkpoints-*.tar.gz
./docker-manage.sh restart
```

## Updates

### Update ACE-Step Code
```bash
cd /home/brian/ACE-Step
git pull origin main
./docker-manage.sh rebuild
```

### Update Base Image
Edit `Dockerfile.scorpy` to use newer CUDA/Ubuntu versions, then:
```bash
./docker-manage.sh rebuild
```

## Production Deployment

### Auto-start on Boot
```bash
# Using systemd
sudo systemctl enable docker

# Container auto-restart is already enabled via:
# restart: unless-stopped
```

### Monitoring
```bash
# Health check status
docker inspect ace-step-server | grep -A 5 Health

# Resource usage
docker stats ace-step-server
```

## Security

### Recommended Practices
1. Run behind reverse proxy (nginx/traefik)
2. Enable HTTPS with Let's Encrypt
3. Use firewall to limit access
4. Regular backups of checkpoints
5. Keep base images updated

## License

ACE-Step is licensed under Apache License 2.0.
See main README.md for details.
