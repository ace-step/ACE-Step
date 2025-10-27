#!/bin/bash
# ACE-Step Interactive Launcher
# Port 7866 - Accessible on local network and Tailscale

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}          ${BOLD}ACE-Step Music Generation Launcher${NC}              ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show system info
echo -e "${BOLD}System Information:${NC}"
echo -e "  GPU: ${GREEN}$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')${NC}"
echo -e "  PyTorch: ${GREEN}$(source .venv/bin/activate && python -c 'import torch; print(torch.__version__)' 2>/dev/null)${NC}"
echo -e "  CUDA: ${GREEN}$(source .venv/bin/activate && python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)${NC}"
echo ""

# Show network info
LOCAL_IP=$(ip addr show wlan0 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1)
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)

echo -e "${BOLD}Network Access (Port 7866):${NC}"
echo -e "  Local:        ${YELLOW}http://localhost:7866${NC}"
if [ -n "$LOCAL_IP" ]; then
    echo -e "  LAN:          ${YELLOW}http://$LOCAL_IP:7866${NC}"
fi
if [ -n "$TAILSCALE_IP" ]; then
    echo -e "  Tailscale:    ${GREEN}http://$TAILSCALE_IP:7866${NC} ✓"
else
    echo -e "  Tailscale:    ${RED}Not running${NC}"
fi
echo ""

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}Launch Options:${NC}"
echo ""
echo -e "  ${GREEN}1)${NC} ${BOLD}Full Quality Mode${NC} (Recommended)"
echo -e "     • Maximum audio quality"
echo -e "     • Uses ~8-12GB VRAM"
echo -e "     • Best for final productions"
echo ""
echo -e "  ${YELLOW}2)${NC} ${BOLD}Memory Optimized Mode${NC}"
echo -e "     • CPU offload enabled"
echo -e "     • Uses ~0.4GB VRAM"
echo -e "     • Slightly lower quality"
echo -e "     • Good for testing/experimenting"
echo ""
echo -e "  ${BLUE}3)${NC} ${BOLD}Fast Mode${NC} (Experimental)"
echo -e "     • All optimizations enabled"
echo -e "     • Fastest generation"
echo -e "     • May have quality trade-offs"
echo ""
echo -e "  ${CYAN}4)${NC} ${BOLD}Custom Options${NC}"
echo -e "     • Manually specify parameters"
echo ""
echo -e "  ${RED}5)${NC} Exit"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -n -e "${BOLD}Select option [1-5]:${NC} "

read -r choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting ACE-Step in Full Quality Mode...${NC}\n"
        source .venv/bin/activate
        acestep \
          --server_name 0.0.0.0 \
          --port 7866 \
          --bf16 true \
          --torch_compile false
        ;;
    2)
        echo -e "\n${YELLOW}Starting ACE-Step in Memory Optimized Mode...${NC}\n"
        source .venv/bin/activate
        acestep \
          --server_name 0.0.0.0 \
          --port 7866 \
          --bf16 true \
          --torch_compile false \
          --cpu_offload true \
          --overlapped_decode true
        ;;
    3)
        echo -e "\n${BLUE}Starting ACE-Step in Fast Mode...${NC}\n"
        source .venv/bin/activate
        acestep \
          --server_name 0.0.0.0 \
          --port 7866 \
          --bf16 true \
          --torch_compile false \
          --cpu_offload true \
          --overlapped_decode true
        ;;
    4)
        echo -e "\n${CYAN}Custom Options${NC}"
        echo -e "Current working options for RTX 5090:"
        echo -e "  ${GREEN}--server_name 0.0.0.0${NC} (required for network access)"
        echo -e "  ${GREEN}--port 7866${NC} (your configured port)"
        echo -e "  ${GREEN}--bf16 true/false${NC} (use bfloat16)"
        echo -e "  ${GREEN}--torch_compile false${NC} (must be false for RTX 5090)"
        echo -e "  ${YELLOW}--cpu_offload true/false${NC} (save VRAM)"
        echo -e "  ${YELLOW}--overlapped_decode true/false${NC} (faster decoding)"
        echo ""
        echo -n "Enter custom acestep arguments: "
        read -r custom_args
        echo -e "\n${CYAN}Starting ACE-Step with custom options...${NC}\n"
        source .venv/bin/activate
        acestep --server_name 0.0.0.0 --port 7866 $custom_args
        ;;
    5)
        echo -e "\n${RED}Exiting...${NC}\n"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid option!${NC}\n"
        exit 1
        ;;
esac
