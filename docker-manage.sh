#!/bin/bash
# ACE-Step Docker Management Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.scorpy.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

show_help() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}          ${BOLD}ACE-Step Docker Management${NC}                      ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 [command]"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo -e "  ${GREEN}build${NC}       Build the Docker image"
    echo -e "  ${GREEN}start${NC}       Start the container (detached)"
    echo -e "  ${GREEN}stop${NC}        Stop the container"
    echo -e "  ${GREEN}restart${NC}     Restart the container"
    echo -e "  ${GREEN}logs${NC}        View container logs (live)"
    echo -e "  ${GREEN}status${NC}      Show container status"
    echo -e "  ${GREEN}shell${NC}       Open bash shell in container"
    echo -e "  ${GREEN}clean${NC}       Stop and remove container"
    echo -e "  ${GREEN}rebuild${NC}     Clean, rebuild, and start"
    echo -e "  ${GREEN}help${NC}        Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  $0 build        # Build the image"
    echo -e "  $0 start        # Start ACE-Step server"
    echo -e "  $0 logs         # Watch the logs"
    echo -e "  $0 rebuild      # Full rebuild from scratch"
    echo ""
}

check_requirements() {
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi

    # Check nvidia-container-toolkit
    if ! docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-container-toolkit may not be properly configured${NC}"
        echo -e "${YELLOW}GPU support may not work. Install with:${NC}"
        echo -e "  sudo pacman -S nvidia-container-toolkit"
        echo ""
    fi
}

case "${1:-help}" in
    build)
        echo -e "${BLUE}Building ACE-Step Docker image...${NC}"
        check_requirements
        docker compose -f "$COMPOSE_FILE" build
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;

    start)
        echo -e "${BLUE}Starting ACE-Step container...${NC}"
        check_requirements
        docker compose -f "$COMPOSE_FILE" up -d
        echo -e "${GREEN}✓ Container started${NC}"
        echo -e "${CYAN}Access ACE-Step at: http://localhost:7866${NC}"
        echo -e "${YELLOW}View logs with: $0 logs${NC}"
        ;;

    stop)
        echo -e "${BLUE}Stopping ACE-Step container...${NC}"
        docker compose -f "$COMPOSE_FILE" stop
        echo -e "${GREEN}✓ Container stopped${NC}"
        ;;

    restart)
        echo -e "${BLUE}Restarting ACE-Step container...${NC}"
        docker compose -f "$COMPOSE_FILE" restart
        echo -e "${GREEN}✓ Container restarted${NC}"
        ;;

    logs)
        echo -e "${BLUE}Showing container logs (Ctrl+C to exit)...${NC}"
        docker compose -f "$COMPOSE_FILE" logs -f
        ;;

    status)
        echo -e "${CYAN}Container Status:${NC}"
        docker compose -f "$COMPOSE_FILE" ps
        echo ""
        if docker compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
            echo -e "${GREEN}✓ ACE-Step is running${NC}"
            echo -e "${CYAN}Access at: http://localhost:7866${NC}"
        else
            echo -e "${YELLOW}ACE-Step is not running${NC}"
        fi
        ;;

    shell)
        echo -e "${BLUE}Opening shell in container...${NC}"
        docker compose -f "$COMPOSE_FILE" exec ace-step /bin/bash
        ;;

    clean)
        echo -e "${YELLOW}Stopping and removing container...${NC}"
        docker compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✓ Container removed${NC}"
        ;;

    rebuild)
        echo -e "${YELLOW}Performing full rebuild...${NC}"
        docker compose -f "$COMPOSE_FILE" down
        docker compose -f "$COMPOSE_FILE" build --no-cache
        docker compose -f "$COMPOSE_FILE" up -d
        echo -e "${GREEN}✓ Rebuild complete${NC}"
        echo -e "${CYAN}Access ACE-Step at: http://localhost:7866${NC}"
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
