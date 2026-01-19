#실행방법 
#chmod +x run.sh
#./run.sh



#!/bin/bash
set -e

PINK="\033[35m"   # magenta
GREEN="\033[32m"
BLUE="\033[34m"
NC="\033[0m"

echo -e "${PINK}"
cat <<'EOF'
          .-') _     ('-.   ('-.     _   .-')
         (  OO) )  _(  OO) ( OO ).-.( '.( OO )_
    .---./     '._(,------./ . --. / ,--.   ,--.)
   / .  ||'--...__)|  .---'| \-.  \  |   `.'   |
  / /|  |'--.  .--'|  |  .-'-'  |  | |         |
 / / |  |_  |  |  (|  '--.\| |_.'  | |  |'.'|  |
/  '-'    | |  |   |  .--' |  .-.  | |  |   |  |
`----|  |-' |  |   |  `---.|  | |  | |  |   |  |
     `--'   `--'   `------'`--' `--' `--'   `--'
EOF
echo -e "${NC}"

# 가상환경 활성화
source /opt/jhub-venv/bin/activate

# 파이썬 사용자 site 패키지 무시 (꼬임 방지)
export PYTHONNOUSERSITE=1

# 앱 디렉토리 이동
cd ~/codeit_ad_smallbiz

# 서버 URL 설정
HOST="0.0.0.0"
PORT="9000"
# 외부 IP 자동 감지 (실패하면 localhost 사용)
EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
URL="http://${EXTERNAL_IP}:${PORT}"

echo -e "${BLUE}🚀 Starting FastAPI Server...${NC}"
echo -e "${GREEN}📍 Server will be available at: ${URL}${NC}"

# 백그라운드에서 서버 시작
python -m uvicorn main:app --host ${HOST} --port ${PORT} &
SERVER_PID=$!

# Cleanup function
cleanup() {
    echo -e "\n${PINK}Shutting down server...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit
}
trap cleanup SIGINT SIGTERM

# 서버가 준비될 때까지 대기
echo -e "${BLUE}⏳ Waiting for server to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:${PORT} > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is ready!${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${PINK}⚠️  Server startup timeout${NC}"
        cleanup
    fi
done

# 브라우저 자동 열기 (Linux 환경)
echo -e "${BLUE}🌐 Opening browser...${NC}"
if command -v xdg-open > /dev/null; then
    xdg-open "${URL}" 2>/dev/null &
elif command -v python3 > /dev/null; then
    python3 -m webbrowser "${URL}" 2>/dev/null &
fi

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Server is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📍 Local:    http://localhost:${PORT}${NC}"
echo -e "${BLUE}📍 Network:  ${URL}${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PINK}Press Ctrl+C to stop the server${NC}\n"

# 서버 프로세스 대기
wait $SERVER_PID


#db확인
#psql -U aduser -d adbizdb

# db 마이그레이션
# alembic upgrade head