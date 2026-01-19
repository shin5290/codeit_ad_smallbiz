#실행방법 
#chmod +x run.sh
#./run.sh



#!/bin/bash
set -e

PINK="\033[35m"   # magenta
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

# 서버 실행
exec python -m uvicorn main:app \
  --host 0.0.0.0 \
  --port 9000


#db확인
#psql -U aduser -d adbizdb

# db 마이그레이션
# alembic upgrade head