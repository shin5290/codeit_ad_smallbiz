# GCP VM에서 디스크 용량을 확인하는 대표적인 명령어들

## 1. 전체 디스크 사용량 확인

```bash
# 파일시스템별 사용량 (사람이 읽기 쉬운 형태)
df -h

# inode 사용량 확인
df -i
```

## 2. 디렉토리별 용량 확인

```bash
# 현재 디렉토리의 하위 디렉토리별 용량
du -h --max-depth=1

# 특정 디렉토리의 전체 용량
du -sh /path/to/directory

# 용량 큰 순서대로 정렬 (상위 10개)
du -h --max-depth=1 | sort -hr | head -10

# 루트 디렉토리의 주요 디렉토리별 용량
sudo du -h --max-depth=1 / 2>/dev/null | sort -hr
```

## 3. 큰 파일/디렉토리 찾기

```bash
# 현재 위치에서 100MB 이상 파일 찾기
find . -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# 전체 시스템에서 1GB 이상 파일 찾기 (상위 20개)
sudo find / -type f -size +1G -exec ls -lh {} \; 2>/dev/null | head -20

# 가장 큰 파일 10개 찾기
sudo find / -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -10 | awk '{print $1/1024/1024 "MB", $2}'
```

## 4. 사용자별 디스크 사용량

```bash
# 모든 사용자의 홈 디렉토리 사용량
sudo du -sh /home/* 2>/dev/null

# 특정 사용자의 전체 파일 용량
sudo find / -user username -type f -exec du -ch {} + 2>/dev/null | tail -1

# 사용자별 용량 (홈 디렉토리 기준)
for user in $(ls /home); do 
    echo -n "$user: "; 
    sudo du -sh /home/$user 2>/dev/null; 
done
```

## 5. 디스크 quota 확인 (설정된 경우)

```bash
# 현재 사용자의 quota 확인
quota -v

# 모든 사용자의 quota 확인 (관리자 권한)
sudo repquota -a
```

## 6. 실시간 모니터링

```bash
# 대화형 디스크 사용량 분석 도구
ncdu /

# 특정 디렉토리 분석
ncdu /var
```

## 7. 로그 파일 용량 확인

```bash
# 로그 디렉토리 용량
sudo du -sh /var/log/*

# 큰 로그 파일 찾기
sudo find /var/log -type f -size +100M -exec ls -lh {} \;
```

## 8. Docker/Container 용량 (사용 중인 경우)

```bash
# Docker 디스크 사용량
docker system df

# Docker 상세 정보
docker system df -v
```

## 실용적인 조합 예시

```bash
# 루트 디렉토리에서 용량 큰 디렉토리 Top 10
sudo du -h / --max-depth=1 2>/dev/null | sort -hr | head -10

# /var 디렉토리 상세 분석
sudo du -h /var --max-depth=2 | sort -hr | head -20

# 최근 7일간 생성된 큰 파일 찾기
sudo find / -type f -mtime -7 -size +100M -exec ls -lh {} \; 2>/dev/null
```

이 명령어들을 사용하면 GCP VM의 디스크 사용 현황을 효과적으로 파악할 수 있습니다. 특히 `ncdu` 도구는 설치 후 사용하면 대화형으로 디렉토리를 탐색하며 용량을 확인할 수 있어 매우 유용합니다 (`sudo apt install ncdu` 또는 `sudo yum install ncdu`로 설치).