# 필요한 도구 설치
apt-get update && apt-get install -y wget gnupg

# Google의 공식 키를 다운로드하여 추가
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -

# Chrome 저장소를 시스템에 추가
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list

# 패키지 목록을 업데이트하고 Chrome 설치
apt-get update
apt-get install -y google-chrome-stable