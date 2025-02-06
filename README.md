# Cassiopeia


1. Install web scraping tools
```bash
# Install Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y

# Install required dependencies
sudo apt-get install -y xvfb

pip install selenium webdriver-manager xvfbwrapper
```

2. Install ChromeDriver for Chrome
```bash
# Check Chrome version
google-chrome --version

# https://googlechromelabs.github.io/chrome-for-testing/#stable
# Download matching ChromeDriver
wget https://storage.googleapis.com/chrome-for-testing-public/$(google-chrome --version)/linux64/chrome-linux64.zip

# Unzip and move to /usr/local/bin/
unzip chrome-linux64.zip
sudo mv chrome-linux64 /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/chrome-linux64
```
