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

Instructions for obtaining API keys:

**Infura URL:**
1. Go to https://infura.io and sign up
2. Create new project from dashboard
3. Select "Web3 API" 
4. Copy the HTTP endpoint URL that looks like: `https://mainnet.infura.io/v3/YOUR-PROJECT-ID`

**Etherscan API Key:**
1. Visit https://etherscan.io/apis
2. Create free account
3. Click "Add" in API Keys section
4. Name your API key and click "Create New Key"
5. Copy the generated key

Code setup example:
```python
web3_url = "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"
etherscan_key = "YOUR-ETHERSCAN-KEY"

analyzer = OnChainAnalyzer(
    web3_url=web3_url,
    etherscan_api_key=etherscan_key
)
```

Important: Keep these keys secure and never commit them to version control. Use environment variables or a config file instead.