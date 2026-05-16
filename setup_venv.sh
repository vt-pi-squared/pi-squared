conda create --name pisquared python=3.12.12 -y
conda activate pisquared

pip install vllm dspy-ai 'litellm[proxy]' tiktoken
pip install pandas tabulate lxml python-dateutil 
pip install pymediawiki markdownify=0.11.6 trafilatura func-timeout
pip install "bm25s[core]" PyStemmer 

# install openserp for web search. ddgs is not stable enough for scaling.
wget https://go.dev/dl/go1.26.3.linux-amd64.tar.gz # https://go.dev/dl/
tar -xzf go1.26.3.linux-amd64.tar.gz
export LOCAL_GO_PATH=$(pwd)
alias go="$LOCAL_GO_PATH/go/bin/go"

wget https://github.com/karust/openserp/archive/refs/tags/v0.7.2.zip
unzip v0.7.2.zip
mv openserp-0.7.2 openserp
cd openserp
go build -o openserp .
# ./openserp serve -p $PORT: Run to serve

echo "Setup of virtual environments completed."