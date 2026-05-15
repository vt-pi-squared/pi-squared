conda create --name pisquared python=3.12.12 -y
conda activate pisquared

pip install vllm dspy-ai 'litellm[proxy]' tiktoken
pip install pandas tabulate lxml python-dateutil 
pip install pymediawiki markdownify trafilatura func-timeout
pip install "bm25s[core]" PyStemmer 

echo "Setup of virtual environments completed."