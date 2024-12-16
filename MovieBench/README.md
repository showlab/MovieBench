# Shot-Level Annotation Generation with GPT4

We developed our Shot-Level Annotation Generation system based on MovieSeq, leveraging GPT-4 to enhance its functionality.

## Environments
```
conda create --name MovieBench python=3.10
conda activate MovieBench
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install git+https://github.com/m-bain/whisperx.git
pip install tqdm moviepy openai opencv-python

pip install -r requirement.txt
```

## Guideline


Before running the script, you need to set the ```hftoken``` and ```openaikey``` parameters in ```generate.sh``` to your respective Hugging Face and OpenAI API keys.

```Shell
sh generate.sh
```