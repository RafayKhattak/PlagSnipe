# PlagSnipe - AI-Powered Plagiarism Detection Tool
PlagSnipe is an advanced AI-powered plagiarism detection tool that leverages the power of GPT-2 language model and the natural language processing capabilities of NLTK to determine whether a given text was authored by AI or not. By analyzing perplexity and burstiness, PlagSnipe provides accurate insights into the authorship of the text, distinguishing between human-written and AI-generated content.
![Screenshot (458)](https://github.com/RafayKhattak/PlagSnipe/assets/90026724/a992ae75-86fe-4318-9777-bd5c7ddaf21e)

## Features
- Measure perplexity: PlagSnipe calculates the perplexity of the input text using GPT-2 language model. Perplexity is a metric used to measure the quality of a language model's predictions. Higher perplexity values indicate more complex or less likely sequences, which are often associated with AI-generated text.
- Analyze burstiness: PlagSnipe analyzes the burstiness score of the input text using NLTK. Burstiness refers to the extent of repetition or the number of repeated words in the text. AI-generated text often exhibits low burstiness compared to human-written text.
- Determine AI-generated content: Based on the calculated perplexity and burstiness score, PlagSnipe determines whether the input text is likely to be AI-generated or not.
- Visualize most repeated words: PlagSnipe generates a bar chart visualization of the top 10 most repeated words in the input text, providing further insights into the text's content.
## Requirements
- Python 3.6 or above
- Streamlit
- Transformers (Hugging Face)
- Torch
- NLTK
- Plotly
- Collections
## Installation
1. Clone the repository and navigate to the project directory:
```
git clone https://github.com/yourusername/plagsnipe.git
cd plagsnipe
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run app.py
```

