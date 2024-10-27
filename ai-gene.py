# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import spacy
from typing import List, Dict, Optional
import random
from pathlib import Path
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextParaphraser:
    def __init__(self):
        # Load spaCy model for English
        self.nlp = spacy.load('en_core_web_sm')
        
        # Common simple words to replace complex ones
        self.simple_replacements = {
            "utilize": "use",
            "implement": "use",
            "facilitate": "help",
            "acquire": "get",
            "obtain": "get",
            "purchase": "buy",
            "sufficient": "enough",
            "demonstrate": "show",
            "illustrate": "show",
            "comprehend": "understand",
            "expedite": "speed up",
            "commence": "start",
            "terminate": "end",
            "optimize": "improve",
            "endeavor": "try",
            "inquire": "ask",
            "numerous": "many",
            "initiate": "start",
            "utilize": "use",
            "subsequently": "then",
            "approximately": "about",
            "additional": "more",
            "assist": "help",
            "require": "need",
            "construct": "build"
        }
        
        # Informal transitions
        self.informal_transitions = [
            "Also",
            "Plus",
            "By the way",
            "Another thing",
            "On top of that",
            "Speaking of that",
            "Not only that",
            "And then",
            "So",
            "But",
            "Actually"
        ]

    def get_synonyms(self, word: str) -> List[str]:
        """Get simpler synonyms for a word"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and len(lemma.name()) < len(word):
                    synonyms.append(lemma.name())
        return list(set(synonyms))

    def simplify_sentence(self, sentence: str) -> str:
        """Simplify a sentence while maintaining its meaning"""
        doc = self.nlp(sentence)
        words = []
        
        for token in doc:
            # Skip punctuation
            if token.is_punct:
                words.append(token.text)
                continue
                
            word = token.text.lower()
            
            # Check if we have a simple replacement
            if word in self.simple_replacements:
                words.append(self.simple_replacements[word])
            else:
                # Try to find a simpler synonym
                synonyms = self.get_synonyms(word)
                if synonyms and len(min(synonyms, key=len)) < len(word):
                    words.append(min(synonyms, key=len))
                else:
                    words.append(token.text)
        
        return ' '.join(words)

    def break_long_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones"""
        sentences = sent_tokenize(text)
        result = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:
                doc = self.nlp(sentence)
                clauses = []
                current_clause = []
                
                for token in doc:
                    current_clause.append(token.text)
                    if token.dep_ in ['cc', 'punct'] and len(current_clause) > 5:
                        clauses.append(' '.join(current_clause))
                        current_clause = []
                
                if current_clause:
                    clauses.append(' '.join(current_clause))
                
                result.extend(clauses)
            else:
                result.append(sentence)
        
        return ' '.join(result)

    def add_conversational_elements(self, text: str) -> str:
        """Add conversational elements to make text more natural"""
        sentences = sent_tokenize(text)
        result = []
        
        for i, sentence in enumerate(sentences):
            # Randomly add transitions
            if i > 0 and random.random() < 0.3:
                transition = random.choice(self.informal_transitions)
                sentence = f"{transition}, {sentence.lower()}"
            
            result.append(sentence)
        
        return ' '.join(result)

    def paraphrase(self, text: str, simplicity_level: float = 0.7) -> str:
        """Main method to paraphrase text"""
        if not text.strip():
            return ""
            
        # Preserve formatting
        paragraphs = text.split('\n')
        result_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                result_paragraphs.append('')
                continue
                
            # Process the paragraph
            simplified = self.break_long_sentences(paragraph)
            simplified = self.simplify_sentence(simplified)
            
            # Add conversational elements based on simplicity level
            if random.random() < simplicity_level:
                simplified = self.add_conversational_elements(simplified)
            
            result_paragraphs.append(simplified)
        
        return '\n'.join(result_paragraphs)

# FastAPI app setup
app = FastAPI(title="Text Paraphraser")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    simplicity_level: Optional[float] = 0.7

class TextResponse(BaseModel):
    original: str
    paraphrased: str

# Initialize paraphraser
paraphraser = TextParaphraser()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simple Text Paraphraser</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-4xl font-bold text-center mb-8 text-blue-600">Simple Text Paraphraser</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Input Section -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4">Original Text</h2>
                    <textarea 
                        id="input-text" 
                        class="w-full h-64 p-4 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Enter your text here..."></textarea>
                    
                    <div class="mt-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Simplicity Level
                            <span id="slider-value" class="ml-2">70%</span>
                        </label>
                        <input 
                            type="range" 
                            id="simplicity-level" 
                            min="0" 
                            max="100" 
                            value="70"
                            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        >
                    </div>
                    
                    <button 
                        id="paraphrase-btn"
                        class="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
                    >
                        Paraphrase Text
                    </button>
                </div>

                <!-- Output Section -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4">Simplified Text</h2>
                    <div 
                        id="output-text" 
                        class="w-full h-64 p-4 border rounded-lg overflow-y-auto bg-gray-50"
                    ></div>
                    
                    <button 
                        id="copy-btn"
                        class="mt-4 w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
                    >
                        Copy to Clipboard
                    </button>
                </div>
            </div>
        </div>

        <script>
            const slider = document.getElementById('simplicity-level');
            const sliderValue = document.getElementById('slider-value');
            const inputText = document.getElementById('input-text');
            const outputText = document.getElementById('output-text');
            const paraphraseBtn = document.getElementById('paraphrase-btn');
            const copyBtn = document.getElementById('copy-btn');

            slider.addEventListener('input', (e) => {
                sliderValue.textContent = `${e.target.value}%`;
            });

            paraphraseBtn.addEventListener('click', async () => {
                const text = inputText.value;
                if (!text.trim()) {
                    alert('Please enter some text first!');
                    return;
                }

                paraphraseBtn.disabled = true;
                paraphraseBtn.textContent = 'Processing...';

                try {
                    const response = await fetch('/paraphrase', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            simplicity_level: parseInt(slider.value) / 100
                        })
                    });

                    const data = await response.json();
                    outputText.textContent = data.paraphrased;
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred while processing your text.');
                } finally {
                    paraphraseBtn.disabled = false;
                    paraphraseBtn.textContent = 'Paraphrase Text';
                }
            });

            copyBtn.addEventListener('click', () => {
                const text = outputText.textContent;
                navigator.clipboard.writeText(text).then(() => {
                    const originalText = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => {
                        copyBtn.textContent = originalText;
                    }, 2000);
                });
            });
        </script>
    </body>
    </html>
    """

@app.post("/paraphrase", response_model=TextResponse)
async def paraphrase_text(request: TextRequest):
    try:
        paraphrased = paraphraser.paraphrase(request.text, request.simplicity_level)
        return TextResponse(
            original=request.text,
            paraphrased=paraphrased
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

