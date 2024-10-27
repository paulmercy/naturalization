from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import spacy
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import torch
from collections import defaultdict
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class AdvancedParaphraser:
    def __init__(self):
        # Initialize spaCy
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize the translation models for back-translation
        self.models = {
            'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
            'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
            'en-de': 'Helsinki-NLP/opus-mt-en-de',
            'de-en': 'Helsinki-NLP/opus-mt-de-en'
        }
        
        # Load models and tokenizers
        self.tokenizers = {}
        self.translator_models = {}
        
        print("Loading translation models...")
        for lang_pair, model_name in self.models.items():
            self.tokenizers[lang_pair] = MarianTokenizer.from_pretrained(model_name)
            self.translator_models[lang_pair] = MarianMTModel.from_pretrained(model_name)
        
        # Common simplification patterns
        self.simplification_patterns = {
            r'\b(utilize|utilization)\b': 'use',
            r'\b(implement|implementation)\b': 'use',
            r'\b(sufficient)\b': 'enough',
            r'\b(commence)\b': 'start',
            r'\b(terminate)\b': 'end',
            r'\b(possess)\b': 'have',
            r'\b(perceive)\b': 'see',
            r'\b(regarding)\b': 'about',
            r'\b(numerous)\b': 'many',
            r'\b(attempt)\b': 'try'
        }
        
        # Initialize synonym cache
        self.synonym_cache = {}

    def get_synonyms(self, word: str, pos: str) -> List[str]:
        """Get contextually appropriate synonyms for a word"""
        cache_key = f"{word}_{pos}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        
        synonyms = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        
        # Filter synonyms by length and complexity
        filtered_synonyms = [syn for syn in synonyms if len(syn) <= len(word) + 3]
        self.synonym_cache[cache_key] = filtered_synonyms
        return filtered_synonyms

    def back_translate(self, text: str, intermediate_lang: str = 'fr') -> str:
        """Perform back-translation using the loaded models"""
        # Tokenize input text
        en_to_foreign = f'en-{intermediate_lang}'
        foreign_to_en = f'{intermediate_lang}-en'
        
        # Translate to intermediate language
        tokens = self.tokenizers[en_to_foreign](text, return_tensors="pt", padding=True)
        translated = self.translator_models[en_to_foreign].generate(**tokens)
        intermediate = self.tokenizers[en_to_foreign].decode(translated[0], skip_special_tokens=True)
        
        # Translate back to English
        tokens = self.tokenizers[foreign_to_en](intermediate, return_tensors="pt", padding=True)
        translated = self.translator_models[foreign_to_en].generate(**tokens)
        back_translated = self.tokenizers[foreign_to_en].decode(translated[0], skip_special_tokens=True)
        
        return back_translated

    def simplify_text(self, text: str) -> str:
        """Simplify complex words and phrases"""
        simplified = text
        for pattern, replacement in self.simplification_patterns.items():
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
        return simplified

    def rephrase_sentence(self, sentence: str) -> str:
        """Rephrase a single sentence using various techniques"""
        doc = self.nlp(sentence)
        words = []
        
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and len(token.text) > 3:
                synonyms = self.get_synonyms(token.text.lower(), token.pos_)
                if synonyms and random.random() < 0.3:  # 30% chance to replace with synonym
                    words.append(random.choice(synonyms))
                else:
                    words.append(token.text)
            else:
                words.append(token.text)
        
        rephrased = ' '.join(words)
        return rephrased

    def paraphrase(self, text: str, style: str = 'simple') -> str:
        """Main paraphrasing method with different styles"""
        if not text.strip():
            return ""
        
        # Split into sentences
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            # Choose paraphrasing method based on sentence length and style
            if len(sentence.split()) > 10 and random.random() < 0.4:
                # Use back-translation for longer sentences
                paraphrased = self.back_translate(sentence)
            else:
                # Use synonym replacement and restructuring
                paraphrased = self.rephrase_sentence(sentence)
            
            if style == 'simple':
                paraphrased = self.simplify_text(paraphrased)
            
            paraphrased_sentences.append(paraphrased)
        
        # Combine sentences with varied conjunctions
        conjunctions = ['and', 'also', 'plus', 'moreover', 'furthermore']
        result = []
        for i, sentence in enumerate(paraphrased_sentences):
            if i > 0 and random.random() < 0.2:  # 20% chance to add conjunction
                result.append(random.choice(conjunctions))
            result.append(sentence)
        
        return ' '.join(result)

# FastAPI setup
app = FastAPI(title="Advanced Text Paraphraser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    style: Optional[str] = "simple"

class TextResponse(BaseModel):
    original: str
    paraphrased: str

# Initialize paraphraser
paraphraser = AdvancedParaphraser()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Advanced Text Paraphraser</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-4xl font-bold text-center mb-8 text-indigo-600">Advanced Text Paraphraser</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Input Section -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4">Original Text</h2>
                    <textarea 
                        id="input-text" 
                        class="w-full h-64 p-4 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        placeholder="Enter your text here..."></textarea>
                    
                    <div class="mt-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Writing Style
                        </label>
                        <select 
                            id="style-select" 
                            class="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        >
                            <option value="simple">Simple & Clear</option>
                            <option value="casual">Casual & Friendly</option>
                            <option value="formal">Professional & Formal</option>
                        </select>
                    </div>
                    
                    <button 
                        id="paraphrase-btn"
                        class="mt-4 w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                        Paraphrase Text
                    </button>
                </div>

                <!-- Output Section -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4">Paraphrased Text</h2>
                    <div 
                        id="output-text" 
                        class="w-full h-64 p-4 border rounded-lg overflow-y-auto bg-gray-50"
                    ></div>
                    
                    <div class="flex gap-4 mt-4">
                        <button 
                            id="copy-btn"
                            class="flex-1 bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
                        >
                            Copy to Clipboard
                        </button>
                        <button 
                            id="try-again-btn"
                            class="flex-1 bg-indigo-200 text-indigo-800 py-2 px-4 rounded-lg hover:bg-indigo-300 transition-colors"
                        >
                            Try Another Version
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const inputText = document.getElementById('input-text');
            const outputText = document.getElementById('output-text');
            const paraphraseBtn = document.getElementById('paraphrase-btn');
            const copyBtn = document.getElementById('copy-btn');
            const tryAgainBtn = document.getElementById('try-again-btn');
            const styleSelect = document.getElementById('style-select');

            async function paraphraseText() {
                const text = inputText.value;
                if (!text.trim()) {
                    alert('Please enter some text first!');
                    return;
                }

                paraphraseBtn.disabled = true;
                paraphraseBtn.textContent = 'Processing...';
                outputText.textContent = 'Paraphrasing...';

                try {
                    const response = await fetch('/paraphrase', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            style: styleSelect.value
                        })
                    });

                    const data = await response.json();
                    outputText.textContent = data.paraphrased;
                } catch (error) {
                    console.error('Error:', error);
                    outputText.textContent = 'An error occurred while processing your text.';
                } finally {
                    paraphraseBtn.disabled = false;
                    paraphraseBtn.textContent = 'Paraphrase Text';
                }
            }

            paraphraseBtn.addEventListener('click', paraphraseText);
            tryAgainBtn.addEventListener('click', paraphraseText);

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
        paraphrased = paraphraser.paraphrase(request.text, request.style)
        return TextResponse(
            original=request.text,
            paraphrased=paraphrased
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
