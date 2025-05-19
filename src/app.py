from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import hashlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from typing import List, Dict
import uvicorn
from contextlib import asynccontextmanager
import requests # Add import for requests

# Import sentiment analysis libraries
import torch
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def check_tor_status():
    """Checks if Tor is working by making a request to the official Tor check website."""
    tor_proxy = "socks5://127.0.0.1:9050"
    proxies = {
        "http": tor_proxy,
        "https": tor_proxy,
    }
    try:
        response = requests.get("https://check.torproject.org", proxies=proxies, timeout=10)
        if "Congratulations. This browser is configured to use Tor." in response.text:
            print("Tor is working.")
            return True
        else:
            print("Tor is not working. Response did not contain success message.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Tor is not working: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("Loading sentiment analysis model...")
    load_model()
    # Check Tor status
    print("Checking Tor status...")
    if not check_tor_status():
        # Optionally, you might want to raise an error or prevent app startup
        # For now, just printing a message.
        print("Warning: Tor is not operational. Scraping functionality might be affected.")
    yield
    # Clean up the ML models and release the resources
    # (if necessary, though for this model it might not be strictly needed
    # as it's loaded once and used throughout the app's lifecycle)

app = FastAPI(title="Comments Scraper API", lifespan=lifespan)

# Initialize sentiment analysis model
MODEL_PATH = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = None
model = None
config = None

def load_model():
    global tokenizer, model, config
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        config = AutoConfig.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

class UrlInput(BaseModel):
    url: str

class CommentResponse(BaseModel):
    comments: List[str]
    total_comments: int
    
class CommentInput(BaseModel):
    comment: str
    
class SentimentResult(BaseModel):
    results: List[Dict[str, float|str]]
    sentiment: str
    execution_time: float

def setup_driver():
    # Set up Tor proxy
    tor_proxy = "socks5://127.0.0.1:9050"
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument(f"--proxy-server={tor_proxy}")
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Set up the Chrome WebDriver with Tor proxy
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver

@app.post("/scrape-comments", response_model=CommentResponse)
async def scrape_comments(url_input: UrlInput):
    try:
        # Set up the driver
        driver = setup_driver()
        
        try:
            # Navigate to the website
            driver.get(url_input.url)
            
            # Wait for comments to load (20 seconds maximum)
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.XPATH, '//div[starts-with(@id, "div-comment-")]')))
            
            # Find all comment divs using a dynamic XPath pattern
            comment_divs = driver.find_elements(By.XPATH, '//div[starts-with(@id, "div-comment-")]/div[2]/p')
            
            # Extract text from each comment
            comments_text = []
            for div in comment_divs:
                try:
                    comment = div.text.strip()
                    if comment:  # Only add non-empty comments
                        comments_text.append(comment)
                except Exception as e:
                    continue
            
            return CommentResponse(
                comments=comments_text,
                total_comments=len(comments_text)
            )
            
        finally:
            # Always close the browser
            driver.quit()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/comment-classification")
async def classify_comment(comment_input: CommentInput):
    try:
        # Load model if not already loaded
        if model is None:
            success = load_model()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to load sentiment model")
        
        start_time = time.time()
        
        # Preprocess and analyze text
        text = preprocess(comment_input.comment)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Get ordered results
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        # Format results
        results = []
        dominant_sentiment = config.id2label[ranking[0]]
        
        for i in range(scores.shape[0]):
            label = config.id2label[ranking[i]]
            score = float(scores[ranking[i]])
            results.append({
                "rank": i+1, 
                "label": label, 
                "score": round(score, 4)
            })
        end_time = time.time()
        execution_time = round(end_time - start_time, 4)
        
        return SentimentResult(
            results=results,
            sentiment=dominant_sentiment,
            execution_time=execution_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)