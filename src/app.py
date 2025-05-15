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
from typing import List
import uvicorn

app = FastAPI(title="Comments Scraper API")

class UrlInput(BaseModel):
    url: str

class CommentResponse(BaseModel):
    comments: List[str]
    total_comments: int

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)