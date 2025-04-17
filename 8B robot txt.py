#pip install requests beautifulsoup4 selenium webdriver-manager
import requests
import time
import re
from urllib.robotparser import RobotFileParser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Function to check if a URL is allowed by robots.txt
def is_allowed_by_robots(url):
    robots_url = url + "/robots.txt"
    try:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", url)
    except Exception as e:
        print(f"Error with robots.txt: {e}")
        return True

# Function to get HTML content of a page (handles dynamic content using Selenium)
def get_dynamic_html(url):
    # Set up the Chrome WebDriver using Selenium
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run headless (without opening browser)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(url)  # Open the URL
        time.sleep(3)  # Wait for the page to load fully (adjust if needed)
        page_source = driver.page_source  # Get the dynamically loaded content
        driver.quit()
        return page_source
    except Exception as e:
        print(f"Error loading dynamic content: {e}")
        driver.quit()
        return None

# Function to extract links from HTML content
def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            links.append(href)
        else:
            links.append(base_url + href)
    return links

# Function to crawl a website
def crawl(start_url, max_depth=2, delay=2):
    visited_urls = set()

    def recursive_crawl(url, depth):
        if depth > max_depth or url in visited_urls or not is_allowed_by_robots(url):
            return

        visited_urls.add(url)
        print(f"Crawling: {url}")
        
        # Get HTML content (handle dynamic content)
        html = get_dynamic_html(url) if 'dynamic' in url else requests.get(url).text
        
        if html:
            links = extract_links(html, url)
            for link in links:
                recursive_crawl(link, depth + 1)
                
        time.sleep(delay)  # Delay between requests

    # Start crawling from the initial URL
    recursive_crawl(start_url, 1)

# Example usage
start_url = "https://example.com"
crawl(start_url, max_depth=2, delay=3)

