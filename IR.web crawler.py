#Develop a basic web crawler to fetch and index web pages.

#output:

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def crawl(start_url, max_pages=10):
    visited = set()
    queue = deque([start_url])
    index = {}

    while queue and len(visited) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except (requests.RequestException, ValueError):
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        # Save page content to index
        text = soup.get_text()
        index[url] = text[:500]  # Index first 500 characters

        print(f"Crawled: {url}")
        visited.add(url)

        # Extract and normalize links
        for link_tag in soup.find_all('a', href=True):
            link = urljoin(url, link_tag['href'])
            if is_valid_url(link) and link not in visited:
                queue.append(link)

    return index

# Example usage
if __name__ == "__main__":
    start_url = 'https://example.com'
    index = crawl(start_url, max_pages=5)

    print("\n--- Indexed Pages ---")
    for url, content in index.items():
        print(f"\nURL: {url}\nContent Snippet: {content}\n")

