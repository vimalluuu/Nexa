import requests, re

s = requests.Session()
s.headers['User-Agent'] = 'NexaDatasetCollector/1.0 (academic research; 7770vijayan@gmail.com)'

# Check pagination params and total count
r = s.get('https://standardebooks.org/ebooks?tags%5B%5D=fiction&per-page=48', timeout=30)
print('Status:', r.status_code)
matches = re.findall(r'href="(/ebooks/[a-z0-9][^"#]+)"', r.text)
unique = list(dict.fromkeys(matches))
print(f'Unique ebook slugs with per-page=48: {len(unique)}')

# Check pagination links
pages = re.findall(r'href="([^"]*page=\d+[^"]*)"', r.text)
print('Pagination links:', pages[:5])

# Check total count from header or text
total_m = re.search(r'(\d+)\s+(?:ebooks?|results?|books?)', r.text, re.I)
if total_m:
    print('Total found in text:', total_m.group(0))

# Now look at a single book's page to find the EPUB download URL
if unique:
    slug = unique[0]
    print(f'\nFetching book page: {slug}')
    import time; time.sleep(2)
    r2 = s.get(f'https://standardebooks.org{slug}', timeout=30)
    epub_links = re.findall(r'href="([^"]*\.epub[^"]*)"', r2.text)
    print('EPUB links found:', epub_links[:5])
