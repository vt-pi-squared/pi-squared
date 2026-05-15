from mediawiki import MediaWiki
USER_AGENT = "search-and-aggregate (qtq.the.channel@gmail.com)" # for our project.
wikipedia = MediaWiki(user_agent=USER_AGENT)

"""
Sources:
https://en.wikipedia.org/wiki/Wikipedia:Top_25_Report/Records
https://en.wikipedia.org/wiki/Portal:Current_events
https://en.wikipedia.org/wiki/Wikipedia:Contents

https://en.wikipedia.org/wiki/Category:Wikipedia_Top_25_Reports_in_2015 (from 2012 to 2026) are good sources of seed pages for 2-hop crawling.

Recursively take links and backlinks three times, starting from these three pages
"""

seed_pages = [
    "Wikipedia:Top_25_Report/Records",
    "Portal:Current_events",
    "Wikipedia:Contents/Reference",
    "Wikipedia:Contents/Culture and the arts",
    "Wikipedia:Contents/Geography and places",
    "Wikipedia:Contents/Health and fitness",
    "Wikipedia:Contents/History and events",
    "Wikipedia:Contents/Human activities",
    "Wikipedia:Contents/Mathematics and logic",
    "Wikipedia:Contents/Natural and physical sciences",
    "Wikipedia:Contents/People and self",
    "Wikipedia:Contents/Philosophy and thinking",
    "Wikipedia:Contents/Religion and belief systems",
    "Wikipedia:Contents/Society and social sciences",
    "Wikipedia:Contents/Technology and applied sciences",
]

# Such crawling with 3 iterations may be a stress for the Wikipedia servers
# So let's cherry pick a few seed pages, and do 1 iteration only.
for i in range(1):
    new_pages = set(seed_pages)
    for page in seed_pages:
        try:
            wiki_page = wikipedia.page(page)
            new_pages.update(wiki_page.links)
            new_pages.update(wiki_page.backlinks)
        except Exception as e:
            print(f"Error fetching page '{page}': {e}")
    seed_pages = set(new_pages)
    print(f"Iteration {i+1}: {len(seed_pages)} pages collected.")

# Save the collected page titles to a file
import random
random.seed(2026)

seed_pages = list(seed_pages)
random.shuffle(seed_pages)
with open("datasets/ours/v2/collected_wiki_pages.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(seed_pages))
print(f"Save collected page titles to 'datasets/ours/v2/collected_wiki_pages.txt'")
