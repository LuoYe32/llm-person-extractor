from src.crawler.crawler import Crawler, save_pages, load_pages
from src.scraper.scraper import Scraper

crawler = Crawler()
# pages = crawler.crawl("https://kkglo.lenobl.ru/")

# save_pages(pages, "pages.json")
# save_pages(pages, "pages_full.json", include_html=True)
pages = load_pages("pages.json")


scraper = Scraper()
persons = scraper.scrape_pages(pages)

scraper.to_csv(persons, "result.csv")
