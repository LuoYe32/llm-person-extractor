from src.crawler.crawler import Crawler, save_pages, load_pages
from src.scraper.scraper import Scraper

PAGES_FILE = "pages.json"

if __name__ == "__main__":
    import sys

    if "--load" in sys.argv:
        # Пропустить краулер, загрузить сохранённые страницы
        pages = load_pages(PAGES_FILE)
    else:
        crawler = Crawler()
        pages = crawler.crawl("https://kkglo.lenobl.ru/")
        save_pages(pages, PAGES_FILE)

    scraper = Scraper()
    persons = scraper.scrape_pages(pages)
    scraper.to_csv(persons, "result.csv")
