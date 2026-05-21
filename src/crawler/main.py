from src.crawler.crawler import Crawler


if __name__ == "__main__":
    crawler = Crawler()

    # pages = crawler.crawl("https://zdrav74.ru/")
    pages = crawler.crawl("https://digital.gov74.ru/digital.htm")
    # pages = crawler.crawl("https://kkglo.lenobl.ru/")

    for p in pages:
        print(p.url, len(p.links))
