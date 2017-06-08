import scrapy
import logging
from datetime import datetime

from YahooStockNews.items import YahoostocknewsItem

class YahooStockNewsSpider(scrapy.Spider):
    name = 'YahooStockNews'
    allowed_domains = ['tw.stock.yahoo.com']
    start_urls = ('https://tw.stock.yahoo.com/q/h?s=2330', )

    _pages = 0
    MAX_PAGES = 2

    def parse(self, response):
        self._pages += 1
        for link in response.xpath('//a[starts-with(@href, "/news_content/")]'):
            url = response.urljoin(link.xpath('@href').extract_first())
            title = link.xpath('text()').extract_first()
            #logging.debug('Title: ' + title + '. Get Url: ' + url)
            yield scrapy.Request(url, callback=self.parse_post, meta={'title': title})

    def parse_post(self, response):
        item = YahoostocknewsItem()
        item['title'] = response.meta.get('title')
        title = response.xpath('//span[text()="'+item['title']+'"]')
        date = title.xpath('/following::span/text()').re(r'^\d+/\d+/\d+ \d+:\d+')
        if len(date) == 0:
            # Fail to find
            date = response.xpath('//table//span/text()').re(r'^\d+/\d+/\d+ \d+:\d+$')
        item['date'] = datetime.strptime(date[0], '%Y/%m/%d %H:%M')
        content = ''
        for p in title.xpath('ancestor::table//p/text()').extract():
            content += p
        item['content'] = content
        item['url'] = response.url

        yield item
