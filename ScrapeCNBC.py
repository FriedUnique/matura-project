from Scraper import Scraper    

import pickle
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import NoSuchElementException

from time import sleep, time
from datetime import datetime, date, timedelta
from typing import List, Tuple

EMAIL = "ochs.stef.2018@ksz.edu-zg.ch"
PASS = "Matura2024!"



def WriteToFile(fileName: str, data: List[str]):
    with open(fileName, "w+") as f:
        for d in data:
            f.write("\n" + d)


class CNBC(Scraper):
    def __init__(self, query: str, articleLimit: int):
        super().__init__()

        self.query = query;
        self.maxDate = date.today() - timedelta(days=365) # two years back
        self.pageLimit = 100 # for debugging
        self.articleLimit = articleLimit

    def Login(self):
        # /html/body/div[3]/div/div[1]/header/div[2]/div/div/div[3]/span[4]/div/div/div/div/a
        # email: /html/body/div[3]/div[1]/div/div/div/div[2]/form/div[1]/div/div/input
        # pass:  /html/body/div[3]/div[1]/div/div/div/div[2]/form/div[2]/div/div/input
        # submit: /html/body/div[3]/div[1]/div/div/div/div[2]/form/button[1]

        self.driver.get("https://www.cnbc.com")
        sleep(2)
        loginPageButton = self.driver.find_element(By.XPATH, "/html/body/div[3]/div/div[1]/header/div[2]/div/div/div[3]/span[4]/div/div/div/div/a")
        loginPageButton.click()
        sleep(1)

        self.driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div/div/div/div[2]/form/div[1]/div/div/input").send_keys(EMAIL)
        sleep(0.2)
        self.driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div/div/div/div[2]/form/div[2]/div/div/input").send_keys(PASS)
        sleep(1)
        self.driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div/div/div/div[2]/form/button[1]").click()

        sleep(5)


    def BuildURL(self, query):
        return f"https://www.cnbc.com/search/?query={query}"

    def ExtractArticleText(self):
        #return super().ExtractArticleText()
        i = 0

        self.driver.get(self.BuildURL(self.query))
        sleep(3)
        screen_height = self.driver.execute_script("return window.screen.height;")
        self.driver.execute_script("queryly.search.switchsort('date');")

        while True:
            self.driver.execute_script(f"window.scrollTo(0, {100000*i});")

            i += 1
            sleep(2)

            articlesLoaded = self.driver.find_element(By.CSS_SELECTOR, "div#searchcontainer").find_elements(By.CSS_SELECTOR, "div.SearchResult-searchResultContent")


            if len(articlesLoaded) >= self.articleLimit:
                break

        masterLinksList, dates = self.GetLinks()

        linksAlreadySeen = []
        with open("linksSaveFile.txt", "r") as f:
            linksAlreadySeen = f.readlines()

        for i, link in enumerate(masterLinksList):
            if link in linksAlreadySeen:
                masterLinksList.remove(link)

        
        print("Links extraction done")
        WriteToFile("linksSaveFile.txt", masterLinksList)
        articleInformation = self.ExtractTextFromLinks(masterLinksList, dates)
        print(articleInformation)
        WriteToFile("articles.txt", articleInformation)
        print("articleInformation extraction done")

    def GetLinks(self):
        """Returns a list of links and wether the limit (date or pages) is reached"""
        links: List[str] = []

        results = self.driver.find_elements(By.CSS_SELECTOR, "div.SearchResult-searchResultContent")
        links = [res.find_element(By.CSS_SELECTOR, "a.resultlink").get_attribute("href") for res in results]

        for link in links:
            if "video" in link:
                continue


        dates = [res.find_element(By.CSS_SELECTOR, "span.SearchResult-publishedDate").text for res in results]

        return links, dates

    def ExtractTextFromLinks(self, links: List[str], dates):
        articleInformations: List[str] = []

        for i, link in enumerate(links):
            sleep(2)
            self.driver.get(link)

            try:
                isClub = "investingclub" in self.driver.find_element(By.XPATH, "/html/body/div[3]/div/div[1]/div[3]/div/div/div/div[2]/div/div/header/div/div[1]/a").get_attribute("href")
                if isClub: 
                    print("a")
                    continue
            
            except NoSuchElementException:
                pass
        
            sleep(2)

            try:
            
                parent = self.driver.find_element(By.XPATH, "/html/body/div[3]/div/div[1]/div[3]/div/div/div/div[3]/div[1]/div/div/div[2]")
                paragraphs = "".join([p.text for p in parent.find_elements(By.TAG_NAME, "p")])
                paragraphs = paragraphs.replace("\n", "")

                if paragraphs == "":
                    continue

                title = self.driver.find_element(By.XPATH, "/html/body/div[3]/div/div[1]/div[3]/div/div/div/div[2]/div/div/header/div/div[1]/div[1]/h1").text


                m, d, y = dates[i].split(" ")[0].split("/")

                saveString = f"{m},{d},{y}|{title}|{paragraphs}"

                articleInformations.append(saveString)
            except NoSuchElementException:
                pass

        return articleInformations


    def CheckDate(self, dateString: str):
        """dateString: (month, day, year)"""
        m, d, y = dateString[0], dateString[1], dateString[2]

        try:
            if self.maxDate < date(int(y), self.MONTHS[m], int(d)): 
                print("Date limit reached for Reuters!")
                return True
        except ValueError:
            pass

        return False