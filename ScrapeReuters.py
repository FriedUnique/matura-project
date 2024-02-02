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
PASS = "Maturaarbeit2024!"



def WriteToFile(fileName: str, data: List[str]):
    with open(fileName, "w+") as f:
        for d in data:
            f.write("\n" + d)


class Reuters(Scraper):
    def __init__(self, query):
        super().__init__()

        self.query = query;
        self.maxDate = date.today() - timedelta(days=365) # two years back
        self.pageLimit = 100 # for debugging

        
        # cookies = pickle.load(open("cookies.pkl", "rb"))
        # for cookie in cookies:
        #     self.driver.add_cookie(cookie)

    def Login(self):
        loginPageURL = "https://www.reuters.com/account/sign-in/?redirect=https%3A%2F%2Fwww.reuters.com%2F"
        self.driver.get(loginPageURL)

        wait = WebDriverWait(self.driver, 1) 
        emailInput  = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[2]/main/div/div[2]/div[1]/div/div/div/form/div[1]/div/div/input")))
        emailInput.send_keys(EMAIL)

        passInput   = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[2]/main/div/div[2]/div[1]/div/div/div/form/div[2]/div/div/input")))
        passInput.send_keys(PASS)

        sleep(2)

        # accept cookies
        wait = WebDriverWait(self.driver, 5, 1.5)
        cookiesButton = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[1]")))
        cookiesButton.click()

        signInButton = self.driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/main/div/div[2]/div[1]/div/div/div/form/button")
        signInButton.click()


    def BuildURL(self, query, page):
        return f"https://www.reuters.com/site-search/?query={query}&offset={page*20}"
    

    def TrySome(self):
        self.driver.get(self.BuildURL(self.query, 0))
        seenLink = []
        allSaveStrings = []

        for i in range(20):

            try:
                ul_element = self.driver.find_element(By.CSS_SELECTOR, "ul.search-results__list__2SxSK")
                li_elements = ul_element.find_elements(By.CSS_SELECTOR, "li")
            except NoSuchElementException:
                print("No more elemtens")
                return
            
            sleep(5)
            
            li_elements[i].click()
            sleep(3)
            articleSaveString = self.GetText()
            sleep(2)

            if articleSaveString != None:
                allSaveStrings.append(articleSaveString)
                print(articleSaveString)

                with open(f"{self.query}_save.txt", "w+") as f:
                    f.write("\n"+articleSaveString)

            self.driver.back()


    def GetText(self):
        try: # if they wana block me again
            pageHeaderElement = self.driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/div")
            
            if pageHeaderElement == None:
                return None

            title = self.driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/h1").text
            dateString = pageHeaderElement.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/div/div[1]/time/span[1]").text

            # continue to next page, because podcasts have no use to us.
            if str(title).startswith("Podcast: "):
                return None

            try:
                articleElement = self.driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/div/div/div[2]")
            except NoSuchElementException:
                return None

            paragraphs = articleElement.find_elements(By.TAG_NAME, "p")
            articleBody = " ".join([p.text for p in paragraphs])

            try:
                dateString = dateString.replace(",", "").split(" ")
                m, d, y = dateString[0], dateString[1], dateString[2]

                saveString = f"{m},{d},{y}|{title}|{articleBody}"
                print(saveString)
                return saveString
            except IndexError:
                print("indexError")

        except Exception:
            print("Some thing else went wrong")
        
        return None






    def ExtractArticleText(self):
        self.TrySome()








    # def ExtractArticleText(self):
    #     print(self.driver.current_url)
    #     sleep(15)

    #     pickle.dump(self.driver.get_cookies(), open("cookies.pkl", "wb"))

    #     masterLinksList = []

    #     page = 0

    #     while True:
    #         self.driver.get(self.BuildURL(self.query, page))
    #         sleep(10)

    #         links, isLimitReached = self.GetLinks()

    #         masterLinksList += links

    #         # a limit is either a date limit or there are no more articles to be read
    #         if isLimitReached or page >= self.pageLimit:
    #             print("Breaking out of loop")
    #             break

    #         page += 1
    #         print(f"{page}, ", end="\r")
        
    #     print("Links extraction done")
    #     WriteToFile("linksSaveFile.txt", masterLinksList)
    #     articleInformation = self.ExtractTextFromLinks(masterLinksList)
    #     WriteToFile("articleInformationCompact.txt", articleInformation)
    #     print("articleInformation extraction done")

    def GetLinks(self) -> Tuple[List[str], bool]:
        """Returns a list of links and wether the limit (date or pages) is reached"""
        links: List[str] = []
        isDateLimit = False

        try:
            ul_element = self.driver.find_element(By.CSS_SELECTOR, "ul.search-results__list__2SxSK")
            li_elements = ul_element.find_elements(By.CSS_SELECTOR, "li")
        except NoSuchElementException:
            print("No more elemtens")
            return links, False


        for li in li_elements:
            link = li.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
            
            dateString = li.find_element(By.CSS_SELECTOR, "time").text
            if dateString.split(" ")[0] in list(self.MONTHS.keys()):
                dateString = dateString.replace(",", "").split(" ")
                isDateLimit = self.CheckDate(dateString)

            # remove duplicates
            if link in links:
                continue

            links.append(link)
        return links, isDateLimit

    def ExtractTextFromLinks(self, links: List[str]):
        articleInformations: List[str] = []

        for page in links:
            try: # if they wana block me again
                sleep(5)
                self.driver.get(page)

                pageHeaderElement = self.driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/div")
                
                if pageHeaderElement == None:
                    continue

                title = self.driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/h1").text
                dateString = pageHeaderElement.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/div/div[1]/time/span[1]").text

                # continue to next page, because podcasts have no use to us.
                if str(title).startswith("Podcast: "):
                    continue

                try:
                    articleElement = self.driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/main/article/div[1]/div/div/div/div[2]")
                except NoSuchElementException:
                    continue

                paragraphs = articleElement.find_elements(By.TAG_NAME, "p")
                articleBody = " ".join([p.text for p in paragraphs])

                try:
                    dateString = dateString.replace(",", "").split(" ")
                    m, d, y = dateString[0], dateString[1], dateString[2]

                    saveString = f"{m},{d},{y}|{title}|{articleBody}"
                    print(saveString)
                    articleInformations.append(saveString)
                except IndexError:
                    pass

            except Exception:
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