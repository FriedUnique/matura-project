from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class Scraper:
    def __init__(self):
        self.MONTHS =   {"January": 1, "February": 2, "March": 3, 
                        "April": 4, "May": 5, "June": 6, 
                        "July": 7, "August": 8, "September": 9, 
                        "October": 10, "November": 11, "December": 12}

        options = Options()
        #options.add_argument("--headless")
        options.add_argument("--log-level=3")
        options.add_argument("--ignore-certificate-error")
        options.add_argument("--ignore-ssl-errors")
        options.add_experimental_option("detach", True)
        options.add_argument("--disable-blink-features=AutomationControlled") # Adding argument to disable the AutomationControlled flag  
        options.add_experimental_option("excludeSwitches", ["enable-automation"]) # Exclude the collection of enable-automation switches 
        options.add_experimental_option("useAutomationExtension", False) # Turn-off userAutomationExtension 

        self.driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

    def ScraperMain(self):
        self.Login()
        self.ExtractArticleText()

    def Login(self):
        pass

    def ExtractArticleText(self):
        pass