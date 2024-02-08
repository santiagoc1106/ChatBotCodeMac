import selenium.webdriver as webdriver
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
def get_results(search_term):
    url = "https://www.google.com"
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)  
    browser = Chrome(options = chrome_options)
    browser.get(url)
    search_box = browser.find_element(By.CLASS_NAME, "gLFyf")
    search_box.send_keys(search_term)
    search_box.submit()
    try:
        links = browser.find_elements(By.XPATH, "//ol[@class='web_regular_results']//h3//a")
    except:
        links = browser.find_elements(By.XPATH, "//h3//a")
    results = []
    for link in links:
        href = link.get_attribute("href")
        print(href)
        results.append(href)
    return results

get_results("square root of pi")