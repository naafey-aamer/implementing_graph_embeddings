import os
import time
import shutil
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

# Set up the WebDriver
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))

try:
    # Navigate to the STRING download page
    driver.get("https://string-db.org/cgi/download")

    # Wait for the page to load completely
    time.sleep(4)

    # Find the organism selection dropdown
    organism_input = driver.find_element(By.CSS_SELECTOR, "input[type='search']")
    organism_input.send_keys("Homo sapiens")
    organism_input.send_keys(Keys.RETURN)

    # Wait for the organism list to update
    time.sleep(5)

    # Find and click the download link for the required file
    download_link = driver.find_element(By.PARTIAL_LINK_TEXT, "9606.protein.links.v12.0.txt.gz")
    download_link.click()

    # Wait for the download to complete
    time.sleep(200)  # Adjust this time based on your network speed and file size

    # Define paths
    download_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    file_name = "9606.protein.links.v12.0.txt.gz"
    src_path = os.path.join(download_dir, file_name)
    dest_path = os.path.join(os.getcwd(), file_name)

    # Move the file from the download directory to the current directory
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"File moved to {dest_path}")
    else:
        print(f"File {file_name} not found in {download_dir}")

finally:
    # Close the WebDriver
    driver.quit()

print("File downloaded successfully.")
