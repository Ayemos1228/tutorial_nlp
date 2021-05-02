from selenium import webdriver
from time import sleep
from getpass import getpass

# Change 3 parameters
username = input('username : ')
password = getpass('password : ')
credit = input('credit : ')

base_url = 'https://app.ilect.net/course/203/manager/users'

driver = webdriver.Chrome()
driver.get(base_url)

driver.find_element_by_class_name('btn-outline-info').click()
sleep(1)
driver.find_element_by_name('login').send_keys(username)
sleep(1)
driver.find_element_by_name('password').send_keys(password)
sleep(1)
driver.find_element_by_name('commit').click()
sleep(1)
driver.get(base_url)
sleep(2)
forms = driver.find_elements_by_class_name('form-control')
# add credits
for index in range(2, len(forms), 3):
  forms[index].clear()
  forms[index].send_keys(credit)
# save
saves = driver.find_elements_by_class_name('btn-primary')
for save in saves:
  save.click()
sleep(5)

driver.close()
