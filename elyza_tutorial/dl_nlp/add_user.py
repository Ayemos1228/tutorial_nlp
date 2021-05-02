from selenium import webdriver
from time import sleep
from getpass import getpass

def read_users(path):
    users = [line.strip() for line in open(path, 'r').readlines()]
    return users

username = input('username : ')
password = getpass('password : ')
users = read_users(input('user file path : '))

driver = webdriver.Chrome()

base_url = 'https://app.ilect.net/course/203'
driver.get(base_url)

driver.find_element_by_class_name('btn-outline-info').click()
sleep(1)
driver.find_element_by_name('login').send_keys(username)
sleep(1)
driver.find_element_by_name('password').send_keys(password)
sleep(1)
driver.find_element_by_name('commit').click()
sleep(1)

driver.get(base_url + '/manager/users')
sleep(5)

form = driver.find_element_by_class_name('form-control')
save = driver.find_element_by_class_name('btn')

for user in users:
    form.clear()
    form.send_keys(user)
    save.click()
    sleep(2)
sleep(5)

driver.close()
