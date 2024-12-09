import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from PIL import Image
from datetime import datetime
import time
import math


def calculate_distance(coord1, coord2):
    """
    Tính khoảng cách lệch theo mét giữa hai tọa độ.

    Args:
        coord1 (tuple): Tọa độ thứ nhất (latitude, longitude).
        coord2 (tuple): Tọa độ thứ hai (latitude, longitude).

    Returns:
        dict: Khoảng cách lệch theo vĩ độ và kinh độ (m),
              và hướng lệch (trái/phải, lên/xuống).
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat_distance = (lat2 - lat1) * 111320
    lon_distance = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))

    lat_direction = "lên trên" if lat_distance > 0 else "xuống dưới"
    lon_direction = "sang trái" if lon_distance > 0 else "sang phải"
    if lat_distance != 0:
        print(f"Lệch {round(lat_distance)}m {lat_direction}")
    if lon_distance != 0:
        print(f"Lệch {round(math.fabs(lon_distance))}m {lon_direction}")
    return {
        "lat_distance": abs(lat_distance),
        "lon_distance": abs(lon_distance),
        "lat_direction": lat_direction,
        "lon_direction": lon_direction
    }

def adjust_coordinates(latitude, longitude, offset):
    """
    Điều chỉnh tọa độ theo khoảng cách lệch bằng mét.

    Args:
        latitude (float): Vĩ độ ban đầu.
        longitude (float): Kinh độ ban đầu.
        offset (dict): Định nghĩa lệch bao gồm:
            - "up_down" (str): "up" để lệch lên, "down" để lệch xuống.
            - "left_right" (str): "left" để lệch sang trái, "right" để lệch sang phải.
            - "distance" (float): Khoảng cách lệch theo mét.

    Returns:
        tuple: Tọa độ đã điều chỉnh (latitude, longitude).
    """
    print(f"Tọa độ gốc: {latitude},{longitude}")
    lat_offset_meters = 0
    lon_offset_meters = 0

    if offset["up_down"] == "up":
        lat_offset_meters = offset["lat_distance"]
    elif offset["up_down"] == "down":
        lat_offset_meters = -offset["lat_distance"]

    if "left_right" in offset and offset["left_right"] == "left":
        lon_offset_meters = -offset["lon_distance"]
    elif "left_right" in offset and offset["left_right"] == "right":
        lon_offset_meters = offset["lon_distance"]
    # 111320 là số mét tương ứng với 1 độ kinh độ tại xích đạo
    adjusted_latitude = latitude + (lat_offset_meters / 111320)
    adjusted_longitude = longitude + (lon_offset_meters / (111320 * math.cos(math.radians(latitude))))
    print(f"Tọa độ sau điều chỉnh: {adjusted_latitude},{adjusted_longitude}")
    return adjusted_latitude, adjusted_longitude


def setup_firefox_driver(geckodriver_path, firefox_binary_path):
    options = Options()
    options.binary_location = firefox_binary_path
    options.headless = True
    service = Service(geckodriver_path)
    driver = webdriver.Firefox(service=service, options=options)
    return driver


def open_google_maps(driver):
    driver.get("https://www.google.com/maps")
    time.sleep(3)


def search_coordinates(driver, latitude, longitude):
    button = driver.find_element(By.CLASS_NAME, "yHc72.qk5Wte")
    button.click()
    time.sleep(4)
    search_box = driver.find_element(By.ID, "searchboxinput")
    search_box.clear()
    search_box.send_keys(f"{latitude}, {longitude}")
    search_box.send_keys(Keys.RETURN)
    time.sleep(5)


def zoom_in_map(driver, times=3):
    zoom_in_button = driver.find_element(By.ID, "widget-zoom-in")
    for _ in range(times):
        zoom_in_button.click()
        time.sleep(1)


def save_screenshot(driver, latitude, longitude, save_path):
    os.makedirs(save_path, exist_ok=True)
    screenshot_path = f"{save_path}{latitude}_{longitude}.png"
    driver.save_screenshot(screenshot_path)
    return screenshot_path


def crop_screenshot(screenshot_path):
    image = Image.open(screenshot_path)
    width, height = image.size
    cropped_image = image.crop((0, 100, width, height - 200))
    cropped_image.save(screenshot_path)


def get_map_image():
    geckodriver_path = "alert/geckodriver.exe"
    firefox_binary_path = "C:\\Program Files\\Mozilla Firefox\\firefox.exe"
    formatted_time = datetime.now().strftime("%Y-%m-%d")
    save_path = "alert/image/" + formatted_time + "/"
    latitude = 13.97059357521045
    longitude = 107.48724707760834
    # Điều chỉnh tọa độ (ví dụ lệch 1m lên trên, 2m sang phải)
    offset = {
        "up_down": "up",  # Lệch lên trên
        "left_right":  "left", # Lệch sang trái
        "lat_distance": 1,  # Lệch lên trên
        "lon_distance": 2   # Lệch sang phải
    }

    # Adjust coordinates
    latitude, longitude = adjust_coordinates(latitude, longitude, offset)

    # Set up driver
    driver = setup_firefox_driver(geckodriver_path, firefox_binary_path)

    try:
        open_google_maps(driver)
        search_coordinates(driver, latitude, longitude)
        zoom_in_map(driver)
        screenshot_path = save_screenshot(driver, latitude, longitude, save_path)
        crop_screenshot(screenshot_path)
        print(f"Ảnh chụp màn hình đã được lưu tại {screenshot_path}")
    finally:
        driver.quit()


if __name__ == "__main__":
    get_map_image()
    # calculate_distance((13.97059357521045,107.48724707760834), (13.9706025583222,107.48722856374022))
