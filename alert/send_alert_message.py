from telegram import Bot
from telegram.constants import ParseMode
import asyncio

# Thông tin bot
API_TOKEN = "7737813059:AAF-cVEnw_6em6T9DlkkS2gT6TT54RU1E6o"  # Thay bằng API Token của bot
CHAT_ID = "-4633241419"       # Thay bằng Chat ID của nhóm hoặc người dùng


# Khởi tạo bot
bot = Bot(token=API_TOKEN)

# Gửi tin nhắn văn bản
async def send_text_message(chat_id, message):
    try:
        await asyncio.wait_for(bot.send_message(chat_id=chat_id, text=message), timeout=10)
        print("Tin nhắn đã được gửi thành công!")
    except asyncio.TimeoutError:
        print("Gửi tin nhắn bị hết thời gian!")
    except Exception as e:
        print(f"Lỗi khi gửi tin nhắn: {e}")

# Gửi hình ảnh kèm tin nhắn mô tả
async def send_image_with_description(chat_id, image_path, description_message, caption=None):
    try:
        # Gửi tin nhắn mô tả trước
        await asyncio.wait_for(bot.send_message(chat_id=chat_id, text=description_message), timeout=10)
        # Gửi hình ảnh với chú thích
        with open(image_path, "rb") as image_file:
            await asyncio.wait_for(bot.send_photo(chat_id=chat_id, photo=image_file, caption=caption), timeout=10)
        print("Hình ảnh và tin nhắn mô tả đã được gửi thành công!")
    except asyncio.TimeoutError:
        print("Gửi hình ảnh hoặc tin nhắn mô tả bị hết thời gian!")
    except Exception as e:
        print(f"Lỗi khi gửi hình ảnh hoặc tin nhắn mô tả: {e}")

# Sử dụng các hàm trên
async def main():
    # Gửi tin nhắn văn bản
    # await send_text_message(CHAT_ID, "Xin chào, đây là tin nhắn thử nghiệm!")

    # Gửi hình ảnh với mô tả và chú thích
    image_path = "alert/image/2024-12-09/13.9706025583222_107.48722856374022.png"  # Đường dẫn đến tệp hình ảnh
    description_message = "Phát hiện bất thường X tại vị trí tọa độ xxx,yyy."
    await send_image_with_description(CHAT_ID, image_path, description_message, caption="")

if __name__ == "__main__":
    asyncio.run(main())
