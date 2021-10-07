from io import BytesIO
import win32clipboard
def send_to_clipboard(image):
    """https://stackoverflow.com/questions/34322132/copy-image-to-clipboard"""
    output = BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()