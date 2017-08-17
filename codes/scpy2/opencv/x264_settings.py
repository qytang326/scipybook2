import winreg

X264_KEY = r"Software\GNU\x264"


def set_quantizer(value):
    hkey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, X264_KEY, 0, winreg.KEY_ALL_ACCESS)
    winreg.SetValueEx(hkey, "quantizer", None, winreg.REG_DWORD, value)