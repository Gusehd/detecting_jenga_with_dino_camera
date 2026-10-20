"""Dino-Lite LED on/off on Linux.

The LED is switched by writing 3 bytes (80 01 f0 = off, 80 01 f1 = on) to
extension unit 4, selector 3. The camera reports that control as 1 byte long,
so uvcvideo rejects the write (ENOBUFS). We bypass uvcvideo and send the UVC
SET_CUR request straight over USB with pyusb.

Needs write access to the USB device (udev rule for idVendor a168), and most
Dino-Lite firmware only acts on it while video is streaming.
"""

from __future__ import annotations

import usb.core

VENDOR_ID = 0xA168
XU_UNIT = 4
XU_SELECTOR = 3
SET_CUR = 0x01
LED_OFF = bytes([0x80, 0x01, 0xF0])
LED_ON = bytes([0x80, 0x01, 0xF1])

# Class request; device recipient first because usbfs lets it through while
# uvcvideo owns the interface, interface recipient as the standard fallback.
REQUEST_TYPES = (0x20, 0x21)


def _device() -> usb.core.Device:
    dev = usb.core.find(idVendor=VENDOR_ID)
    if dev is None:
        raise RuntimeError("Dino-Lite not found on USB")
    return dev


def set_led(on: bool) -> int:
    """Send the LED command. Returns the bmRequestType that was accepted."""
    dev = _device()
    payload = LED_ON if on else LED_OFF
    errors = []
    for request_type in REQUEST_TYPES:
        try:
            dev.ctrl_transfer(request_type, SET_CUR, XU_SELECTOR << 8, XU_UNIT << 8, payload)
            return request_type
        except usb.core.USBError as e:
            errors.append(f"{request_type:#04x}: {e}")
    raise RuntimeError("LED command rejected: " + "; ".join(errors))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Turn the Dino-Lite LED on or off")
    parser.add_argument("state", choices=["on", "off"])
    args = parser.parse_args()
    request_type = set_led(args.state == "on")
    print(f"LED {args.state} (bmRequestType {request_type:#04x})")


if __name__ == "__main__":
    main()
