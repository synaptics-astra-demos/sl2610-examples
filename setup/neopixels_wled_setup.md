# Sparkle Motion WLED Setup

A few demos use a NeoPixel led strip. 

The NeoPixels are driven by a LED driver running WLED software. 

The SL2619 communicates with the LED driver over a USB CDC (virtual serial port).

The specific module tested is: [Sparkle Motion Mini](https://www.adafruit.com/product/6160) from Adafruit.


## Setup Instructions


1. Plug in the Sparkle Motion Mini USB type-C to your computer.

1. Visit https://install.wled.me/

1. Choose Basic, Plain. Then click install.

1. Follow the pop-up instructions to erase the existing firmware and install WLED.

1. With WLED installed, the device will now broadcast a Wi-Fi access point (WLED-AP). Connect to it from a mobile phone or computer.

1. After connecting to the WLED-AP access point, the configuration page should automatically appear. Choose `TO THE CONTROLS`

1. Select `Config`, `LED & Hardware`. 

  - For `Length` choose the number of LEDs, such as `36`. 

  - For `Data GPIO`, change to `33`. Click Save. 



For more information see [Adafruit Instructions](https://learn.adafruit.com/adafruit-sparkle-motion-mini/wled-software-2).

