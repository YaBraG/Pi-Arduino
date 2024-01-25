#!/usr/bin/env python3
import serial
import time

line = "Sended"

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        x=input(b"Enter Pan and Tilt separated with a semicolon (;)")
        ser.write(b""+x)
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        time.sleep(1)
