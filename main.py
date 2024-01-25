#!/usr/bin/env python3
import serial
import time

line = "Sended"

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        x = input("Enter Pan and Tilt separated with a semicolon (;): ")
        b = x.encode('utf-8')
        ser.write(str(x).encode('utf-8'))
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        time.sleep(1)
