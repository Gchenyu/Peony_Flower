# -*- coding:utf-8 -*-

# @Author : GCY

# @Date  :   2018/10/27 11:04

# @Flie  :  main.py
import train as tr
import test as ts


if __name__ == '__main__':
    while True:
        tem = input("训练or测试？X/C:")
        if tem == 'X':
            tr.training()
        elif tem == 'C':
            ts.test()
        else:
            print('输入错误，重新输入。')
