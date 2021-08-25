# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 12:43:23 2021

@author: Beth
"""

l  = [8,2,3,0,7]

def sum_list(list):
    sum = 0
    for i in l:
        sum = sum + i
    return(sum)
sum_list(l)

def multiplication(list):
    mult = 1
    for i in list:
        mult = mult * i
    return(mult)
l2 = [8,2,3,-1,7]
multiplication(l2)
