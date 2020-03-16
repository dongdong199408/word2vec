#!/usr/bin/env python3
# coding: utf-8
# File: demo.py.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-25
import codecs
import os
class DataLoader:
    def __init__(self,file):
        self.datafile = os.path.join('data',file)#'data/data.txt'
        self.dataset = self.load_data()

    '''加载数据集'''
    def load_data(self):
        dataset = []
        input = codecs.open(self.datafile, 'r', 'utf-8')
        for line in input.readlines():
            line = line.strip().split(',')
            dataset.append([word for word in line[1].split(' ') if 'nbsp' not in word and len(word) < 11 and word !='' and word !='\n'])
        return dataset
