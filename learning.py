#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 18:39:56 2021

@author: yuuma
"""

import dlib

#学習用のオプション変数の生成.
options = dlib.simple_object_detector_training_options()

#認識はSVMアルゴリズムを使っている。それのコストパラメータを設定.
options.C = 2

#学習処理を行うときの実行スレッドの数.
options.num_threads = 10

#値を小さくすると、トレーナーのソルバーはより正確になりますが、トレーニングに時間がかかる場合があります。

epsilon=0.001

#HOG特徴画像上にフィルターを畳み込む数

nuclear_norm_regularization_strength=100

#左右反転のイメージは生成しない（鯉が一方向に進むだけなので）
options.add_left_right_image_flips = True

#学習処理の経過を出力する（ターミナルで確認可能）
options.be_verbose = True

#矩形のXMLデータから学習データを作成.
dlib.train_simple_object_detector("final.xml", "detector.svm", options)

#学習データから認識オブジェクトの作成.
detector = dlib.simple_object_detector("detector.svm")