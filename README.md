# エコー動画分析
PFMTデバイス開発における，測定アルゴリズムの技術検証用

## 目的（完成の定義）
ひとまずの機能は以下の完成を目指す。
- エコー~~動画~~画像（変更：10/21‗TRIGGER出張による）における膀胱底挙上量を正としてPoCプローブの測定値を用いて検量線が引けている
- エコー~~動画~~画像（変更：10/21‗TRIGGER出張による）から挙上量を判定する際の膀胱底判定アルゴリズムが完成している
- ~~エコー動画をMモード化する機能ができている~~（不要につき削除：10/21‗TRIGGER出張の際）
- 半田先生測定の挙上量とのICCが算出されている

## やること
- コード書いてできそうかできなさそうかを判定
- 半田先生からフィードバックを得る

## ファイル概要
- m_mode_slicer.py：エコー動画をMモード画像化できる。エコー動画の冒頭画像をもとに処理断面及びビーム径に相当する処理幅を設定できる
- elevation_metet.py：エコー動画からアルゴリズムを通した膀胱底位置を検定でき，動画と同じ時間軸での検定結果を動画ファイルなどで出力できる。
- ICC_calculator.py：エコー**画像**からアルゴリズムを通した膀胱底位置を検定でき，挙上前後の対となる画像を比較し挙上量をCSVファイルとして出力できる。
