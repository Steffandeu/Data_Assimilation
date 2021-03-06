# データ同化Aの課題をスイスイ解くためのTips

## まずはルンゲクッタのアルゴリズムを理解しよう

微分方程式をプログラムを用いて解きたいとする。  
もちろん、微分みたいな無限小なんてプログラム上では表現できない。

じゃあどうするか？

近似的にめっちゃ小さい数を使って、「微分っぽい操作」を行う。
例えば1e-15みたいなめっちゃ小さい数など。　　

もうこれがわかればほぼ大丈夫だと思われる。  
例えば簡単にdx/dt=f(x)のような微分方程式を解くとすれば、以下の通りに式変形ができる。

~~~  
dx/dt = 3x
( x(t+1) - x(t) ) / dt = f(x)
x(t+1) = x(t) + f(x) dt
~~~

パッと思いつく形での微分方程式の数値解法はこんな感じとなる。  
ちなみにこの解法(アルゴリズム)をオイラー法という。  

今回課題として出てきているルンゲクッタ法というのは、この式において微分のところをもっと精度良く表せないか？というモチベーションで改良が行われた方法である。  

まあ所詮はオイラー法の延長のようなもので、数学的に厳密に評価しようとするととんでもないことになるが、まあ今回はそういうことはせず頭空っぽにして実装すればいいのでそこまで大変ではない。  

ちなみに dx/dt = f(x, t) をルンゲクッタ法のアルゴリズムで解くと以下の通りになる。

~~~
q1 = f(x, t)
q2 = f(x + (dt * q1)/2.0, t + dt/2.0)
q3 = f(x + (dt * q2)/2.0, t + dt/2.0)
q4 = f(x + (dt * q3), t + dt)
x = x + dt * (q1 + 2.0 * q2 + 2.0 * q3 + q4) / 6.0
~~~

先ほどのオイラー法では f(x)dt であったところが、ルンゲクッタではちょっと複雑になっているのがわかるだろう。

多少複雑になっているが、基本的に「現在時点の値から微分の速度ベクトルの分だけ少しズラす」ということがわかっていればもう大丈夫である。


## Attracterについて理解しよう

今回搭載しているモデルはLorenz96で、このモデルについては詳細を省く。  
このモデルの実際についての実装は[Wikipediaの解説](https://en.wikipedia.org/wiki/Lorenz_96_model)を参照してもらいたい。  
ここでは、Attracterとは何か、実際に実装する際にどのようなアルゴリズムを組めばいいのかということを簡単に説明する。  

Attracterとは、モデルの定常解の周辺に対してどのような作用を及ぼすのかということを示す。  
具体的に、今回のモデルにおいて定常解は x0, x1,...,xN = F である。  
実際にこの値を代入すると速度ベクトルが0となってこのモデルはこの解で停止し続ける。  
ではこの値から少しズラしてみたらxはどのように遷移するだろうか？  
つまり解の安定性について議論するということである。  

その問いについて答えとしては、このリポジトリのトップディレクトリにおいてあるplot.pyを実行することでわかる。
このLorenz96モデルは小さな摂動をキッカケとして大きくうねり出すのである。
つまり、この x0, x1,...,xN = F という解は不安定であると表現できる。  
まあこれで解の安定性については簡明には説明できるが、実を言うとこの議論の中で一つ重要なポイントが抜けている。  

それは、小さな摂動をキッカケとしてモデルを動き出させるとき、「どのように定常解からズラすか」という問題である。  

もちろん、このモデルではxはN次元あり、初期値の中で一つだけを1e-10だけズラすといってもズラす箇所はN通りもの選び方があるのだ(まあこのモデルは各次元が循環的に相互作用しあうものなので対してどれを選ぼうと大きな変化はないのだが)  

2次元以上を初期値としてズラすときも同様の議論ができ、結論として小さな摂動を与えるにあたってどのように摂動を与えるかで解がどのように振る舞うかは非自明である。  

そこで、様々な摂動の与え方を考える。実際にこのリポジトリにある実装では初期値として δ というズレを採用し x = F + δ とする。ここで、δは分散を十分に小さくした正規分布に従うN次元のノイズである。  
これにより、δについて様々な取り方を採用することでデータを複数回とり(この実装では30回程度)、各サンプリングに対して平均をとることでこの解の振る舞いについて結論を出すこととしている。  

Attracterについてザッと定性的に説明すると以上の通りである。  
基本思想としては「ノイズを加えてたくさんサンプリングしてそれらの平均をとることで有意なデータを得よう」というものである。  
この思想はデータ同化全般において通じるところで、今後の章ではカルマンフィルタやアンサンブルカルマンフィルタなどを解説していくが、ものすごい数のノイズを仮定していく。  

とりあえずこの章についてはこの程度で結びとする。


## カルマンフィルタって？

## アンサンブルカルマンフィルタ。強そう。

## 3次元変分法、こいつは強そうだが実はショボいんだ...

## 実際に現場でも採用されている4次元変分法
