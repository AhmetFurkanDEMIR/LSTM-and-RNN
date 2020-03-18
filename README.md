# LSTM and RNN 

işleyeceğimiz konular sırayla:

1-) Yinelenen sinir ağı (RNN)

2-) Uzun kısa süreli bellek (LSTM)


# Yinelenen sinir ağı (RNN)

* Yinelenen sinir ağları döngüsel olarak çalışır. Bu sayede önceden gördüğü şeyleri aklında tutarak ona göre sonuç verir.
* Konuşma tanıma, doğal dil işleme, çeviri yapma, ritim öğrenme, müzik üretme gibi bir çok farklı alanda kullanılır.

![Screenshot_2020-03-18_17-47-58](https://user-images.githubusercontent.com/54184905/76989085-de8b5a80-6956-11ea-80dc-0089b81a06ac.png)

X = input

A = Sinir ağı

H = Output

![Screenshot_2020-03-18_17-49-41](https://user-images.githubusercontent.com/54184905/76989202-0a0e4500-6957-11ea-8b6f-2e3d9dfd8791.png)

(ht yani yeni state in oluşumu)


# RNN ile bazı örnekler

![Screenshot_2020-03-18_17-50-52](https://user-images.githubusercontent.com/54184905/76989633-bb14df80-6957-11ea-94f0-862776c3f318.png)

(Hello kelimesinden yola çıkarak yeni kelime üretme)

![Screenshot_2020-03-18_17-51-41](https://user-images.githubusercontent.com/54184905/76989635-bbad7600-6957-11ea-91ec-d52df9f38e83.png)

(William Shakespeare eserlerini kullanarak yeni bir eser üretmek)

![Screenshot_2020-03-18_17-52-56](https://user-images.githubusercontent.com/54184905/76989637-bc460c80-6957-11ea-9540-aab4fd4139a9.png)

(C++ kodunu örnek alarak yeni bir kod yazmak)


# Uzun kısa süreli bellek (LSTM)

* RNN deki back propagation sorununu çözer.
* Bir sinir ağı yerine dört tane birbiri ie ilişkilenmiş sinir ağı vardır (3 'ü sigmoid fonk. kullanır, biri tanh fonk. kullanır)

