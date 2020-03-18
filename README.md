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

![Screenshot_2020-03-18_18-23-36](https://user-images.githubusercontent.com/54184905/76993369-ca972700-695d-11ea-9efb-e03ef36fa7f1.png)

* Sarı renkli kutucuklar: eğitilecek sinir ağları.
* Daire içinde x: yapılacak işler.
* Okların birleşmesi: taşınan vektörlerin birbirine bağlanması.
* Okların ayrılması: kopyalama işlemi.

* LSTM geri yayılımda düz bir şekilde cell state bulunur, geri yayılım işlemini matris çarpma işlemi kalmadan basitçe yapabiliriz.

![Screenshot_2020-03-18_18-27-04](https://user-images.githubusercontent.com/54184905/76993907-aa1b9c80-695e-11ea-98f9-1790530366bb.png)

(Ct = Cell state)

* Hücre içindeki sigmoid fonksiyonlar her bileşenin ne kadarının geçeceğine karar verir, 1 'e yakınsa çok 0 'a yakınsa az akış gerçekleşir.


# Cell state 'yi kontrol etmek için LSTM içersinde 3 tane Gate vardır

* 1-) Forget Gate:
* Bu Gate 'te hangi bilgiler unutulacak buna karar verilecek.

![Screenshot_2020-03-18_20-00-21](https://user-images.githubusercontent.com/54184905/76994467-79883280-695f-11ea-985e-6e10145e1f33.png)

* Ft fonksiyonundan çıkan değer 0 ise tamamen unutur, 1 ise tamamen hatırlar, 0.9 ise çoğunu hatırlar.


* 2-) Input gate
* Hangi bilgiler saklanacak hangileri saklanmayacak buna karar verir.
* Sigmoid kullanan Input gate hangi değerlerin güncelleneceğine karar verir.
* tanh ise yeni değerlerden bir vektör oluşturup cell state ye ekleme yapar

![Screenshot_2020-03-18_20-01-24](https://user-images.githubusercontent.com/54184905/76995140-79d4fd80-6960-11ea-8804-76378f2e8725.png)


* 3-) Output Gate

![Screenshot_2020-03-18_20-03-27](https://user-images.githubusercontent.com/54184905/76995680-45157600-6961-11ea-834b-c6bd7dbe34fa.png)

* ilk olarak verileri sigmoid fonk. dan geçiririz burada neleri output olarak alacağız onu seçeriz.
* Sonra ise Cell State 'yi tanh tan geçirip Output Gate ile çarparız
* Bu şekilde sadece istenilen kısımların output olarak verilmesini sağlarız.


* People:
* Tüm geçitlerin Cell State 'e erişmesini sağlar

