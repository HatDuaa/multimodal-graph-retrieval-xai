# Kiểm tra bằng mắt: gộp trùng và gom nhóm (30 ảnh train ngẫu nhiên, seed 0)

Hàng trên: trước, gộp ở IoU ≥ 0,5 / 0,7 / 0,9 (node được gộp vẽ đậm, ghi số khung đã gộp; khung xám là node không đổi). Hàng dưới: gom nhóm trên bản gộp 0,7 theo (a) vị từ, (b) hình học, (a) hoặc (b), và gần nhau (node nhóm vẽ đậm, ghi "tên x số lượng").

## Ảnh 2368068

![2368068](img/overview_2368068.png)

## Ảnh 2373519

![2373519](img/overview_2373519.png)

## Ảnh 2316224

![2316224](img/overview_2316224.png)

## Ảnh 2349174

![2349174](img/overview_2349174.png)

## Ảnh 2387259

![2387259](img/overview_2387259.png)

## Ảnh 2383519

![2383519](img/overview_2383519.png)

## Ảnh 2371152

![2371152](img/overview_2371152.png)

## Ảnh 2355778

![2355778](img/overview_2355778.png)

## Ảnh 2382097

![2382097](img/overview_2382097.png)

## Ảnh 2363994

![2363994](img/overview_2363994.png)

## Ảnh 2398111

![2398111](img/overview_2398111.png)

## Ảnh 2343077

![2343077](img/overview_2343077.png)

## Ảnh 2386277

![2386277](img/overview_2386277.png)

## Ảnh 2331163

![2331163](img/overview_2331163.png)

## Ảnh 2352614

![2352614](img/overview_2352614.png)

## Ảnh 2331244

![2331244](img/overview_2331244.png)

## Ảnh 2324451

![2324451](img/overview_2324451.png)

## Ảnh 2403512

![2403512](img/overview_2403512.png)

## Ảnh 2347921

![2347921](img/overview_2347921.png)

## Ảnh 2390501

![2390501](img/overview_2390501.png)

## Ảnh 2416243

![2416243](img/overview_2416243.png)

## Ảnh 2400931

![2400931](img/overview_2400931.png)

## Ảnh 2332335

![2332335](img/overview_2332335.png)

## Ảnh 2356797

![2356797](img/overview_2356797.png)

## Ảnh 2325047

![2325047](img/overview_2325047.png)

## Ảnh 2321168

![2321168](img/overview_2321168.png)

## Ảnh 2413105

![2413105](img/overview_2413105.png)

## Ảnh 2359901

![2359901](img/overview_2359901.png)

## Ảnh 2381356

![2381356](img/overview_2381356.png)

## Ảnh 2394450

![2394450](img/overview_2394450.png)

# Ví dụ cho từng tiêu chí gom nhóm (gộp trùng ở IoU 0,7)

## (a) cùng vị từ với cùng node neo

- Ảnh 2325328: bear ['grooming', 'measuring'] / bear ['grooming', 'measuring'] — neo bear

  ![a_group 2325328](img/a_group_2325328.png)

- Ảnh 2363398: mountain ['on'] / mountain ['on'] — neo snow

  ![a_group 2363398](img/a_group_2363398.png)

- Ảnh 2357752: candle ['on'] / candle ['on'] — neo window sill

  ![a_group 2357752](img/a_group_2357752.png)

## (b) chỉ cùng hình học (VG thiếu nhãn quan hệ)

- Ảnh 2325328: bear ['grooming', 'measuring'] / bear [] — neo bear

  ![b_only_group 2325328](img/b_only_group_2325328.png)

- Ảnh 2363398: cloud [] / cloud ['in'] — neo sky

  ![b_only_group 2363398](img/b_only_group_2363398.png)

- Ảnh 2357752: floor ['on'] / floor [] — neo short

  ![b_only_group 2357752](img/b_only_group_2357752.png)

## mâu thuẫn: cùng vị từ, khác hình học

- Ảnh 2325328: bear ['grooming', 'measuring'] / bear ['grooming', 'measuring'] — neo bear

  ![a_not_b 2325328](img/a_not_b_2325328.png)

- Ảnh 2363398: mountain ['on'] / mountain ['on'] — neo snow

  ![a_not_b 2363398](img/a_not_b_2363398.png)

- Ảnh 2357752: candle ['on'] / candle ['on'] — neo window sill

  ![a_not_b 2357752](img/a_not_b_2357752.png)

## mâu thuẫn: cùng hình học, khác vị từ

- Ảnh 2363398: mountain ['has'] / mountain ['on'] — neo snow

  ![b_not_a 2363398](img/b_not_a_2363398.png)

- Ảnh 2382154: water ['crouched over in', 'in'] / water ['above'] — neo man

  ![b_not_a 2382154](img/b_not_a_2382154.png)

- Ảnh 2319468: teeth ['brushing h'] / teeth ['brushing'] — neo man

  ![b_not_a 2319468](img/b_not_a_2319468.png)

