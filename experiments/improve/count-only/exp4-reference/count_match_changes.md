# Ca g làm đổi node được ghép (groupcount_factual_r3_seed0, ảnh đúng của câu val)

Mỗi dòng: node câu (lớp số lượng) và node ảnh được ghép khi bỏ g (theo sim_ij) và khi có g (theo sim_ij + g), kèm nhóm số lượng của node ảnh. Chọn ngẫu nhiên (seed 0).

## g kéo sang node sai nhãn (cũ đúng nhãn, mới sai nhãn)

- 163601_3 "Three clocks set to different times suspended from an arch.": câu 3-4 clock — không g: >=5 clock (sim 0.944) → có g: 3-4 clock face (sim + g 1.032, g +0.111)
- 451949_2 "A pair of tennis players smile in an historical photo.": câu 2 tennis player — không g: 1 tennis (sim 0.708) → có g: 2 racket (sim + g 0.767, g +0.073)
- 491169_4 "A group of motorcycles sitting on the tar.": câu >=3 motorcycle — không g: 1 motorcycle (sim 0.810) → có g: >=5 bike (sim + g 0.816, g +0.092)
- 186856_1 "A group of people standing around each other in a pool.": câu >=3 people — không g: 1 people (sim 0.775) → có g: >=5 child (sim + g 0.789, g +0.092)
- 118065_1 "A produce stand with several fruits including bananas and oragnes.": câu >=3 fruit — không g: 1 fruit (sim 0.920) → có g: 3-4 orange (sim + g 0.888, g +0.040)
- 11760_0 "A group of zebras walking away from trees.": câu >=2 tree — không g: 1 tree (sim 0.556) → có g: 2 branch (sim + g 0.521, g -0.000)
- 41247_0 "people enjoy swimming in the waves of the ocean on a sunny day at the beach.": câu >=2 wave — không g: 1 wave (sim 0.875) → có g: 2 person (sim + g 0.829, g -0.000)
- 491169_0 "A motorcycle on a gravel area with people in the background.": câu >=2 people — không g: 1 people (sim 0.842) → có g: >=5 man (sim + g 0.847, g +0.053)

## g kéo sang node đúng nhãn (cũ sai nhãn, mới đúng nhãn)

- 76822_4 "A person on skis being watched by a crowd of people.": câu >=2 ski — không g: 1 ski track (sim 0.767) → có g: 3-4 skier (sim + g 0.787, g +0.039)
- 36607_3 "Two people talking while one holds a mobile phone.": câu 2 people — không g: 3-4 man (sim 0.755) → có g: 2 person (sim + g 0.824, g +0.073)
- 282847_0 "A group of young men playing a game of frisbee.": câu >=3 man — không g: 3-4 shirt (sim 0.814) → có g: >=5 man (sim + g 0.898, g +0.092)
- 37549_2 "The giraffes are looking for something good to eat.": câu >=2 giraffe — không g: 1 baby giraffe (sim 0.818) → có g: 3-4 adult giraffe (sim + g 0.848, g +0.039)
- 573770_3 "two people snowboarding on a hill by some trees": câu >=2 tree — không g: 1 tree branches (sim 0.848) → có g: >=5 tree (sim + g 0.898, g +0.053)
- 537695_0 "A cow has a ring in its nose near two men.": câu 2 man — không g: 1 shirt (sim 0.855) → có g: 2 human (sim + g 0.922, g +0.073)
- 206876_3 "Children helping clean an elephants back with a brush": câu >=2 elephant — không g: 1 trunk (sim 0.775) → có g: 2 elephant (sim + g 0.760, g -0.000)
- 577182_3 "a bench with a couple of purple benches": câu 2 bench — không g: 1 information (sim 0.848) → có g: 3-4 bench (sim + g 0.873, g +0.057)

