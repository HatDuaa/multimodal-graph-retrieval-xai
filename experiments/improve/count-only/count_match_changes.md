# Ca g làm đổi node được ghép (countonly_factual_r3_seed0, ảnh đúng của câu val)

Mỗi dòng: node câu (lớp số lượng) và node ảnh được ghép khi bỏ g (theo sim_ij) và khi có g (theo sim_ij + g), kèm nhóm số lượng của node ảnh. Chọn ngẫu nhiên (seed 0).

## g kéo sang node sai nhãn (cũ đúng nhãn, mới sai nhãn)

- 578350_4 "The giraffes are standing by the people outside eating.": câu >=2 people — không g: 1 people (sim 0.903) → có g: 3-4 man (sim + g 0.900, g +0.035)
- 151938_1 "A lone woman huddles under her umbrella at a picnic table in the rain.": câu 1 woman — không g: 3-4 woman (sim 0.925) → có g: 1 hair (sim + g 0.917, g +0.087)
- 145597_3 "A group of people crowded around a picnic blanket.": câu >=3 people — không g: 1 people (sim 0.902) → có g: 3-4 woman (sim + g 0.877, g +0.031)
- 500514_3 "Many people are dressed as zombies covered in blood.": câu >=3 people — không g: 1 people (sim 0.940) → có g: 3-4 man (sim + g 0.912, g +0.031)
- 142182_2 "very many teddy bears in a shop with different colors": câu >=3 teddy bear — không g: 3-4 teddy bear (sim 0.923) → có g: >=5 bear (sim + g 0.972, g +0.064)
- 543525_2 "An empty kitchen, including a refrigerator, sink, and cupboards.": câu >=2 cupboard — không g: 1 cupboard (sim 0.917) → có g: 2 cabinet (sim + g 0.882, g -0.005)
- 101053_0 "a big bus that has some people by it": câu >=2 people — không g: 1 people (sim 0.879) → có g: >=5 man (sim + g 0.908, g +0.037)
- 326168_0 "A bus passes several people riding their bicycles.": câu >=2 bicycle — không g: 1 bicycle (sim 0.934) → có g: >=5 bicycler (sim + g 0.903, g +0.037)

## g kéo sang node đúng nhãn (cũ sai nhãn, mới đúng nhãn)

- 80895_4 "Two people sitting on a dock working on their cell phones.": câu 2 people — không g: 1 man woman (sim 0.871) → có g: >=5 person (sim + g 0.833, g -0.030)
- 9077_4 "A market has many fruits and vegetables out for display.": câu >=3 fruit — không g: 1 orange (sim 0.864) → có g: >=5 fruit (sim + g 0.880, g +0.064)
- 452027_0 "a number of animals in a field near some trees": câu >=2 animal — không g: 2 gazelle (sim 0.793) → có g: >=5 animal (sim + g 0.825, g +0.037)
- 577862_3 "The two giraffes are standing next to each other outside.": câu 2 giraffe — không g: 1 baby giraffe (sim 0.829) → có g: 2 giraffe (sim + g 0.841, g +0.071)
- 37549_2 "The giraffes are looking for something good to eat.": câu >=2 giraffe — không g: 1 baby giraffe (sim 0.831) → có g: 3-4 giraffe (sim + g 0.842, g +0.035)
- 577862_1 "Two giraffes standing next to each other in a holding pen.": câu 2 giraffe — không g: 1 baby giraffe (sim 0.876) → có g: 2 giraffe (sim + g 0.905, g +0.071)
- 76822_4 "A person on skis being watched by a crowd of people.": câu >=2 ski — không g: 1 slope (sim 0.811) → có g: 3-4 skier (sim + g 0.839, g +0.035)
- 539962_2 "There are two people in the middle of the pool.": câu 2 people — không g: 1 couple (sim 0.839) → có g: 2 person (sim + g 0.804, g +0.071)

