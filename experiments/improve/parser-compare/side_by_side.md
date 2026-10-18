# 20 câu val ngẫu nhiên qua các parser (seed 0)

Đồ thị phía câu đúng như parser trả ra (trước khi lọc đại từ). Dạng: vật thể; quan hệ chủ ngữ —vị từ→ tân ngữ.

## 1. `230151_0` — two cat getting ready to get into a fight

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | cat; fight | ∅ |
| factual-base | cat ×2 | ∅ |
| factual-large | cat ×2 | ∅ |
| qwen2.5-3b | cat ×2 | ∅ |

## 2. `30012_3` — A freshly baked pizza that has not been eaten yet.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | pizza; that | ∅ |
| factual-base | pizza | ∅ |
| factual-large | pizza | ∅ |
| qwen2.5-3b | pizza ×1; pizza; un eat | ∅ |

## 3. `138838_3` — The man is flying a kite and someone is laying in the sand.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | man; kite; someone; sand | man —flying→ kite; someone —in→ sand |
| factual-base | man; kite; person; sand | man —fly→ kite; person —lay in→ sand |
| factual-large | man; kite; person; sand | man —fly→ kite; person —lay in→ sand |
| qwen2.5-3b | man; kite; sand | man —flying→ kite; man —laying next to→ sand |

## 4. `173161_1` — A man and woman walking huddled under an umbrella.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | walking; umbrella | walking —under→ umbrella |
| factual-base | man; umbrella; woman | man —walk under→ umbrella; woman —walk under→ umbrella |
| factual-large | man; umbrella; woman | man —huddle under→ umbrella; woman —huddle under→ umbrella |
| qwen2.5-3b | man; woman; umbrella | man —huddled under→ umbrella; woman —huddled under→ umbrella |

## 5. `376371_3` — Cows and brown lambs grassing in a green field.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | cow; lamb; field | cow —in→ field; lamb —in→ field |
| factual-base | cow; field; lamb | cow —grass in→ field; lamb —grass in→ field |
| factual-large | cow; field; lamb | cow —grass in→ field; lamb —grass in→ field |
| qwen2.5-3b | cow; lamb; field; grass | cow —grassing→ lamb |

## 6. `132615_3` — A person on a field with a baseball bat.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | person; field; baseball bat | person —on→ field; person —with→ baseball bat |
| factual-base | person; field; baseball bat | person —on→ field; person —with→ baseball bat |
| factual-large | person; field; baseball bat | person —on→ field; person —with→ baseball bat |
| qwen2.5-3b | person; baseball bat | person —on→ baseball bat |

## 7. `105974_3` — A man jumps to catch a white Frisbee.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | man; frisbee | ∅ |
| factual-base | frisbee; man | man —catch→ frisbee |
| factual-large | frisbee; man | man —catch→ frisbee |
| qwen2.5-3b | man; frisbee | man —catches→ frisbee; man —jumps to→ frisbee |

## 8. `497593_3` — A small coffee cup and a plate of food.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | coffee cup; plate; food | plate —of→ food |
| factual-base | coffee cup; plate; food | coffee cup —on top of→ plate; plate —have→ food |
| factual-large | coffee cup; plate; food | coffee cup —on top of→ plate; plate —have→ food |
| qwen2.5-3b | coffee cup ×1; plate ×1 | ∅ |

## 9. `519916_0` — People playing beach tennis on a cool day

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | people; beach tennis; day | people —playing→ beach tennis; people —on→ day |
| factual-base | day; people; tennis; beach | people —play→ tennis; people —play on→ beach |
| factual-large | people; beach; tennis | people —on→ beach; people —play→ tennis |
| qwen2.5-3b | person; tennis | person —playing→ tennis |

## 10. `283254_0` — A man in a tie sitting in front of a ashtray.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | man; tie; ashtray | man —in→ tie |
| factual-base | man; ashtray; tie | man —sit in front of→ ashtray; man —wear→ tie |
| factual-large | man; ashtray; tie | man —sit in front of→ ashtray; man —wear→ tie |
| qwen2.5-3b | man; tie; ashtray | man —sitting in front of→ tie |

## 11. `412621_0` — A person that is doing a trick in the air.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | person; that; trick; air | that —doing→ trick; trick —in→ air |
| factual-base | person; air | person —trick in→ air |
| factual-large | person; air | person —trick in→ air |
| qwen2.5-3b | person | ∅ |

## 12. `551086_2` — A woman in a pink hat is looking at a cellphone.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | woman; hat; cellphone | woman —in→ hat; woman —at→ cellphone |
| factual-base | hat; woman; cellphone | woman —look at→ cellphone; woman —wear→ hat |
| factual-large | hat; woman; cellphone | woman —look at→ cellphone; woman —wear→ hat |
| qwen2.5-3b | woman; hat; cellphone | woman —looking at→ hat |

## 13. `161447_0` — A woman and three men at a table with most of the pizza gone.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | woman; man; table; pizza | man —at→ table |
| factual-base | man ×3; table; pizza; woman | man ×3 —at→ table; woman —at→ table |
| factual-large | man ×3; table; pizza; woman | man ×3 —at→ table; pizza —on top of→ table; woman —at→ table |
| qwen2.5-3b | woman ×1; man ×3; table; pizza | woman ×1 —at→ table |

## 14. `323295_1` — A group of elephants in grassy field next to a fence.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | group; elephant; field; fence | group —of→ elephant; elephant —in→ field; group —next to→ fence |
| factual-base | elephant; field; fence | elephant —in→ field; field —next to→ fence |
| factual-large | elephant; field; fence | elephant —in→ field; field —next to→ fence |
| qwen2.5-3b | elephant; field; fence | elephant —in→ field; elephant —next to→ fence |

## 15. `52790_1` — a close up of a train on a train track near plant life

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | close; train; train track; plant life | close —on→ train track; train track —near→ plant life |
| factual-base | train; plant life; train track | train —near→ plant life; train —on→ train track |
| factual-large | train; plant life; train track | train —near→ plant life; train —on→ train track |
| qwen2.5-3b | train ×1; train track ×1; plant | train ×1 —near→ plant |

## 16. `248152_4` — A person walks along the seashore flying a kite.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | person; seashore; kite | person —along→ seashore |
| factual-base | person; kite; seashore | person —fly→ kite; person —walk along→ seashore |
| factual-large | person; kite; seashore | person —fly→ kite; person —walk along→ seashore |
| qwen2.5-3b | person; kite; seashore | person —flying→ seashore |

## 17. `359945_3` — An apple is carved with facial features and teeth

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | apple; feature; tooth | ∅ |
| factual-base | apple; facial feature; tooth | apple —with→ facial feature; apple —with→ tooth |
| factual-large | apple; facial feature; tooth | apple —carve with→ facial feature; apple —carve with→ tooth |
| qwen2.5-3b | apple ×1 | ∅ |

## 18. `542205_3` — Close-up picture of a pigeon on a ledge with people in background.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | picture; pigeon; ledge; people; background | picture —of→ pigeon; pigeon —on→ ledge; ledge —with→ people; people —in→ background |
| factual-base | people; background; pigeon; ledge | people —in→ background; pigeon —on→ ledge |
| factual-large | people; background; pigeon; ledge | people —in→ background; pigeon —on→ ledge |
| qwen2.5-3b | pigeon; ledge; person | pigeon —on→ ledge; pigeon —in background of→ person |

## 19. `468411_3` — a cat sitting next to a glass window

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | cat; glass window | cat —next to→ glass window |
| factual-base | cat; window | cat —sit next to→ window |
| factual-large | cat; window | cat —sit next to→ window |
| qwen2.5-3b | cat ×1; window | cat ×1 —sitting next to→ window |

## 20. `25005_2` — A cross country skier skies on a field of snow.

| Parser | Vật thể | Quan hệ |
|---|---|---|
| sng | skier sky; field; snow | skier sky —on→ field; field —of→ snow |
| factual-base | skier; snow | skier —ski on→ snow |
| factual-large | skier; snow | skier —ski on→ snow |
| qwen2.5-3b | skier; cross country ski; field; snow | skier —skies on→ field |
