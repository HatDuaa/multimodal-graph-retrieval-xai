# Ví dụ ghép node bằng FGW nửa nới lỏng (vector CLIP đóng băng, ảnh đúng)

Mỗi node của câu được nối với node ảnh nhận nhiều khối lượng vận chuyển nhất (phần trăm khối lượng của node câu đó).

- `331799_0` A woman standing in front of a box handing a woman a bag of food. — điểm −FGW -0.062
  - bộ ba câu: bag → food; woman → bag; woman → box
  - ghép: bag → bag (100%); food → customer (100%); woman → woman (100%); box → box (100%)
- `331799_1` A blond lady with a flowered shirt passes out watermelon. — điểm −FGW -0.004
  - bộ ba câu: lady → watermelon; lady → shirt
  - ghép: lady → woman (100%); watermelon → watermelon (100%); shirt → shirt (100%)
- `331799_3` A woman wearing plastic gloves handing out fruit slices from behind a table. — điểm −FGW -0.104
  - bộ ba câu: woman → table; woman → fruit; woman → glove
  - ghép: fruit → fruit (100%); glove → glove (100%); woman → woman (100%); table → table (100%)
- `331799_4` The woman is handing a package to another person. — điểm −FGW -0.132
  - bộ ba câu: person → package; woman → package
  - ghép: person → boy (100%); package → box (100%); woman → woman (100%)
- `440508_1` An orange train is on the track with many cars behind it. — điểm −FGW -0.043
  - bộ ba câu: car → train; train → track
  - ghép: car → car (100%); train → car (100%); track → track (100%)
- `440508_2` Multiple orange train engines on tracks with trees on each side. — điểm −FGW -0.012
  - bộ ba câu: engine → track; tree → track
  - ghép: engine → engine (100%); track → track (100%); tree → tree (100%)
- `440508_3` Orange train on tracks in the country side. — điểm −FGW -0.058
  - bộ ba câu: train → country side; train → track
  - ghép: train → car (100%); country side → tree (100%); track → track (100%)
- `440284_2` Three teenage boys wearing bike helmets posing with their bicycles. — điểm −FGW -0.056
  - bộ ba câu: boy → bicycle; boy → helmet
  - ghép: boy → man (100%); bicycle → bicycle (100%); helmet → helmet (100%)
