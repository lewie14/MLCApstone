import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#Exploring the data

print(df.job.head())

#***********************************************
#print(df.age.value_counts())
'''
26     3724
27     3685
28     3583
25     3531
29     3295
24     3242
30     3149
31     2735
23     2592
32     2587
33     2206
22     1934
34     1902
35     1755
36     1583
37     1427
38     1330
21     1282
39     1172
42     1072
40     1030
41      980
20      953
43      858
44      708
45      643
19      611
46      578
47      529
48      481
49      459
50      437
51      350
52      344
18      309
56      271
54      267
55      265
57      256
53      252
59      221
58      197
60      195
61      176
62      167
63      138
64      113
65      109
66      105
67       66
68       59
69       31
110       1
109       1
'''



#***********************************************
#print(df.body_type.head(10))
#print(df.body_type.value_counts())
'''
average           14652
fit               12711
athletic          11819
thin               4711
curvy              3924
a little extra     2629
skinny             1777
full figured       1009
overweight          444
jacked              421
used up             355
rather not say      198
'''
#*******************************************************
#print(df.diet.value_counts())
'''
mostly anything        16585
anything                6183
strictly anything       5113
mostly vegetarian       3444
mostly other            1007
strictly vegetarian      875
vegetarian               667
strictly other           452
mostly vegan             338
other                    331
strictly vegan           228
vegan                    136
mostly kosher             86
mostly halal              48
strictly halal            18
strictly kosher           18
halal                     11
kosher                    11
'''
#*******************************************************
#print(df.drinks.value_counts())
'''
socially       41780
rarely          5957
often           5164
not at all      3267
very often       471
desperately      322
'''
#*******************************************************
#print(df.drugs.value_counts())
'''
never        37724
sometimes     7732
often          410
'''
#*******************************************************
#print(df.education.value_counts())
'''
graduated from college/university    23959
graduated from masters program        8961
working on college/university         5712
working on masters program            1683
graduated from two-year college       1531
graduated from high school            1428
graduated from ph.d program           1272
graduated from law school             1122
working on two-year college           1074
dropped out of college/university      995
working on ph.d program                983
college/university                     801
graduated from space camp              657
dropped out of space camp              523
graduated from med school              446
working on space camp                  445
working on law school                  269
two-year college                       222
working on med school                  212
dropped out of two-year college        191
dropped out of masters program         140
masters program                        136
dropped out of ph.d program            127
dropped out of high school             102
high school                             96
working on high school                  87
space camp                              58
ph.d program                            26
law school                              19
dropped out of law school               18
dropped out of med school               12
med school                              11
'''
#*******************************************************
#print(df.ethnicity.value_counts())
'''
white                                                                                                      32831
asian                                                                                                       6134
hispanic / latin                                                                                            2823
black                                                                                                       2008
other                                                                                                       1706
hispanic / latin, white                                                                                     1301
indian                                                                                                      1077
asian, white                                                                                                 811
white, other                                                                                                 641
pacific islander                                                                                             432
asian, pacific islander                                                                                      395
native american, white                                                                                       338
middle eastern                                                                                               329
middle eastern, white                                                                                        300
black, white                                                                                                 298
pacific islander, white                                                                                      156
hispanic / latin, other                                                                                      138
black, other                                                                                                 133
black, hispanic / latin                                                                                      119
hispanic / latin, white, other                                                                               117
black, native american, white                                                                                110
black, native american                                                                                       100
asian, other                                                                                                  95
asian, hispanic / latin                                                                                       88
native american, hispanic / latin, white                                                                      87
native american, hispanic / latin                                                                             73
asian, white, other                                                                                           69
native american                                                                                               67
asian, middle eastern, black, native american, indian, pacific islander, hispanic / latin, white, other       66
asian, black                                                                                                  59
                                                                                                           ...
asian, indian, hispanic / latin, other                                                                         1
asian, black, white, other                                                                                     1
middle eastern, black, other                                                                                   1
asian, black, native american, indian, pacific islander, white                                                 1
asian, middle eastern, native american, hispanic / latin, white                                                1
black, native american, pacific islander, hispanic / latin                                                     1
asian, middle eastern, other                                                                                   1
asian, native american, indian, pacific islander, hispanic / latin, white, other                               1
asian, black, native american, indian                                                                          1
black, native american, pacific islander, white, other                                                         1
asian, black, pacific islander, hispanic / latin, white                                                        1
asian, black, native american, indian, pacific islander, hispanic / latin                                      1
asian, indian, pacific islander, hispanic / latin, white, other                                                1
asian, middle eastern, black, white, other                                                                     1
middle eastern, black, hispanic / latin                                                                        1
middle eastern, pacific islander, hispanic / latin                                                             1
black, native american, indian, white                                                                          1
asian, black, indian, hispanic / latin, other                                                                  1
asian, middle eastern, indian, hispanic / latin, white, other                                                  1
middle eastern, black, native american, white, other                                                           1
black, native american, indian, pacific islander                                                               1
asian, black, native american, indian, hispanic / latin, white, other                                          1
asian, middle eastern, native american, pacific islander, hispanic / latin, white, other                       1
asian, middle eastern, native american, pacific islander, other                                                1
middle eastern, black, indian, pacific islander, hispanic / latin, white                                       1
middle eastern, black, native american, indian, hispanic / latin, white                                        1
middle eastern, pacific islander                                                                               1
asian, middle eastern, black, native american, indian, pacific islander, white                                 1
asian, middle eastern, black, native american, indian, pacific islander, hispanic / latin                      1
asian, middle eastern, black, pacific islander, hispanic / latin                                               1
'''
#*******************************************************
#print(df.height.value_counts())
'''
in inches e.g.
88.0       2
'''
#*******************************************************
#print(df.income.value_counts())
'''
-1          48442
 20000       2952
 100000      1621
 80000       1111
 30000       1048
 40000       1005
 50000        975
 60000        736
 70000        707
 150000       631
 1000000      521
 250000       149
 500000        48
'''
#*******************************************************
#print(df.job.value_counts())
'''
other                                7589
student                              4882
science / tech / engineering         4848
computer / hardware / software       4709
artistic / musical / writer          4439
sales / marketing / biz dev          4391
medicine / health                    3680
education / academia                 3513
executive / management               2373
banking / financial / real estate    2266
entertainment / media                2250
law / legal services                 1381
hospitality / travel                 1364
construction / craftsmanship         1021
clerical / administrative             805
political / government                708
rather not say                        436
transportation                        366
unemployed                            273
retired                               250
military                              204
'''
#*******************************************************
#print(df.offspring.value_counts())
'''
doesn&rsquo;t have kids                                7560
doesn&rsquo;t have kids, but might want them           3875
doesn&rsquo;t have kids, but wants them                3565
doesn&rsquo;t want kids                                2927
has kids                                               1883
has a kid                                              1881
doesn&rsquo;t have kids, and doesn&rsquo;t want any    1132
has kids, but doesn&rsquo;t want more                   442
has a kid, but doesn&rsquo;t want more                  275
has a kid, and might want more                          231
wants kids                                              225
might want kids                                         182
has kids, and might want more                           115
has a kid, and wants more                                71
has kids, and wants more                                 21
'''
#*******************************************************
#print(df.orientation.value_counts())
'''
straight    51606
gay          5573
bisexual     2767
'''
#*******************************************************
#print(df.pets.value_counts())
'''
likes dogs and likes cats          14814
likes dogs                          7224
likes dogs and has cats             4313
has dogs                            4134
has dogs and likes cats             2333
likes dogs and dislikes cats        2029
has dogs and has cats               1474
has cats                            1406
likes cats                          1063
has dogs and dislikes cats           552
dislikes dogs and likes cats         240
dislikes dogs and dislikes cats      196
dislikes cats                        122
dislikes dogs and has cats            81
dislikes dogs                         44
'''
#*******************************************************
#print(df.religion.value_counts())
'''
agnosticism                                   2724
other                                         2691
agnosticism but not too serious about it      2636
agnosticism and laughing about it             2496
catholicism but not too serious about it      2318
atheism                                       2175
other and laughing about it                   2119
atheism and laughing about it                 2074
christianity                                  1957
christianity but not too serious about it     1952
other but not too serious about it            1554
judaism but not too serious about it          1517
atheism but not too serious about it          1318
catholicism                                   1064
christianity and somewhat serious about it     927
atheism and somewhat serious about it          848
other and somewhat serious about it            846
catholicism and laughing about it              726
judaism and laughing about it                  681
buddhism but not too serious about it          650
agnosticism and somewhat serious about it      642
judaism                                        612
christianity and very serious about it         578
atheism and very serious about it              570
catholicism and somewhat serious about it      548
other and very serious about it                533
buddhism and laughing about it                 466
buddhism                                       403
christianity and laughing about it             373
buddhism and somewhat serious about it         359
agnosticism and very serious about it          314
judaism and somewhat serious about it          266
hinduism but not too serious about it          227
hinduism                                       107
catholicism and very serious about it          102
buddhism and very serious about it              70
hinduism and somewhat serious about it          58
islam                                           48
hinduism and laughing about it                  44
islam but not too serious about it              40
judaism and very serious about it               22
islam and somewhat serious about it             22
islam and laughing about it                     16
hinduism and very serious about it              14
islam and very serious about it                 13
'''
#*******************************************************
#print(df.sex.value_counts())
'''
m    35829
f    24117
'''
#*******************************************************
#print(df.sign.value_counts())
'''
gemini and it&rsquo;s fun to think about         1782
scorpio and it&rsquo;s fun to think about        1772
leo and it&rsquo;s fun to think about            1692
libra and it&rsquo;s fun to think about          1649
taurus and it&rsquo;s fun to think about         1640
cancer and it&rsquo;s fun to think about         1597
pisces and it&rsquo;s fun to think about         1592
sagittarius and it&rsquo;s fun to think about    1583
virgo and it&rsquo;s fun to think about          1574
aries and it&rsquo;s fun to think about          1573
aquarius and it&rsquo;s fun to think about       1503
virgo but it doesn&rsquo;t matter                1497
leo but it doesn&rsquo;t matter                  1457
cancer but it doesn&rsquo;t matter               1454
gemini but it doesn&rsquo;t matter               1453
taurus but it doesn&rsquo;t matter               1450
libra but it doesn&rsquo;t matter                1408
aquarius but it doesn&rsquo;t matter             1408
capricorn and it&rsquo;s fun to think about      1376
sagittarius but it doesn&rsquo;t matter          1375
aries but it doesn&rsquo;t matter                1373
capricorn but it doesn&rsquo;t matter            1319
pisces but it doesn&rsquo;t matter               1300
scorpio but it doesn&rsquo;t matter              1264
leo                                              1159
libra                                            1098
cancer                                           1092
virgo                                            1029
scorpio                                          1020
gemini                                           1013
taurus                                           1001
aries                                             996
pisces                                            992
aquarius                                          954
sagittarius                                       937
capricorn                                         833
scorpio and it matters a lot                       78
leo and it matters a lot                           66
aquarius and it matters a lot                      63
cancer and it matters a lot                        63
gemini and it matters a lot                        62
pisces and it matters a lot                        62
libra and it matters a lot                         52
taurus and it matters a lot                        49
sagittarius and it matters a lot                   47
aries and it matters a lot                         47
capricorn and it matters a lot                     45
virgo and it matters a lot                         41
'''
#*******************************************************
#print(df.smokes.value_counts())
'''
no                43896
sometimes          3787
when drinking      3040
yes                2231
trying to quit     1480
'''
#*******************************************************
#print(df.speaks.value_counts())
'''
english                                                                                              21828
english (fluently)                                                                                    6628
english (fluently), spanish (poorly)                                                                  2059
english (fluently), spanish (okay)                                                                    1917
english (fluently), spanish (fluently)                                                                1288
english, spanish                                                                                       859
english (fluently), french (poorly)                                                                    756
english, spanish (okay)                                                                                655
english, spanish (poorly)                                                                              609
english (fluently), chinese (fluently)                                                                 535
english (fluently), french (okay)                                                                      532
english (fluently), chinese (okay)                                                                     430
english (poorly)                                                                                       310
english, chinese                                                                                       306
english (okay)                                                                                         306
english (fluently), german (poorly)                                                                    263
english, french (poorly)                                                                               245
english (fluently), french (fluently)                                                                  215
english, french (okay)                                                                                 210
english, french                                                                                        209
english, spanish (fluently)                                                                            198
english (fluently), japanese (poorly)                                                                  184
english (fluently), chinese (poorly)                                                                   175
english (fluently), german (okay)                                                                      160
english (fluently), russian (fluently)                                                                 147
english (fluently), spanish (okay), french (poorly)                                                    143
english (fluently), spanish (fluently), french (poorly)                                                143
english (fluently), french (okay), spanish (poorly)                                                    133
english (fluently), french (poorly), spanish (poorly)                                                  130
english (fluently), italian (poorly)                                                                   115
                                                                                                     ...
english (fluently), german (okay), french (okay), japanese (okay), c++ (fluently)                        1
english (fluently), spanish (poorly), hebrew (poorly), c++ (fluently)                                    1
english (fluently), korean (okay), french (okay), spanish (poorly)                                       1
english (fluently), croatian (fluently), bengali (okay)                                                  1
english (fluently), chinese (okay), japanese (poorly), spanish (poorly)                                  1
english (okay), spanish (okay), indonesian (poorly)                                                      1
english (fluently), spanish (fluently), hebrew (fluently), dutch (okay), swedish (poorly)                1
english (fluently), russian (fluently), japanese (poorly), french (poorly)                               1
english (fluently), spanish (poorly), japanese (okay), sign language (poorly)                            1
english (fluently), turkish (okay), serbian (poorly)                                                     1
english (fluently), spanish (fluently), catalan (poorly), french (poorly)                                1
english (fluently), finnish (fluently), spanish (poorly)                                                 1
english (fluently), russian (fluently), spanish (okay), lithuanian (fluently)                            1
english (fluently), german (fluently), french (fluently), spanish (fluently), portuguese (poorly)        1
english (fluently), c++ (fluently), norwegian (poorly), yiddish (poorly), sign language (poorly)         1
english, danish (poorly), spanish (poorly)                                                               1
english, japanese (poorly), spanish (poorly), italian (poorly)                                           1
english (fluently), chinese (poorly), c++ (fluently), russian (poorly), german (poorly)                  1
english (fluently), swedish (fluently), german (okay), french (okay), japanese (poorly)                  1
english, latvian (fluently), french (okay)                                                               1
english, russian (fluently), spanish (okay), french (okay), hebrew (poorly)                              1
english, c++ (fluently), french (poorly)                                                                 1
english (fluently), japanese (okay), french (okay), hawaiian (okay)                                      1
english (fluently), french (poorly), c++                                                                 1
english (fluently), french (fluently), spanish (fluently), italian (okay), breton (poorly)               1
english (fluently), spanish (poorly), japanese (poorly), chinese (okay), french (poorly)                 1
english (poorly), japanese (poorly), c++ (fluently), yiddish (fluently), ancient greek (fluently)        1
english (fluently), tagalog (fluently), french (okay)                                                    1
english (fluently), italian (okay), portuguese (okay)                                                    1
english (fluently), french (okay), spanish (poorly), bulgarian (okay)                                    1
'''
#*******************************************************
#print(df.status.value_counts())
'''
single            55697
seeing someone     2064
available          1865
married             310
unknown              10
'''
#*******************************************************
#print(df.essay0.head())
'''
'''
#*******************************************************
