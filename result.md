###### Process:  my_model_run
【text_block】
Edit_dist:
---------------  ---------
ALL_page_avg     0.0744374
edit_whole       0.065055
edit_sample_avg  0.0342366
---------------  ---------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
-------------------------------  ---------
formula_type: print              0.269479
text_background: multi_colored   0.0198424
text_background: single_colored  0.0359496
text_background: white           0.0336407
text_language: other             0.176265
text_language: text_english      0.0339617
text_rotate: normal              0.0346311
text_rotate: other               0
text_rotate: rotate270           0.0223955
text_rotate: rotate90            0
-------------------------------  ---------
====================================================================================================
sample_count:
-------------------------------  -----
formula_type: print                  6
text_background: multi_colored    1520
text_background: single_colored  17087
text_background: white           10722
text_language: other                51
text_language: text_english      29271
text_rotate: normal              28419
text_rotate: other                  57
text_rotate: rotate270             844
text_rotate: rotate90                2
-------------------------------  -----
====================================================================================================
Edit_dist:
-----------------------------------------  ----------
ALL                                        0.0744374
challenge_type:                            0.0282486
challenge_type: domain_reasoning           0.00172159
challenge_type: perception                 0.0367761
challenge_type: structural_reconstruction  0.0919088
colorful_backgroud                         0.0616862
data_source: academic_literature           0.0693524
data_source: book                          0.0725234
data_source: colorful_textbook             0.0701213
data_source: exam_paper                    0.00374817
data_source: magazine                      0.0925131
data_source: note                          0.0427807
english                                    0.0977856
french                                     0.0889431
fuzzy_scan                                 0.0820578
language: english                          0.0496474
layout: 1andmore_column                    0.114777
layout: double_column                      0.0918685
layout: other_layout                       0.0719409
layout: single_column                      0.0692645
layout: three_column                       0.0198687
other                                      0.00656477
subject: ARCHITECTURE                      0.009745
subject: ART                               0
subject: BIOGRAPHY&AUTOBIOGRAPHY           0.00389118
subject: BODY,MIND&SPIRIT                  0.0120331
subject: BUSINESS&ECONOMICS                0.0031236
subject: COMICS&GRAPHICNOVELS              0.0106005
subject: COMPUTERS                         0.070442
subject: COOKING                           0.128411
subject: CRAFTS&HOBBIES                    0.114686
subject: DESIGN                            0.0538976
subject: EDUCATION                         0.0467
subject: FAMILY&RELATIONSHIPS              0.0715753
subject: FICTION                           0
subject: GAMES&ACTIVITIES                  0.128898
subject: GARDENING                         0.107735
subject: HISTORY                           0.0685339
subject: HOUSE&HOME                        0.0164049
subject: HUMOR                             0.0384867
subject: JUVENILENONFICTION                0.102054
subject: LANGUAGEARTS&DISCIPLINES          0.00374817
subject: LAW                               0.105065
subject: LITERARYCRITICISM                 0.133757
subject: MEDICAL                           0.228431
subject: PERFORMINGARTS                    0.0454061
subject: PETS                              0.0738008
subject: PHILOSOPHY                        0.0326418
subject: POETRY                            0.0104252
subject: POLITICALSCIENCE                  0.0953728
subject: PSYCHOLOGY                        0.0414484
subject: SELF-HELP                         0.0452489
subject: SOCIALSCIENCE                     0.0730481
subject: SPORTS&RECREATION                 0.0819955
subject: STUDYAIDS                         0.0662289
subject: TECHNOLOGY&ENGINEERING            0.33147
subject: TRANSPORTATION                    0.337178
subject: TRUECRIME                         0.0425084
subject: YOUNGADULTFICTION                 0.0597694
subject: YOUNGADULTNONFICTION              0.0464062
-----------------------------------------  ----------
====================================================================================================
/home/htthong/document-understanding/DrDocBench/metrics/cdm_metric.py:123: FutureWarning: `random_state` is a deprecated argument name for `ransac`. It will be removed in version 0.23. Please use `rng` instead.
  model, inliers_1 = ransac(
/home/htthong/document-understanding/DrDocBench/metrics/cdm_metric.py:123: FutureWarning: `random_state` is a deprecated argument name for `ransac`. It will be removed in version 0.23. Please use `rng` instead.
  model, inliers_1 = ransac(
/home/htthong/document-understanding/DrDocBench/metrics/cdm_metric.py:123: FutureWarning: `random_state` is a deprecated argument name for `ransac`. It will be removed in version 0.23. Please use `rng` instead.
  model, inliers_1 = ransac(
/home/htthong/document-understanding/DrDocBench/metrics/cdm_metric.py:123: FutureWarning: `random_state` is a deprecated argument name for `ransac`. It will be removed in version 0.23. Please use `rng` instead.
  model, inliers_1 = ransac(
/home/htthong/document-understanding/.venv/lib/python3.10/site-packages/PIL/Image.py:1056: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
/home/htthong/document-understanding/.venv/lib/python3.10/site-packages/PIL/Image.py:1056: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
/home/htthong/document-understanding/.venv/lib/python3.10/site-packages/PIL/Image.py:1056: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
/home/htthong/document-understanding/.venv/lib/python3.10/site-packages/PIL/Image.py:1056: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
magick: profile 'icc': 'RGB ': RGB color space not permitted on grayscale PNG `result/my_model_run_display_formula/CDM/pred/vis/130_base.png' @ warning/png.c/MagickPNGWarningHandler/1525.
magick: profile 'icc': 'RGB ': RGB color space not permitted on grayscale PNG `result/my_model_run_display_formula/CDM/pred/vis/131_base.png' @ warning/png.c/MagickPNGWarningHandler/1525.
magick: profile 'icc': 'RGB ': RGB color space not permitted on grayscale PNG `result/my_model_run_display_formula/CDM/pred/vis/132_base.png' @ warning/png.c/MagickPNGWarningHandler/1525.
【display_formula】
Edit_dist:
---------------  ---------
ALL_page_avg     0.0489578
edit_whole       0.0170146
edit_sample_avg  0.0191201
---------------  ---------
====================================================================================================
CDM:
---  --------
all  0.989265
---  --------
====================================================================================================
----Anno Attribute---------------
CDM:
---------------------------  --------
formula_type: print          0.989036
text_background: white       1
text_language: text_english  1
text_rotate: normal          1
---------------------------  --------
====================================================================================================
Edit_dist:
---------------------------  ----------
formula_type: print          0.0194172
text_background: white       0.00520833
text_language: text_english  0.00520833
text_rotate: normal          0.00520833
---------------------------  ----------
====================================================================================================
sample_count:
---------------------------  ---
formula_type: print          281
text_background: white         6
text_language: text_english    6
text_rotate: normal            6
---------------------------  ---
====================================================================================================
CDM:
-----------------------------------------  --------
ALL                                        0.965992
challenge_type: domain_reasoning           1
challenge_type: structural_reconstruction  0.965992
data_source: book                          0.965992
english                                    0.964819
language: english                          1
layout: double_column                      0.872469
layout: single_column                      1
subject: COOKING                           1
subject: PHILOSOPHY                        1
subject: STUDYAIDS                         0.85425
subject: TECHNOLOGY&ENGINEERING            1
-----------------------------------------  --------
====================================================================================================
Edit_dist:
-----------------------------------------  ----------
ALL                                        0.0489578
challenge_type: domain_reasoning           0.00122249
challenge_type: structural_reconstruction  0.0488062
data_source: book                          0.0489578
english                                    0.0506461
language: english                          0
layout: double_column                      0.161508
layout: single_column                      0.0080304
subject: COOKING                           0
subject: PHILOSOPHY                        0.00209372
subject: STUDYAIDS                         0.184581
subject: TECHNOLOGY&ENGINEERING            0.0139671
-----------------------------------------  ----------
====================================================================================================
TEDS: 812it [04:39,  2.91it/s] 
【table】
TEDS:
---  --------
all  0.919951
---  --------
====================================================================================================
TEDS_structure_only:
---  --------
all  0.919951
---  --------
====================================================================================================
Edit_dist:
---------------  ---------
ALL_page_avg     0.0222446
edit_whole       0.0602821
edit_sample_avg  0.0800493
---------------  ---------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
----------------------------------------  ---------
include_background: false                 0.0278207
include_background: true                  0.284848
include_equation: false                   0.0800493
include_photo: false                      0.0800493
language: other                           0
language: table_en                        0.0802469
line: fewer_line                          0.011976
line: full_line                           0.108997
line: wireless_line                       0
table_layout: ['horizontal', 'vertical']  0
table_layout: ['horizontal']              0
table_layout: ['vertical', 'horizontal']  0
table_layout: ['vertical']                0.097561
table_layout: horizontal                  0.109195
table_layout: vertical                    0
with_span: false                          0.0401338
with_span: true                           0.103314
with_structured_text: false               0.112033
with_structured_text: true                0.0665499
----------------------------------------  ---------
====================================================================================================
TEDS:
----------------------------------------  --------
include_background: false                 0.972179
include_background: true                  0.715152
include_equation: false                   0.919951
include_photo: false                      0.919951
language: other                           1
language: table_en                        0.919753
line: fewer_line                          0.988024
line: full_line                           0.891003
line: wireless_line                       1
table_layout: ['horizontal', 'vertical']  1
table_layout: ['horizontal']              1
table_layout: ['vertical', 'horizontal']  1
table_layout: ['vertical']                0.902439
table_layout: horizontal                  0.890805
table_layout: vertical                    1
with_span: false                          0.959866
with_span: true                           0.896686
with_structured_text: false               0.887967
with_structured_text: true                0.93345
----------------------------------------  --------
====================================================================================================
TEDS_structure_only:
----------------------------------------  --------
include_background: false                 0.972179
include_background: true                  0.715152
include_equation: false                   0.919951
include_photo: false                      0.919951
language: other                           1
language: table_en                        0.919753
line: fewer_line                          0.988024
line: full_line                           0.891003
line: wireless_line                       1
table_layout: ['horizontal', 'vertical']  1
table_layout: ['horizontal']              1
table_layout: ['vertical', 'horizontal']  1
table_layout: ['vertical']                0.902439
table_layout: horizontal                  0.890805
table_layout: vertical                    1
with_span: false                          0.959866
with_span: true                           0.896686
with_structured_text: false               0.887967
with_structured_text: true                0.93345
----------------------------------------  --------
====================================================================================================
sample_count:
----------------------------------------  ---
include_background: false                 647
include_background: true                  165
include_equation: false                   812
include_photo: false                      812
language: other                             2
language: table_en                        810
line: fewer_line                          167
line: full_line                           578
line: wireless_line                        67
table_layout: ['horizontal', 'vertical']    5
table_layout: ['horizontal']              167
table_layout: ['vertical', 'horizontal']    5
table_layout: ['vertical']                 82
table_layout: horizontal                  522
table_layout: vertical                     31
with_span: false                          299
with_span: true                           513
with_structured_text: false               241
with_structured_text: true                571
----------------------------------------  ---
====================================================================================================
Edit_dist:
-----------------------------------------  -----------
ALL                                        0.0222446
challenge_type: structural_reconstruction  0.0222446
colorful_backgroud                         3.13321e-05
data_source: academic_literature           0
data_source: book                          0.0330424
data_source: exam_paper                    0
data_source: note                          0
english                                    0.0229275
french                                     0
fuzzy_scan                                 0
language: english                          0.0209148
layout: 1andmore_column                    0
layout: double_column                      0
layout: other_layout                       0.0282229
layout: single_column                      0.0400146
layout: three_column                       0
subject: BUSINESS&ECONOMICS                0
subject: COMPUTERS                         0
subject: COOKING                           0
subject: EDUCATION                         0
subject: GARDENING                         0
subject: HISTORY                           0
subject: HOUSE&HOME                        0
subject: JUVENILENONFICTION                0.0874618
subject: LANGUAGEARTS&DISCIPLINES          0
subject: POLITICALSCIENCE                  0
subject: SELF-HELP                         0
subject: SOCIALSCIENCE                     0
subject: SPORTS&RECREATION                 0
subject: STUDYAIDS                         0.0748379
subject: TRANSPORTATION                    0.071182
-----------------------------------------  -----------
====================================================================================================
TEDS:
-----------------------------------------  --------
ALL                                        0.9731
challenge_type: structural_reconstruction  0.9731
colorful_backgroud                         0.988336
data_source: academic_literature           1
data_source: book                          0.958718
data_source: exam_paper                    1
data_source: note                          1
english                                    0.972915
french                                     1
fuzzy_scan                                 1
language: english                          0.973913
layout: 1andmore_column                    1
layout: double_column                      1
layout: other_layout                       0.969697
layout: single_column                      0.954802
layout: three_column                       1
subject: BUSINESS&ECONOMICS                1
subject: COMPUTERS                         1
subject: COOKING                           1
subject: EDUCATION                         1
subject: GARDENING                         1
subject: HISTORY                           1
subject: HOUSE&HOME                        1
subject: JUVENILENONFICTION                0.890909
subject: LANGUAGEARTS&DISCIPLINES          1
subject: POLITICALSCIENCE                  1
subject: SELF-HELP                         1
subject: SOCIALSCIENCE                     1
subject: SPORTS&RECREATION                 1
subject: STUDYAIDS                         0.942857
subject: TRANSPORTATION                    0.896015
-----------------------------------------  --------
====================================================================================================
TEDS_structure_only:
-----------------------------------------  --------
ALL                                        0.9731
challenge_type: structural_reconstruction  0.9731
colorful_backgroud                         0.988336
data_source: academic_literature           1
data_source: book                          0.958718
data_source: exam_paper                    1
data_source: note                          1
english                                    0.972915
french                                     1
fuzzy_scan                                 1
language: english                          0.973913
layout: 1andmore_column                    1
layout: double_column                      1
layout: other_layout                       0.969697
layout: single_column                      0.954802
layout: three_column                       1
subject: BUSINESS&ECONOMICS                1
subject: COMPUTERS                         1
subject: COOKING                           1
subject: EDUCATION                         1
subject: GARDENING                         1
subject: HISTORY                           1
subject: HOUSE&HOME                        1
subject: JUVENILENONFICTION                0.890909
subject: LANGUAGEARTS&DISCIPLINES          1
subject: POLITICALSCIENCE                  1
subject: SELF-HELP                         1
subject: SOCIALSCIENCE                     1
subject: SPORTS&RECREATION                 1
subject: STUDYAIDS                         0.942857
subject: TRANSPORTATION                    0.896015
-----------------------------------------  --------
====================================================================================================
【reading_order】
Edit_dist:
---------------  ---------
ALL_page_avg     0.0391976
edit_whole       0.0308671
edit_sample_avg  0.0391976
---------------  ---------
====================================================================================================
----Anno Attribute---------------
sample_count:

====================================================================================================
Edit_dist:
-----------------------------------------  ----------
ALL                                        0.0391976
challenge_type:                            0
challenge_type: domain_reasoning           0.0123499
challenge_type: perception                 0.0284088
challenge_type: structural_reconstruction  0.0426178
colorful_backgroud                         0.0585113
data_source: academic_literature           0.0361346
data_source: book                          0.0379373
data_source: colorful_textbook             0.0533472
data_source: exam_paper                    0.0318182
data_source: magazine                      0.0192381
data_source: note                          0
english                                    0.053747
french                                     0.0597386
fuzzy_scan                                 0.0417021
language: english                          0.0260452
layout: 1andmore_column                    0.0507816
layout: double_column                      0.0371893
layout: other_layout                       0.0577922
layout: single_column                      0.0398266
layout: three_column                       0.0376526
other                                      0.0428135
subject: ARCHITECTURE                      0.0541681
subject: ART                               0
subject: BIOGRAPHY&AUTOBIOGRAPHY           0
subject: BODY,MIND&SPIRIT                  0
subject: BUSINESS&ECONOMICS                0
subject: COMICS&GRAPHICNOVELS              0.00378788
subject: COMPUTERS                         0.0413264
subject: COOKING                           0
subject: CRAFTS&HOBBIES                    0.017094
subject: DESIGN                            0.00392638
subject: EDUCATION                         0.00452137
subject: FAMILY&RELATIONSHIPS              0.0517513
subject: FICTION                           0
subject: GAMES&ACTIVITIES                  0.00605539
subject: GARDENING                         0.147124
subject: HISTORY                           0
subject: HOUSE&HOME                        0.00564794
subject: HUMOR                             0.0689311
subject: JUVENILENONFICTION                0.123308
subject: LANGUAGEARTS&DISCIPLINES          0.0318182
subject: LAW                               0.0432431
subject: LITERARYCRITICISM                 0
subject: MEDICAL                           0.190826
subject: PERFORMINGARTS                    0.0323709
subject: PETS                              0
subject: PHILOSOPHY                        0.00508342
subject: POETRY                            0.0140722
subject: POLITICALSCIENCE                  0.0406404
subject: PSYCHOLOGY                        0
subject: SELF-HELP                         0
subject: SOCIALSCIENCE                     0.011337
subject: SPORTS&RECREATION                 0.024494
subject: STUDYAIDS                         0.143809
subject: TECHNOLOGY&ENGINEERING            0.0756812
subject: TRANSPORTATION                    0.250574
subject: TRUECRIME                         0
subject: YOUNGADULTFICTION                 0.0264545
subject: YOUNGADULTNONFICTION              0.148253
-----------------------------------------  ----------
====================================================================================================
###### Task "end2end_eval" finished in 637.5s
###### Total time: 664.1s
(omnidocbench-eval-py3.10)