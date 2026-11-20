# Step 3: adaptation, forgetting, and return

**P1 FAIL; P2 FAIL. Overall FAIL.**

The primary comparison uses B learning rate **1e-05**, chosen on validation cubes only. Both rates are reported; the alternate rate cannot replace the selected comparison.

Scores are on fresh cubes 290–319, query shoves 5–9. Mean ± sample SD is across five paired seeds; lower is better. Coast means moving-coast steps. The full-state score uses all retained steps. Composite is the equal average of these three separately normalized errors.

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| A: fitted z, five shoves | 0.130257 ± 0.0143157 | 0.282832 ± 0.0405278 | 0.842992 ± 0.0517977 | 0.418694 ± 0.0111599 |
| B lr 1e-05, five shoves (selected) | 0.124249 ± 0.00584065 | 0.268554 ± 0.0295234 | 0.72209 ± 0.0388757 | 0.371631 ± 0.00271688 |
| B lr 1e-04, five shoves | 0.107807 ± 0.0161649 | 0.371622 ± 0.0428161 | 0.78442 ± 0.0809979 | 0.421283 ± 0.0206842 |
| History | 0.119718 ± 0.00888658 | 0.2579 ± 0.00862074 | 0.680691 ± 0.0482659 | 0.35277 ± 0.0163097 |
| Oracle | 0.0553283 ± 0.00885331 | 0.289564 ± 0.0313432 | 0.893159 ± 0.112161 | 0.412684 ± 0.0429087 |
| Plain | 0.127834 ± 0.00291914 | 0.252476 ± 0.0253551 | 0.744735 ± 0.0399906 | 0.375015 ± 0.00497735 |

Oracle receives true mass and combined friction. Its frozen Step 2 checkpoint was selected for push velocity change; its actual scores on coast and full state must be assessed separately rather than assumed to lead every component.

## Adaptation speed

P2: **FAIL**; fails to reach. Earliest qualifying A prefix: **none**.
Each prefix independently restarts from training-mean z or the frozen Plain checkpoint and takes exactly 50 updates. All prefixes use the same query shoves. A worse mean is not counted as a match merely because error bars overlap.

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| A, 1 support shoves | 0.130682 ± 0.0160698 | 0.286446 ± 0.0430849 | 0.912486 ± 0.0561481 | 0.443205 ± 0.0138151 |
| B lr 1e-05, 1 support shoves | 0.132345 ± 0.00378923 | 0.276884 ± 0.0240441 | 0.742312 ± 0.0385986 | 0.383847 ± 0.00422012 |
| B lr 1e-04, 1 support shoves | 0.144148 ± 0.00274981 | 0.39877 ± 0.0253509 | 0.74863 ± 0.0397302 | 0.430516 ± 0.014052 |
| A, 2 support shoves | 0.130436 ± 0.0155271 | 0.296363 ± 0.045218 | 0.807209 ± 0.0560199 | 0.411336 ± 0.0138697 |
| B lr 1e-05, 2 support shoves | 0.124252 ± 0.00512033 | 0.275074 ± 0.0332042 | 0.712129 ± 0.036305 | 0.370485 ± 0.00246992 |
| B lr 1e-04, 2 support shoves | 0.118563 ± 0.0108949 | 0.400518 ± 0.03925 | 0.714237 ± 0.0467435 | 0.411106 ± 0.00756014 |
| A, 3 support shoves | 0.131085 ± 0.0155368 | 0.291824 ± 0.0394145 | 0.821408 ± 0.0472038 | 0.414772 ± 0.00919851 |
| B lr 1e-05, 3 support shoves | 0.123953 ± 0.0050818 | 0.271336 ± 0.0307139 | 0.719198 ± 0.0401094 | 0.371496 ± 0.0029031 |
| B lr 1e-04, 3 support shoves | 0.114424 ± 0.0157887 | 0.385318 ± 0.0441965 | 0.799372 ± 0.085434 | 0.433038 ± 0.017838 |
| A, 4 support shoves | 0.130266 ± 0.0143922 | 0.284005 ± 0.0381385 | 0.841654 ± 0.0500341 | 0.418642 ± 0.0105463 |
| B lr 1e-05, 4 support shoves | 0.124144 ± 0.00529257 | 0.268765 ± 0.0301627 | 0.720126 ± 0.0389899 | 0.371012 ± 0.00231626 |
| B lr 1e-04, 4 support shoves | 0.108511 ± 0.0152683 | 0.371359 ± 0.0451609 | 0.780874 ± 0.0778114 | 0.420248 ± 0.0176951 |
| A, 5 support shoves | 0.130257 ± 0.0143157 | 0.282832 ± 0.0405278 | 0.842992 ± 0.0517977 | 0.418694 ± 0.0111599 |
| B lr 1e-05, 5 support shoves | 0.124249 ± 0.00584065 | 0.268554 ± 0.0295234 | 0.72209 ± 0.0388757 | 0.371631 ± 0.00271688 |
| B lr 1e-04, 5 support shoves | 0.107807 ± 0.0161649 | 0.371622 ± 0.0428161 | 0.78442 ± 0.0809979 | 0.421283 ± 0.0206842 |

![Adaptation speed](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/figures/adaptation_speed.png)

| B rate | Score | P2 | Earliest A prefix reaching B at five shoves |
|---|---|---|---:|
| 1e-05 | Push velocity NMSE | FAIL | none |
| 1e-05 | Moving-coast velocity NMSE | FAIL | none |
| 1e-05 | Full-state NMSE | FAIL | none |
| 1e-05 | Composite | FAIL | none |
| 1e-04 | Push velocity NMSE | FAIL | none |
| 1e-04 | Moving-coast velocity NMSE | PASS | 1 |
| 1e-04 | Full-state NMSE | FAIL | none |
| 1e-04 | Composite | PASS | 2 |

## Forgetting and return

The ascending sequence is 290,…,319. B retains all weights between cubes but starts a fresh Adam optimizer at each cube. A stores independent z vectors and reloads them without optimization. A unchanged return error is a structural consequence of frozen shared weights and preserved vectors, not learned resistance to interference.

Total visits 6,16,30 correspond to returns after 5,15,29 other cubes. P1 requires A non-rise for every seed and B rise beyond seed spread at all three returns: return mean−SD > initial mean+SD. The initial score is after adapting to the first cube.

| B rate | Score | Other cubes | A non-rise | B rise beyond spread | P1 at return | B paired change mean ± SD |
|---|---|---:|---|---|---|---:|
| 1e-05 | Push velocity NMSE | 5 | PASS | FAIL | FAIL | 0.0205153 ± 0.0117328 |
| 1e-05 | Push velocity NMSE | 15 | PASS | FAIL | FAIL | -0.00932423 ± 0.00650488 |
| 1e-05 | Push velocity NMSE | 29 | PASS | FAIL | FAIL | -0.00387623 ± 0.00612821 |
| 1e-05 | Moving-coast velocity NMSE | 5 | PASS | FAIL | FAIL | -0.0242606 ± 0.0134766 |
| 1e-05 | Moving-coast velocity NMSE | 15 | PASS | FAIL | FAIL | -0.0377991 ± 0.0317234 |
| 1e-05 | Moving-coast velocity NMSE | 29 | PASS | FAIL | FAIL | -0.00235532 ± 0.0339191 |
| 1e-05 | Full-state NMSE | 5 | PASS | FAIL | FAIL | 0.00228079 ± 0.00291758 |
| 1e-05 | Full-state NMSE | 15 | PASS | FAIL | FAIL | 0.00351533 ± 0.00485003 |
| 1e-05 | Full-state NMSE | 29 | PASS | FAIL | FAIL | 0.00595459 ± 0.00343284 |
| 1e-05 | Composite | 5 | PASS | FAIL | FAIL | -0.000488164 ± 0.00291346 |
| 1e-05 | Composite | 15 | PASS | FAIL | FAIL | -0.014536 ± 0.012108 |
| 1e-05 | Composite | 29 | PASS | FAIL | FAIL | -9.23207e-05 ± 0.0133664 |
| 1e-04 | Push velocity NMSE | 5 | PASS | FAIL | FAIL | 0.0731958 ± 0.0678262 |
| 1e-04 | Push velocity NMSE | 15 | PASS | FAIL | FAIL | 0.0100606 ± 0.0379217 |
| 1e-04 | Push velocity NMSE | 29 | PASS | FAIL | FAIL | 0.0328723 ± 0.0468733 |
| 1e-04 | Moving-coast velocity NMSE | 5 | PASS | FAIL | FAIL | 0.106027 ± 0.119017 |
| 1e-04 | Moving-coast velocity NMSE | 15 | PASS | FAIL | FAIL | 0.0721369 ± 0.192542 |
| 1e-04 | Moving-coast velocity NMSE | 29 | PASS | FAIL | FAIL | 0.108915 ± 0.169156 |
| 1e-04 | Full-state NMSE | 5 | PASS | FAIL | FAIL | 0.00643189 ± 0.0107349 |
| 1e-04 | Full-state NMSE | 15 | PASS | PASS | PASS | 0.0225454 ± 0.0138496 |
| 1e-04 | Full-state NMSE | 29 | PASS | PASS | PASS | 0.0611645 ± 0.0211694 |
| 1e-04 | Composite | 5 | PASS | FAIL | FAIL | 0.0618847 ± 0.0477682 |
| 1e-04 | Composite | 15 | PASS | FAIL | FAIL | 0.0349143 ± 0.0758909 |
| 1e-04 | Composite | 29 | PASS | FAIL | FAIL | 0.0676508 ± 0.0635521 |

For the selected B rate, the claimed rise beyond seed spread is absent after **5, 15, 29 other cubes**. A retained its first-cube prediction through unchanged frozen weights and stored z.

| B rate | Score | P1 across all three returns |
|---|---|---|
| 1e-05 | Push velocity NMSE | FAIL |
| 1e-05 | Moving-coast velocity NMSE | FAIL |
| 1e-05 | Full-state NMSE | FAIL |
| 1e-05 | Composite | FAIL |
| 1e-04 | Push velocity NMSE | FAIL |
| 1e-04 | Moving-coast velocity NMSE | FAIL |
| 1e-04 | Full-state NMSE | FAIL |
| 1e-04 | Composite | FAIL |

### First-cube query scores

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| History (constant reference) | 0.0777177 ± 0.0123705 | 0.169335 ± 0.0519329 | 0.0246554 ± 0.00157922 | 0.0905693 ± 0.0182532 |
| Oracle (constant reference) | 0.0192842 ± 0.00734668 | 0.203577 ± 0.0381455 | 0.0370847 ± 0.00727134 | 0.0866487 ± 0.011842 |
| Plain (constant reference) | 0.0951076 ± 0.0162277 | 0.187565 ± 0.0352291 | 0.0393225 ± 0.00362645 | 0.107332 ± 0.0143453 |

The reference models remain frozen and receive no across-cube adaptation.

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| A, total visits 1 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 1 | 0.0837865 ± 0.0162336 | 0.227469 ± 0.042387 | 0.0394802 ± 0.00450287 | 0.116912 ± 0.0178791 |
| B lr 1e-04, total visits 1 | 0.0722619 ± 0.0262278 | 0.260432 ± 0.0689962 | 0.0519469 ± 0.00781199 | 0.128214 ± 0.0321652 |
| A, total visits 5 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 5 | 0.107846 ± 0.02013 | 0.192644 ± 0.0237787 | 0.0413404 ± 0.004474 | 0.113944 ± 0.0115741 |
| B lr 1e-04, total visits 5 | 0.125288 ± 0.0231229 | 0.311146 ± 0.120619 | 0.072624 ± 0.0168927 | 0.169686 ± 0.0329243 |
| A, total visits 6 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 6 | 0.104302 ± 0.0225298 | 0.203208 ± 0.0509484 | 0.041761 ± 0.00704634 | 0.116424 ± 0.0197137 |
| B lr 1e-04, total visits 6 | 0.145458 ± 0.0523192 | 0.366459 ± 0.151163 | 0.0583788 ± 0.0123474 | 0.190098 ± 0.0565892 |
| A, total visits 10 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 10 | 0.102848 ± 0.0210904 | 0.215308 ± 0.0377784 | 0.0391081 ± 0.00417541 | 0.119088 ± 0.0151348 |
| B lr 1e-04, total visits 10 | 0.0858105 ± 0.0250949 | 0.372284 ± 0.0889297 | 0.0645901 ± 0.00439671 | 0.174228 ± 0.0359224 |
| A, total visits 15 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 15 | 0.0658922 ± 0.0143184 | 0.20927 ± 0.0656138 | 0.044822 ± 0.0080503 | 0.106661 ± 0.0251506 |
| B lr 1e-04, total visits 15 | 0.0463495 ± 0.0259647 | 0.419319 ± 0.40559 | 0.0965876 ± 0.0227199 | 0.187419 ± 0.142973 |
| A, total visits 16 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 16 | 0.0744622 ± 0.019352 | 0.18967 ± 0.0364028 | 0.0429956 ± 0.00767122 | 0.102376 ± 0.0166494 |
| B lr 1e-04, total visits 16 | 0.0823225 ± 0.0381598 | 0.332569 ± 0.182119 | 0.0744923 ± 0.0136161 | 0.163128 ± 0.0742846 |
| A, total visits 20 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 20 | 0.0791304 ± 0.0158425 | 0.192493 ± 0.0228619 | 0.0409867 ± 0.00743574 | 0.104203 ± 0.0140162 |
| B lr 1e-04, total visits 20 | 0.133457 ± 0.0469478 | 0.451128 ± 0.177133 | 0.0770415 ± 0.0147724 | 0.220542 ± 0.0727954 |
| A, total visits 25 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 25 | 0.0911233 ± 0.020517 | 0.189805 ± 0.0160945 | 0.0429403 ± 0.00558187 | 0.107956 ± 0.00910847 |
| B lr 1e-04, total visits 25 | 0.0769477 ± 0.0161326 | 0.256698 ± 0.0768748 | 0.0933963 ± 0.00772406 | 0.142347 ± 0.0216856 |
| A, total visits 30 | 0.0787124 ± 0.019963 | 0.153225 ± 0.0381647 | 0.0367609 ± 0.00531111 | 0.089566 ± 0.0156084 |
| B lr 1e-05, total visits 30 | 0.0799102 ± 0.015995 | 0.225113 ± 0.0214212 | 0.0454348 ± 0.00355526 | 0.11682 ± 0.00860365 |
| B lr 1e-04, total visits 30 | 0.105134 ± 0.0283235 | 0.369348 ± 0.131214 | 0.113111 ± 0.0164945 | 0.195864 ± 0.0412742 |

![First-cube query scores](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/figures/first_cube_return.png)

### Training-cube validation scores

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| History (constant reference) | 0.139482 ± 0.00348638 | 0.218656 ± 0.00399725 | 0.471025 ± 0.0537316 | 0.276388 ± 0.0186438 |
| Oracle (constant reference) | 0.0580336 ± 0.00948636 | 0.268197 ± 0.0131438 | 0.660493 ± 0.0751405 | 0.328908 ± 0.0242557 |
| Plain (constant reference) | 0.148314 ± 0.00375081 | 0.234536 ± 0.0174474 | 0.716043 ± 0.0599819 | 0.366298 ± 0.0161293 |

The reference models remain frozen and receive no across-cube adaptation.

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| A, total visits 0 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 0 | 0.148314 ± 0.00375081 | 0.234536 ± 0.0174474 | 0.716043 ± 0.0599819 | 0.366298 ± 0.0161293 |
| B lr 1e-04, total visits 0 | 0.148314 ± 0.00375081 | 0.234536 ± 0.0174474 | 0.716043 ± 0.0599819 | 0.366298 ± 0.0161293 |
| A, total visits 1 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 1 | 0.158716 ± 0.00385339 | 0.277771 ± 0.0188479 | 0.698257 ± 0.0531015 | 0.378248 ± 0.0137713 |
| B lr 1e-04, total visits 1 | 0.192262 ± 0.0123463 | 0.416933 ± 0.0353 | 0.724857 ± 0.0554953 | 0.444684 ± 0.0225435 |
| A, total visits 5 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 5 | 0.154225 ± 0.00532469 | 0.254977 ± 0.0192935 | 0.709964 ± 0.0612192 | 0.373055 ± 0.0143192 |
| B lr 1e-04, total visits 5 | 0.230472 ± 0.00918574 | 0.546551 ± 0.090419 | 0.892007 ± 0.0880299 | 0.556343 ± 0.0252733 |
| A, total visits 6 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 6 | 0.152517 ± 0.00337986 | 0.255519 ± 0.0204086 | 0.71946 ± 0.0668037 | 0.375832 ± 0.0196596 |
| B lr 1e-04, total visits 6 | 0.212357 ± 0.0189207 | 0.538016 ± 0.102455 | 0.921927 ± 0.103886 | 0.557433 ± 0.0451408 |
| A, total visits 10 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 10 | 0.168962 ± 0.0103353 | 0.307511 ± 0.0461036 | 0.704997 ± 0.0640895 | 0.393823 ± 0.0159492 |
| B lr 1e-04, total visits 10 | 0.271036 ± 0.0252059 | 0.764931 ± 0.2276 | 0.926313 ± 0.100574 | 0.654094 ± 0.0666502 |
| A, total visits 15 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 15 | 0.175543 ± 0.0114795 | 0.292978 ± 0.0300477 | 0.738851 ± 0.0719982 | 0.402458 ± 0.0151108 |
| B lr 1e-04, total visits 15 | 0.272942 ± 0.0570171 | 0.694215 ± 0.125577 | 1.08889 ± 0.0675151 | 0.685348 ± 0.0644735 |
| A, total visits 16 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 16 | 0.16695 ± 0.00630065 | 0.264582 ± 0.0257722 | 0.727599 ± 0.0700304 | 0.386377 ± 0.017749 |
| B lr 1e-04, total visits 16 | 0.236198 ± 0.022023 | 0.664423 ± 0.0980064 | 0.908481 ± 0.0580783 | 0.603034 ± 0.0292455 |
| A, total visits 20 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 20 | 0.161044 ± 0.0067902 | 0.259111 ± 0.0212717 | 0.747726 ± 0.069901 | 0.389294 ± 0.019992 |
| B lr 1e-04, total visits 20 | 0.25203 ± 0.0186578 | 0.724637 ± 0.1112 | 0.989486 ± 0.120004 | 0.655385 ± 0.0495711 |
| A, total visits 25 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 25 | 0.161716 ± 0.00486457 | 0.263934 ± 0.0155474 | 0.684993 ± 0.0544924 | 0.370215 ± 0.0189685 |
| B lr 1e-04, total visits 25 | 0.259494 ± 0.0329146 | 0.656268 ± 0.0944795 | 0.862771 ± 0.0471254 | 0.592844 ± 0.0326728 |
| A, total visits 30 | 0.157905 ± 0.00778319 | 0.260093 ± 0.00987807 | 0.908732 ± 0.0413001 | 0.442243 ± 0.0178083 |
| B lr 1e-05, total visits 30 | 0.163426 ± 0.00568333 | 0.275036 ± 0.0119481 | 0.696237 ± 0.055619 | 0.378233 ± 0.0174281 |
| B lr 1e-04, total visits 30 | 0.250497 ± 0.00972474 | 0.812094 ± 0.0906552 | 0.849306 ± 0.0835647 | 0.637299 ± 0.0538633 |

![Training-cube validation scores](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/figures/training_validation_forgetting.png)

## Additional saved one-step diagnostics

These five-support-shove query scores do not replace the three declared components. All-coast velocity includes the stationary tail and uses the moving-coast energy denominator. Raw velocity MSE has units (m/s)²; legacy full-state MSE precedes division by the full-state energy.

| Model | All-coast velocity NMSE | Raw push Δv MSE | Raw moving-coast Δv MSE | Legacy full-state MSE |
|---|---:|---:|---:|---:|
| A: fitted z | 0.263529 ± 0.0347695 | 0.000322436 ± 3.54367e-05 | 8.33017e-05 ± 1.19366e-05 | 0.136721 ± 0.00840083 |
| B lr 1e-05 | 0.246453 ± 0.0265934 | 0.000307564 ± 1.44578e-05 | 7.90964e-05 ± 8.69546e-06 | 0.117113 ± 0.00630508 |
| B lr 1e-04 | 0.347519 ± 0.0376903 | 0.000266862 ± 4.00144e-05 | 0.000109453 ± 1.26105e-05 | 0.127222 ± 0.0131367 |
| History | 0.2424 ± 0.00906845 | 0.000296348 ± 2.19977e-05 | 7.59587e-05 ± 2.53905e-06 | 0.110398 ± 0.00782803 |
| Oracle | 0.275019 ± 0.0265799 | 0.000136958 ± 2.19153e-05 | 8.52846e-05 ± 9.23143e-06 | 0.144857 ± 0.0181909 |
| Plain | 0.230466 ± 0.0229564 | 0.000316438 ± 7.22598e-06 | 7.43612e-05 ± 7.46776e-06 | 0.120785 ± 0.00648589 |

## Validation choices and data integrity

Selected A: dimension **4**, penalty **1**, learning rate **0.01**, **50 updates**. Selected B rate: **1e-05**, **50 updates**. The validation-only grid and its component scores are in [the selection report](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/VALIDATION_SELECTION.md).
The written protocol preceded all runs. Settings were frozen and announced before fresh properties, commands, or traces were generated. Original 200 cubes and all traces remain byte-identical; 35 Step 2 checkpoints remain frozen. IDs 200–289 are retired from Step 3.
Fresh collection: **300/300 launches**, **89.12%** moving coast after trimming, maximum coast distance **0.877059 m**, **18,646** retained and **76,213** trimmed transitions. No sample was rejected or replaced.
No phase, length, mask, ID, time, termination feature, or physical label enters A, B, or History. Adaptation uses only support shoves; query scores never update weights or z. True mass and combined friction enter Oracle only.
No world-model retraining or checkpoint selection occurred. Temporary B copies were adapted as specified. Every plotted point comes from saved run records; report generation performs no predictions.

## Saved evidence

- [Protocol and reasons](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/PROTOCOL.md)
- [All frozen comparisons and criterion arithmetic](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/summary.json)
- [Final evaluation ledger](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/final_evaluation.json)
- [Independent final audit](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/final_audit.json)
- [Collection audit](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/full_audit.json)
- [Frozen validation selection](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/selection_seal.json)
- [Announcement before test generation](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/settings_announced.json)
- [Preserved earlier evidence](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/retired_experiment_seal.json)
- [All prefix means and individual seeds](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/prefix_scores.csv)
- [All sequence and return scores and individual seeds](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/sequence_scores.csv)
- [Each currently visited cube in the continual sequence](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/current_cube_sequence_scores.csv)

Raw velocity MSE, the original full-state normalized MSE, and all-coast velocity scores are retained in the per-run JSON records under results/prefix, results/baselines, and results/sequence. Bands show optimization-seed spread, not confidence intervals over possible test cubes or visitation orders. P1 concerns the single designated first cube 290 in one fixed ascending order; it does not establish forgetting across all possible first cubes or orders.
