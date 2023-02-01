# Related Papers

1. **Isovector EMC effect from global QCD analysis with MARATHON data**   
   *C. Cocuzza, C. E. Keppel, H. Liu, W. Melnitchouk, A. Metz, N. Sato, A. W. Thomas*  
   ([PRL][marathon-PRL])([arXiv][marathon-arXiv])([inspire][marathon-inspire])
2. **Bayesian Monte Carlo extraction of sea asymmetry with SeaQuest and STAR data**   
   *C. Cocuzza, W. Melnitchouk, A. Metz, N. Sato*  
   ([PRD][seaquest-PRD])([arXiv][seaquest-arXiv])([inspire][seaquest-inspire])
2. **Polarized Antimatter in the Proton from Global QCD Analysis**   
   *C. Cocuzza, W. Melnitchouk, A. Metz, N. Sato*  
   ([PRD][star-PRD])([arXiv][star-arXiv])([inspire][star-inspire])

<br>

[marathon-PRL]: https://doi.org/10.1103/PhysRevLett.127.242001
[seaquest-PRD]: https://doi.org/10.1103/PhysRevD.104.074031
[star-PRD]:     https://doi.org/10.1103/PhysRevD.106.L031502

[marathon-arXiv]: https://arxiv.org/abs/2104.06946
[seaquest-arXiv]: https://arxiv.org/abs/2109.00677
[star-arXiv]:     https://arxiv.org/abs/2202.03372

[marathon-inspire]: https://inspirehep.net/literature/1858194
[seaquest-inspire]: https://inspirehep.net/literature/1915661
[star-inspire]:     https://inspirehep.net/literature/2029139

# LHAPDF

Will put LHAPDF references here


# Unpolarized Analysis:
Information:
* A = 2 Wavefunction: Paris (unless otherwise stated)
* A = 3 Wavefunction: KPSV  (unless otherwise stated)
* Multiplicative higher twist (if included)
* Initial parameterization: OS for uv, dv, g, ub, db, s, sb, sea1 = sea2
* Input scale: m<sub>c</sub><sup>2</sup>
* DIS Q<sup>2</sup> cut:  m<sub>c</sub><sup>2</sup>

| Step     | Information                               |
| ----     | -----------                               |
| 01       | Fixed Target DIS, W<sup>2</sup> > 10      |    
| 02       | + HERA                                    |
| 03       | + JLab DIS, A = 2 only, W<sup>2</sup> > 3 |
| 04       | + AOT target mass corrections             |
| 05       | + Higher twist corrections (p=n)          |
| 06       | + E866 Drell-Yan                          |
| 07       | + OS++ for uv, dv, g, ub, db              |
| 08       | + Z production (Tevatron)                 |
| 09       | sea1 (light quark) != sea2 (strange)      |
| 10       | + Higher twist corrections (p!=n)         |
| 11       | + Lepton (CMS, LHCb)                      |
| 12       | + W asymmetry (Tevatron)                  |
| 13       | + Jets                                    |
| 14       | + Lepton (RHIC)                           |
| 15       | + SeaQuest                                |
| 16       | + JLab Hall C h/d, Mix parameters         |
| 17       | + MARATHON D/p and h/t                    |
| 18       | + off-shell corrections                   |

In the end the analysis includes the following data:

| process | index | ref                     | experiment            | target(s) | obs                                  |  
| :--:    | :--:  | :--:                    | :--:                  | :--:      | :--:                                 |  
| idis    | 10010 | [link][idis10010-10015] | SLAC                  | p         | F2                                   |  
| idis    | 10011 | [link][idis10010-10015] | SLAC                  | d         | F2                                   |  
| idis    | 10016 | [link][idis10016]       | BCDMS                 | p         | F2                                   |  
| idis    | 10017 | [link][idis10017]       | BCDMS                 | d         | F2                                   |  
| idis    | 10020 | [link][idis10020]       | NMC                   | p         | F2                                   |  
| idis    | 10021 | [link][idis10021]       | NMC                   | d/p       | F2                                   |  
| idis    | 10026 | [link][idis10026]       | HERA II NC e+ (1)     | p         | sigma red                            |  
| idis    | 10027 | [link][idis10026]       | HERA II NC e+ (2)     | p         | sigma red                            |  
| idis    | 10028 | [link][idis10026]       | HERA II NC e+ (3)     | p         | sigma red                            |  
| idis    | 10029 | [link][idis10026]       | HERA II NC e+ (4)     | p         | sigma red                            |  
| idis    | 10030 | [link][idis10026]       | HERA II NC e-         | p         | sigma red                            |  
| idis    | 10031 | [link][idis10026]       | HERA II CC e+         | p         | sigma red                            |  
| idis    | 10032 | [link][idis10026]       | HERA II CC e-         | p         | sigma red                            |  
| idis    | 10002 | [link][idis10001-10004] | JLab Hall C (E00-106) | d         | F2                                   |  
| idis    | 10003 | [link][idis10001-10004] | JLab Hall C (E00-106) | p         | sigma red                            |  
| idis    | 10033 | [link][idis10033]       | BONUS                 | n/d       | F2                                   |  
| idis    | 10041 | [link][idis10041]       | JLab Hall C (E00-106) | h/d       | F2                                   |   
| idis    | 10050 | [link][idis10050-10051] | MARATHON              | d/p       | F2                                   |   
| idis    | 10051 | [link][idis10050-10051] | MARATHON              | h/t       | F2                                   |  
| dy      | 10001 | [link][dy10001]         | Fermilab E866         | pp        | M<sup>3</sup> dsig/dM dx<sub>F</sub> |
| dy      | 20001 | [link][dy20001]         | Fermilab E866         | pd/2pp    | M<sup>3</sup> dsig/dM dx<sub>F</sub> |
| dy      | 20002 | [link][dy20002]         | Fermilab E906         | pd/2pp    | M<sup>3</sup> dsig/dM dx<sub>F</sub> |
| zrap    | 1000  | [link][zrap1000]        | CDF                   | ppb       | sigma                                |
| zrap    | 1001  | [link][zrap1001]        | D0                    | ppb       | norm. sigma                          |
| wasym   | 1000  | [link][wasym1000]       | CDF                   | ppb       | asym                                 |
| wasym   | 1001  | [link][wasym1001]       | D0                    | ppb       | asym                                 |
| wzrv    | 2010  | [link][wzrv2010]        | CMS                   | pp        | asym                                 |      
| wzrv    | 2011  | [link][wzrv2011]        | CMS                   | pp        | asym                                 |      
| wzrv    | 2012  | [link][wzrv2012]        | CMS                   | pp        | asym                                 |      
| wzrv    | 2013  | [link][wzrv2013-2014]   | CMS                   | pp        | asym                                 |      
| wzrv    | 2014  | [link][wzrv2013-2014]   | CMS                   | pp        | asym                                 |      
| wzrv    | 2016  | [link][wzrv2016]        | LHCb                  | pp        | asym                                 |      
| wzrv    | 2017  | [link][wzrv2017]        | LHCb                  | pp        | asym                                 |      
| wzrv    | 2020  | [link][wzrv2020]        | STAR                  | pp        | RW                                   |      
| jet     | 10001 | [link][jet10001]        | D0                    | ppb       | sigma                                |
| jet     | 10002 | [link][jet10002]        | CDF                   | ppb       | sigma                                |
| jet     | 10003 | [link][jet10003]        | STAR                  | pp        | sigma                                |
| jet     | 10004 | [link][jet10004]        | STAR                  | pp        | sigma                                |


[idis10001-10004]: https://inspirehep.net/record/820503?ln=en
[idis10005-10009]: https://inspirehep.net/record/894309
[idis10010-10015]: https://inspirehep.net/literature/319089
[idis10016]:       https://inspirehep.net/record/276661?ln=en
[idis10017]:       https://inspirehep.net/record/285497?ln=en
[idis10020]:       http:s//inspirehep.net/record/424154?ln=en    
[idis10021]:       http:s//inspirehep.net/record/426595?ln=en
[idis10026]:       https://inspirehep.net/record/1377206?ln=en
[idis10033]:       https://inspirehep.net/record/1280957?ln=en
[idis10041]:       https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.202301
[idis10050-10051]: https://inspirehep.net/literature/1858035
[dy10001]:         https://inspirehep.net/record/554316
[dy20001]:         https://inspirehep.net/literature/554316
[dy20002]:         https://inspirehep.net/literature/1849683
[zrap1000]:        https://inspirehep.net/literature/856131
[zrap1001]:        https://inspirehep.net/literature/744624
[wasym1000]:       https://inspirehep.net/literature/811060
[wasym1001]:       https://inspirehep.net/literature/1268647
[wzrv2010]:        https://inspirehep.net/literature/1426517
[wzrv2011]:        https://inspirehep.net/literature/1273570
[wzrv2012]:        https://inspirehep.net/literature/1118047
[wzrv2013-2014]:   https://inspirehep.net/literature/892975
[wzrv2016]:        https://inspirehep.net/literature/1311488
[wzrv2017]:        https://inspirehep.net/literature/1406555
[wzrv2020]:        https://inspirehep.net/literature/1829350
[jet10001]:        https://doi.org/10.1103/PhysRevLett.101.062001
[jet10002]:        https://doi.org/10.1103/PhysRevD.75.092006
[jet10003]:        https://doi.org/10.1103/PhysRevLett.97.252001
[jet10004]:        https://doi.org/10.1103/PhysRevLett.97.252001


<br>

# Polarized Analysis:
Information:
* Continues off of Unpolarized Analysis (results/marathon/final)
* A = 2 Wavefunction: Paris (unless otherwise stated)
* A = 3 Wavefunction: KPSV  (unless otherwise stated)
* Initial PPDF parameterization: OS for uv, dv, g, ub = db = s = sb, sea1 = sea2
* Initial pion FF parameterization: OS for u = db, d = s = sb, c = cb, b = bb, g
* Initial kaon FF parameterization: OS for u, sb, d = s = ub = db, c = cb, b = bb, g
* g1 initially only includes twist-2 contributions
* g2 is initially set to zero
* DIS Q<sup>2</sup> cut:  m<sub>c</sub><sup>2</sup>

| Step     | Information                                                   |
| ----     | -----------                                                   |
| 20       | Fixed Target DIS (no Aperp or A2), W<sup>2</sup> > 10         |    
| 21       | + OS+ for all PPDFs                                           |
| 22       | Free ub, db, s, and sb PPDFs                                  |
| 23       | + SIA pions, 0.2 < z < 0.9                                    |
| 24       | + PSIDIS pions, 0.2 < z < 0.8, W<sup>2</sup> > 10             |
| 25       | Free s pion FF                                                |
| 26       | + Second shapes for u and db pion FFs                         |
| 27       | + SIA kaons, 0.2 < z < 0.9                                    |
| 28       | + PSIDIS kaons, 0.2 < z < 0.8, W<sup>2</sup> > 10             |
| 29       | Free s kaon FF                                                |
| 30       | + Second shapes for u and sb kaon FFs                         |
| 31       | + Polarized jets, p<sub>T</sub> > 8 GeV                       |
| 32       | + Unpolarized pion SIDIS (PPDFs fixed)                        |
| 33       | + Unpolarized kaon SIDIS (PPDFs fixed)                        |
| 34       | + Fit PPDFs, pion FFs, and kaon FFs                           |
| 35       | + SIA hadrons, 0.2 < z < 0.9                                  |
| 36       | + SIDIS hadrons, 0.2 < z < 0.8, W<sup>2</sup> > 10            |
| 37       | + PSIDIS hadrons, 0.2 < z < 0.8, W<sup>2</sup> > 10           |
| 38       | + RHIC polarized W/Z, fit PPDFs only                          |
| 39       | Fit PPDFs and all FFs simultaneously                          |
| 40       | + Aperp and A2 data                                           |
| ----     | -----------                                                   |
| 41       | Lower DIS cut (W<sup>2</sup> > 4)                             |
| 42       | Add g2 through WW relation                                    |
| 43       | Add AOT target mass corrections                               |
| 44       | Add twist-4 corrections to g1                                 |


In the end the analysis includes the following inclusive data:

| process | index | ref                     | experiment            | target(s) | obs                                  |  
| :--:    | :--:  | :--:                    | :--:                  | :--:      | :--:                                 |  
| pidis   | 10001 | [link][pidis10001]      | COMPASS               | d         | A1                                   |
| pidis   | 10002 | [link][pidis10002]      | COMPASS               | p         | A1                                   |
| pidis   | 10003 | [link][pidis10003]      | COMPASS               | p         | A1                                   |
| pidis   | 10004 | [link][pidis10004]      | EMC                   | p         | A1                                   |
| pidis   | 10005 | [link][pidis10005]      | HERMES                | n         | A1                                   |
| pidis   | 10006 | [link][pidis10006]      | HERMES                | d         | Apa                                  |
| pidis   | 10007 | [link][pidis10007]      | HERMES                | p         | Apa                                  |
| pidis   | 10008 | [link][pidis10008]      | HERMES                | p         | A2                                   |
| pidis   | 10010 | [link][pidis10010]      | JLabHA(E06014)        | h         | Apa                                  |
| pidis   | 10011 | [link][pidis10011]      | JLabHA(E06014)        | h         | Ape                                  |
| pidis   | 10014 | [link][pidis10014]      | JLabHA(E99117)        | h         | Apa                                  |
| pidis   | 10015 | [link][pidis10015]      | JLabHA(E99117)        | h         | Ape                                  |
| pidis   | 10016 | [link][pidis10016]      | JLabHB(EG1DVCS)       | d         | Apa                                  |
| pidis   | 10017 | [link][pidis10017]      | JLabHB(EG1DVCS)       | p         | Apa                                  |
| pidis   | 10018 | [link][pidis10018]      | SLAC(E142)            | h         | A1                                   |
| pidis   | 10019 | [link][pidis10019]      | SLAC(E142)            | h         | A2                                   |
| pidis   | 10020 | [link][pidis10020]      | SLAC(E143)            | d         | Ape                                  |
| pidis   | 10021 | [link][pidis10021]      | SLAC(E143)            | d         | Apa                                  |
| pidis   | 10022 | [link][pidis10022]      | SLAC(E143)            | p         | Apa                                  |
| pidis   | 10023 | [link][pidis10023]      | SLAC(E143)            | p         | Ape                                  |
| pidis   | 10024 | [link][pidis10024]      | SLAC(E154)            | h         | Ape                                  |
| pidis   | 10025 | [link][pidis10025]      | SLAC(E154)            | h         | Apa                                  |
| pidis   | 10026 | [link][pidis10026]      | SLAC(E155)            | d         | Ape                                  |
| pidis   | 10027 | [link][pidis10027]      | SLAC(E155)            | d         | Apa                                  |
| pidis   | 10028 | [link][pidis10028]      | SLAC(E155)            | p         | Ape                                  |
| pidis   | 10029 | [link][pidis10029]      | SLAC(E155)            | p         | Apa                                  |
| pidis   | 10030 | [link][pidis10030]      | SLAC(E155x)           | d         | Atpe                                 |
| pidis   | 10031 | [link][pidis10031]      | SLAC(E155x)           | p         | Atpe                                 |
| pidis   | 10032 | [link][pidis10032]      | SLACE80E130           | p         | Apa                                  |
| pidis   | 10033 | [link][pidis10033]      | SMC                   | d         | A1                                   |
| pidis   | 10034 | [link][pidis10034]      | SMC                   | d         | A1                                   |
| pidis   | 10035 | [link][pidis10035]      | SMC                   | p         | A1                                   |
| pidis   | 10036 | [link][pidis10036]      | SMC                   | p         | A1                                   |
| pidis   | 10039 | [link][pidis10039]      | JLabHB(EG1b)          | d         | Apa                                  |
| pidis   | 10040 | [link][pidis10040]      | JLabHB(EG1b)          | d         | Apa                                  |
| pidis   | 10043 | [link][pidis10043]      | JLabHB(EG1b)          | p         | Apa                                  |
| pidis   | 10044 | [link][pidis10044]      | JLabHB(EG1b)          | p         | Apa                                  |
| wzrv    | 1000  | [link][wzrv1000]        | STAR                  | pp        | SSA(W)                               | 
| wzrv    | 1020  | [link][wzrv1020]        | PHENIX                | pp        | SSA(W+Z)                             | 
| wzrv    | 1021  | [link][wzrv1021]        | PHENIX                | pp        | SSA(W+Z)                             | 
| pjet    | 20001 | [link][pjet20001]       | STAR                  | pp        | DSA                                  |
| pjet    | 20002 | [link][pjet20002]       | STAR                  | pp        | DSA                                  |
| pjet    | 20003 | [link][pjet20003]       | STAR                  | pp        | DSA                                  |
| pjet    | 20004 | [link][pjet20004]       | STAR                  | pp        | DSA                                  |
| pjet    | 20005 | [link][pjet20005]       | PHENIX                | pp        | DSA                                  |
| pjet    | 20006 | [link][pjet20006]       | STAR                  | pp        | DSA                                  |
| pjet    | 20007 | [link][pjet20007]       | STAR                  | pp        | DSA                                  |
| pjet    | 20008 | [link][pjet20008]       | STAR                  | pp        | DSA                                  |


[pidis10001]: https://inspirehep.net/literature/1501480
[pidis10002]: https://inspirehep.net/literature/843494
[pidis10003]: https://www.sciencedirect.com/science/article/pii/S037026931500920X
[pidis10004]: https://www.sciencedirect.com/science/article/abs/pii/0550321389900898
[pidis10005]: https://inspirehep.net/literature/440904
[pidis10006]: https://inspirehep.net/literature/726689
[pidis10007]: https://inspirehep.net/literature/726689
[pidis10008]: https://inspirehep.net/literature/1082840
[pidis10010]: https://inspirehep.net/literature/1299339
[pidis10011]: https://inspirehep.net/literature/1299339
[pidis10014]: https://inspirehep.net/literature/650244
[pidis10015]: https://inspirehep.net/literature/650244
[pidis10016]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.90.025212
[pidis10017]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.90.025212
[pidis10018]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.54.6620
[pidis10019]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.54.6620
[pidis10020]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.58.112003
[pidis10021]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.58.112003
[pidis10022]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.58.112003
[pidis10023]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.58.112003
[pidis10024]: https://inspirehep.net/files/5837438e365534e9d3fc7225f13260dd
[pidis10025]: https://inspirehep.net/files/5837438e365534e9d3fc7225f13260dd
[pidis10026]: https://inspirehep.net/literature/493768
[pidis10027]: https://www.sciencedirect.com/science/article/pii/S0370269399009405
[pidis10028]: https://inspirehep.net/literature/493768
[pidis10029]: https://inspirehep.net/literature/530798
[pidis10030]: https://inspirehep.net/literature/585675
[pidis10031]: https://inspirehep.net/literature/585675
[pidis10032]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.51.1135
[pidis10033]: https://inspirehep.net/literature/499139
[pidis10034]: https://inspirehep.net/literature/471981
[pidis10035]: https://inspirehep.net/literature/499139
[pidis10036]: https://inspirehep.net/literature/471981
[pidis10039]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.92.055201
[pidis10040]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.92.055201
[pidis10043]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.96.065208
[pidis10044]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.96.065208
[wzrv1000]:   https://inspirehep.net/record/1708793 
[wzrv1020]:   https://inspirehep.net/literature/1365091
[wzrv1021]:   https://inspirehep.net/literature/1667398
[pjet20001]:  https://doi.org/10.1103/PhysRevLett.97.252001 'DOI'
[pjet20002]:  https://doi.org/10.1103/PhysRevD.86.032006 'DOI'
[pjet20003]:  https://doi.org/10.1103/PhysRevD.86.032006 'DOI'
[pjet20004]:  https://doi.org/10.1103/PhysRevLett.115.092002 'DOI'
[pjet20005]:  https://doi.org/10.1103/PhysRevD.84.012006 'DOI'
[pjet20006]:  https://doi.org/10.1103/PhysRevD.100.052005 'DOI'
[pjet20007]:  https://doi.org/10.1103/PhysRevD.103.L091103 'DOI'
[pjet20008]: https://arxiv.org/abs/2110.11020 'arXiv'

As well as the following semi-inclusive data:

| process | index | ref                 | experiment  |  hadron | target | RS    | obs  |
| :--:    | :--:  | :--:                | :--:        |  :--:   | :--:   | :--:  | :--: |
| sia     | 1001  | [link][TASSO80]     | TASSO       |  pion   | -      | 12    |      |
| sia     | 1002  | [link][TASSO83]     | TASSO       |  pion   | -      | 14    |      |
| sia     | 1003  | [link][TASSO83]     | TASSO       |  pion   | -      | 22    |      |
| sia     | 1004  | [link][TASSO80]     | TASSO       |  pion   | -      | 30    |      |
| sia     | 1005  | [link][TASSO89]     | TASSO       |  pion   | -      | 34    |      |
| sia     | 1006  | [link][TASSO89]     | TASSO       |  pion   | -      | 44    |      |
| sia     | 1007  | [link][TPC84]       | TPC         |  pion   | -      | 29    |      |
| sia     | 1008  | [link][TPC88]       | TPC         |  pion   | -      | 29    |      |
| sia     | 1010  | [link][TPC86]       | TPC(c)      |  pion   | -      | 29    |      |
| sia     | 1011  | [link][TPC86]       | TPC(b)      |  pion   | -      | 29    |      |
| sia     | 1012  | [link][HRS87]       | HRS         |  pion   | -      | 29    |      |
| sia     | 1013  | [link][TOPAZ95]     | TOPAZ       |  pion   | -      | 58    |      |
| sia     | 1014  | [link][SLD04]       | SLD         |  pion   | -      | 91.28 |      |
| sia     | 1016  | [link][SLD04]       | SLD(c)      |  pion   | -      | 91.28 |      |
| sia     | 1017  | [link][SLD04]       | SLD(b)      |  pion   | -      | 91.28 |      |
| sia     | 1018  | [link][ALEPH95]     | ALEPH       |  pion   | -      | 91.2  |      |
| sia     | 1019  | [link][OPAL94]      | OPAL        |  pion   | -      | 91.2  |      |
| sia     | 1023  | [link][OPAL00]      | OPAL(c)     |  pion   | -      | 91.2  |      |
| sia     | 1024  | [link][OPAL00]      | OPAL(b)     |  pion   | -      | 91.2  |      |
| sia     | 1025  | [link][DELPHI98]    | DELPHI      |  pion   | -      | 91.2  |      |
| sia     | 1027  | [link][DELPHI98]    | DELPHI(b)   |  pion   | -      | 91.2  |      |
| sia     | 1028  | [link][BABAR13]     | BABAR       |  pion   | -      | 10.54 |      |
| sia     | 1029  | [link][BELLE13]     | BELLE       |  pion   | -      | 10.52 |      |
| sia     | 1030  | [link][ARGUS89]     | ARGUS       |  pion   | -      | 9.98  |      |
| sia     | 2001  | [link][TASSO80]     | TASSO       |  kaon   | -      | 12    |      |
| sia     | 2002  | [link][TASSO83]     | TASSO       |  kaon   | -      | 14    |      |
| sia     | 2003  | [link][TASSO83]     | TASSO       |  kaon   | -      | 22    |      |
| sia     | 2004  | [link][TASSO80]     | TASSO       |  kaon   | -      | 30    |      |
| sia     | 2005  | [link][TASSO89]     | TASSO       |  kaon   | -      | 34    |      |
| sia     | 2006  | [link][TASSO89]     | TASSO       |  kaon   | -      | 44    |      |
| sia     | 2007  | [link][TPC84]       | TPC84       |  kaon   | -      | 29    |      |
| sia     | 2008  | [link][TPC88]       | TPC88       |  kaon   | -      | 29    |      |
| sia     | 2012  | [link][HRS87]       | HRS         |  kaon   | -      | 29    |      |
| sia     | 2013  | [link][TOPAZ95]     | TOPAZ       |  kaon   | -      | 58    |      |
| sia     | 2014  | [link][SLD04]       | SLD         |  kaon   | -      | 91.28 |      |
| sia     | 2016  | [link][SLD04]       | SLD(c)      |  kaon   | -      | 91.28 |      |
| sia     | 2017  | [link][SLD04]       | SLD(b)      |  kaon   | -      | 91.28 |      |
| sia     | 2018  | [link][ALEPH95]     | ALEPH       |  kaon   | -      | 91.2  |      |
| sia     | 2019  | [link][OPAL94]      | OPAL        |  kaon   | -      | 91.2  |      |
| sia     | 2023  | [link][OPAL00]      | OPAL(c)     |  kaon   | -      | 91.2  |      |
| sia     | 2024  | [link][OPAL00]      | OPAL(b)     |  kaon   | -      | 91.2  |      |
| sia     | 2025  | [link][DELPHI98]    | DELPHI      |  kaon   | -      | 91.2  |      |
| sia     | 2027  | [link][DELPHI98]    | DELPHI(b)   |  kaon   | -      | 91.2  |      |
| sia     | 2028  | [link][BABAR13]     | BABAR       |  kaon   | -      | 10.54 |      |
| sia     | 2029  | [link][BELLE13]     | BELLE       |  kaon   | -      | 10.52 |      |
| sia     | 2030  | [link][ARGUS89]     | ARGUS       |  kaon   | -      | 9.98  |      |
| sia     | 2031  | [link][DELPHI98]    | DELPHI      |  kaon   | -      | 91.2  |      |
| sia     | 4000  |                     |  ALEPH      |  hadron | -      | 91.2  |      |
| sia     | 4001  |                     |  DELPHI     |  hadron | -      | 91.2  |      |
| sia     | 4002  |                     |  SLD        |  hadron | -      | 91.28 |      |
| sia     | 4003  |                     |  TASSO      |  hadron | -      | 12    |      |
| sia     | 4004  |                     |  TPC        |  hadron | -      | 29    |      |
| sia     | 4005  |                     |  OPAL(b)    |  hadron | -      | 91.2  |      |
| sia     | 4006  |                     |  OPAL(c)    |  hadron | -      | 91.2  |      |
| sia     | 4007  |                     |  OPAL       |  hadron | -      | 91.2  |      |
| sia     | 4008  |                     |  TASSO      |  hadron | -      | 14    |      |
| sia     | 4009  |                     |  TASSO      |  hadron | -      | 22    |      |
| sia     | 4010  |                     |  TASSO      |  hadron | -      | 30    |      |
| sia     | 4011  |                     |  TASSO      |  hadron | -      | 35    |      |
| sia     | 4012  |                     |  TASSO      |  hadron | -      | 44    |      |
| sia     | 4013  |                     |  DELPHI(b)  |  hadron | -      | 91.2  |      |
| sia     | 4014  |                     |  SLD(c)     |  hadron | -      | 91.28 |      |
| sia     | 4015  |                     |  SLD(b)     |  hadron | -      | 91.28 |      |
| psidis  | 20004 | [link][refHERMES]   | HERMES      |  pi+    | p      |       |A1    |
| psidis  | 20005 | [link][refHERMES]   | HERMES      |  pi-    | p      |       |A1    |
| psidis  | 20008 | [link][refHERMES]   | HERMES      |  pi+    | d      |       |A1    |
| psidis  | 20009 | [link][refHERMES]   | HERMES      |  pi-    | d      |       |A1    |
| psidis  | 20012 | [link][refHERMES]   | HERMES      |  K+     | d      |       |A1    |
| psidis  | 20013 | [link][refHERMES]   | HERMES      |  K-     | d      |       |A1    |
| psidis  | 20014 | [link][refHERMES]   | HERMES      |  Ksum   | d      |       |A1    |
| psidis  | 20017 | [link][refCOMPASSp] | COMPASS     |  pi+    | p      |       |A1    |
| psidis  | 20018 | [link][refCOMPASSp] | COMPASS     |  pi-    | p      |       |A1    |
| psidis  | 20019 | [link][refCOMPASSp] | COMPASS     |  K+     | p      |       |A1    |
| psidis  | 20020 | [link][refCOMPASSp] | COMPASS     |  K-     | p      |       |A1    |
| psidis  | 20021 | [link][refCOMPASSd] | COMPASS     |  pi+    | d      |       |A1    |
| psidis  | 20022 | [link][refCOMPASSd] | COMPASS     |  pi-    | d      |       |A1    |
| psidis  | 20025 | [link][refCOMPASSd] | COMPASS     |  K+     | d      |       |A1    |
| psidis  | 20026 | [link][refCOMPASSd] | COMPASS     |  K-     | d      |       |A1    |
| psidis  | 20000 | [link][refSMC]      | SMC         |  h+     | p      |       |A1    |
| psidis  | 20001 | [link][refSMC]      | SMC         |  h-     | p      |       |A1    |
| psidis  | 20002 | [link][refSMC]      | SMC         |  h+     | d      |       |A1    |
| psidis  | 20003 | [link][refSMC]      | SMC         |  h-     | d      |       |A1    |
| psidis  | 20006 | [link][refHERMES]   | HERMES      |  h+     | p      |       |A1    |
| psidis  | 20007 | [link][refHERMES]   | HERMES      |  h-     | p      |       |A1    |
| psidis  | 20010 | [link][refHERMES]   | HERMES      |  h+     | d      |       |A1    |
| psidis  | 20011 | [link][refHERMES]   | HERMES      |  h-     | d      |       |A1    |
| psidis  | 20015 | [link][refHERMESh]  | HERMES      |  h+     | h      |       |A1    |
| psidis  | 20016 | [link][refHERMESh]  | HERMES      |  h-     | h      |       |A1    |
| psidis  | 20023 | [link][refCOMPASSd] | COMPASS     |  h+     | d      |       |A1    |
| psidis  | 20024 | [link][refCOMPASSd] | COMPASS     |  h-     | d      |       |A1    |
| sidis   | 1005  | [link][refCOMPASS]  | COMPASS     |  pi+    | d      |       |mult  |
| sidis   | 1006  | [link][refCOMPASS]  | COMPASS     |  pi-    | d      |       |mult  |
| sidis   | 2005  | [link][refCOMPASSK] | COMPASS     |  K+     | d      |       |mult  |
| sidis   | 2006  | [link][refCOMPASSK] | COMPASS     |  K-     | d      |       |mult  |
| sidis   | 3000  | [link][refCOMPASS]  | COMPASS     |  h+     | d      |       |mult  |
| sidis   | 3001  | [link][refCOMPASS]  | COMPASS     |  h-     | d      |       |mult  |



[refHERMES]:   https://inspirehep.net/literature/654756
[refCOMPASSp]: https://inspirehep.net/literature/862410
[refCOMPASSd]: https://inspirehep.net/literature/820721
[refHERMESh]:  https://inspirehep.net/literature/502312
[refSMC]:      https://inspirehep.net/literature/451092
[refCOMPASS]:  https://inspirehep.net/literature/1444985
[refCOMPASSK]: https://inspirehep.net/literature/1483098


[TASSO80]:   https://doi.org/10.1016/0370-2693(81)90104-0
[TASSO83]:   https://inspirehep.net/literature/195333
[TASSO89]:   https://inspirehep.net/literature/267755
[TPC84]:     https://inspirehep.net/literature/195994
[TPC86]:     https://inspirehep.net/literature/241108
[TPC88]:     https://inspirehep.net/literature/262143
[HRS87]:     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.35.2639
[TOPAZ95]:   https://inspirehep.net/literature/381900
[SLD04]:     https://inspirehep.net/literature/630327
[ALEPH95]:   https://inspirehep.net/literature/382179
[OPAL94]:    https://inspirehep.net/literature/372772
[OPAL00]:    https://inspirehep.net/literature/513336
[DELPHI98]:  https://inspirehep.net/literature/473409
[ARGUS89]:   https://inspirehep.net/literature/276860
[BABAR13]:   https://inspirehep.net/literature/1238276
[BELLE13]:   https://inspirehep.net/literature/1216515













