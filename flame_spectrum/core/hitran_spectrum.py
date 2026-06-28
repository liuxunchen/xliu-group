# hitran_spectrum.py
import numpy as np
from scipy import special, constants
import math
import os
import matplotlib.pyplot as plt

class HitranSpectrum:
    """
    HITRAN 光谱仿真类（支持多分子混合）
    自动识别分子、同位素，读取配分函数，计算吸收截面与光谱
    """

    # ---------- 物理常数 (CGS) ----------
    T_ref = 296.0
    cBolts = constants.Boltzmann * 1e7          # erg/K
    cc = constants.speed_of_light * 1e2         # cm/s
    cNA = constants.N_A
    c2 = constants.physical_constants['second radiation constant'][0] * 100   # cm·K
    cP = constants.physical_constants['standard atmosphere'][0] * 10          # 1 atm in Ba (CGS)

    # 多普勒宽度系数 √(2 N_A k_B ln2) / c
    cGammaD = math.sqrt(2.0 * cBolts * cNA * math.log(2.0)) / cc

    # ---------- HITRAN 160 位字段定义 ----------
    HITRAN_FMT = {
        'M':         (1, 2, int),
        'I':         (3, 1, int),
        'nu':        (4, 12, float),
        'S':         (16, 10, float),
        'A':         (26, 10, float),
        'gamma_air': (36, 5, float),
        'gamma_self':(41, 5, float),
        'E':         (46, 10, float),
        'n_air':     (56, 4, float),
        'delta_air': (60, 8, float),
    }

    # ---------- 分子质量字典 (来自 HITRAN ISO 表) ----------
    
    # ISO字典定义 
    # 格式: (mol_id, iso_id): (global_id, formula, abundance, mass_g_per_mol, name)

    ISO = {
        (  1,  1 ):    [      1,  'H2(16O)',                 9.973173E-01,  1.801056E+01,  'H2O'         ], 
        (  1,  2 ):    [      2,  'H2(18O)',                 1.999827E-03,  2.001481E+01,  'H2O'         ], 
        (  1,  3 ):    [      3,  'H2(17O)',                 3.718841E-04,  1.901478E+01,  'H2O'         ], 
        (  1,  4 ):    [      4,  'HD(16O)',                 3.106928E-04,  1.901674E+01,  'H2O'         ], 
        (  1,  5 ):    [      5,  'HD(18O)',                 6.230031E-07,  2.102098E+01,  'H2O'         ], 
        (  1,  6 ):    [      6,  'HD(17O)',                 1.158526E-07,  2.002096E+01,  'H2O'         ], 
        (  1,  7 ):    [    129,  'D2(16O)',                 2.419741E-08,  2.002292E+01,  'H2O'         ], 
        (  2,  1 ):    [      7,  '(12C)(16O)2',             9.842043E-01,  4.398983E+01,  'CO2'         ], 
        (  2,  2 ):    [      8,  '(13C)(16O)2',             1.105736E-02,  4.499318E+01,  'CO2'         ], 
        (  2,  3 ):    [      9,  '(16O)(12C)(18O)',         3.947066E-03,  4.599408E+01,  'CO2'         ], 
        (  2,  4 ):    [     10,  '(16O)(12C)(17O)',         7.339890E-04,  4.499404E+01,  'CO2'         ], 
        (  2,  5 ):    [     11,  '(16O)(13C)(18O)',         4.434456E-05,  4.699743E+01,  'CO2'         ], 
        (  2,  6 ):    [     12,  '(16O)(13C)(17O)',         8.246233E-06,  4.599740E+01,  'CO2'         ], 
        (  2,  7 ):    [     13,  '(12C)(18O)2',             3.957340E-06,  4.799832E+01,  'CO2'         ], 
        (  2,  8 ):    [     14,  '(17O)(12C)(18O)',         1.471799E-06,  4.699829E+01,  'CO2'         ], 
        (  2,  9 ):    [    121,  '(12C)(17O)2',             1.368466E-07,  4.599826E+01,  'CO2'         ], 
        (  2, 10 ):    [     15,  '(13C)(18O)2',             4.446000E-08,  4.900167E+01,  'CO2'         ], 
        (  2, 11 ):    [    120,  '(18O)(13C)(17O)',         1.653540E-08,  4.800165E+01,  'CO2'         ], 
        (  2, 12 ):    [    122,  '(13C)(17O)2',             1.537446E-09,  4.700162E+01,  'CO2'         ], 
        (  3,  1 ):    [     16,  '(16O)3',                  9.929009E-01,  4.798474E+01,  'O3'          ], 
        (  3,  2 ):    [     17,  '(16O)(16O)(18O)',         3.981942E-03,  4.998899E+01,  'O3'          ], 
        (  3,  3 ):    [     18,  '(16O)(18O)(16O)',         1.990971E-03,  4.998899E+01,  'O3'          ], 
        (  3,  4 ):    [     19,  '(16O)(16O)(17O)',         7.404746E-04,  4.898896E+01,  'O3'          ], 
        (  3,  5 ):    [     20,  '(16O)(17O)(16O)',         3.702373E-04,  4.898896E+01,  'O3'          ], 
        (  4,  1 ):    [     21,  '(14N)2(16O)',             9.903328E-01,  4.400106E+01,  'N2O'         ], 
        (  4,  2 ):    [     22,  '(14N)(15N)(16O)',         3.640926E-03,  4.499810E+01,  'N2O'         ], 
        (  4,  3 ):    [     23,  '(15N)(14N)(16O)',         3.640926E-03,  4.499810E+01,  'N2O'         ], 
        (  4,  4 ):    [     24,  '(14N)2(18O)',             1.985822E-03,  4.600531E+01,  'N2O'         ], 
        (  4,  5 ):    [     25,  '(14N)2(17O)',             3.692797E-04,  4.500528E+01,  'N2O'         ], 
        (  5,  1 ):    [     26,  '(12C)(16O)',              9.865444E-01,  2.799491E+01,  'CO'          ], 
        (  5,  2 ):    [     27,  '(13C)(16O)',              1.108364E-02,  2.899827E+01,  'CO'          ], 
        (  5,  3 ):    [     28,  '(12C)(18O)',              1.978224E-03,  2.999916E+01,  'CO'          ], 
        (  5,  4 ):    [     29,  '(12C)(17O)',              3.678671E-04,  2.899913E+01,  'CO'          ], 
        (  5,  5 ):    [     30,  '(13C)(18O)',              2.222500E-05,  3.100252E+01,  'CO'          ], 
        (  5,  6 ):    [     31,  '(13C)(17O)',              4.132920E-06,  3.000249E+01,  'CO'          ], 
        (  6,  1 ):    [     32,  '(12C)H4',                 9.882741E-01,  1.603130E+01,  'CH4'         ], 
        (  6,  2 ):    [     33,  '(13C)H4',                 1.110308E-02,  1.703466E+01,  'CH4'         ], 
        (  6,  3 ):    [     34,  '(12C)H3D',                6.157511E-04,  1.703748E+01,  'CH4'         ], 
        (  6,  4 ):    [     35,  '(13C)H3D',                6.917852E-06,  1.804083E+01,  'CH4'         ], 
        (  7,  1 ):    [     36,  '(16O)2',                  9.952616E-01,  3.198983E+01,  'O2'          ], 
        (  7,  2 ):    [     37,  '(16O)(18O)',              3.991410E-03,  3.399408E+01,  'O2'          ], 
        (  7,  3 ):    [     38,  '(16O)(17O)',              7.422352E-04,  3.299404E+01,  'O2'          ], 
        (  8,  1 ):    [     39,  '(14N)(16O)',              9.939737E-01,  2.999799E+01,  'NO'          ], 
        (  8,  2 ):    [     40,  '(15N)(16O)',              3.654311E-03,  3.099502E+01,  'NO'          ], 
        (  8,  3 ):    [     41,  '(14N)(18O)',              1.993122E-03,  3.200223E+01,  'NO'          ], 
        (  9,  1 ):    [     42,  '(32S)(16O)2',             9.456777E-01,  6.396190E+01,  'SO2'         ], 
        (  9,  2 ):    [     43,  '(34S)(16O)2',             4.195028E-02,  6.595770E+01,  'SO2'         ], 
        (  9,  3 ):    [    137,  '(33S)(16O)2',             7.464462E-03,  6.496129E+01,  'SO2'         ], 
        (  9,  4 ):    [    138,  '(16O)(32S)(18O)',         3.792558E-03,  6.596615E+01,  'SO2'         ], 
        ( 10,  1 ):    [     44,  '(14N)(16O)2',             9.916160E-01,  4.599290E+01,  'NO2'         ], 
        ( 10,  2 ):    [    130,  '(15N)(16O)2',             3.645643E-03,  4.698994E+01,  'NO2'         ], 
        ( 11,  1 ):    [     45,  '(14N)H3',                 9.958716E-01,  1.702655E+01,  'NH3'         ], 
        ( 11,  2 ):    [     46,  '(15N)H3',                 3.661289E-03,  1.802358E+01,  'NH3'         ], 
        ( 12,  1 ):    [     47,  'H(14N)(16O)3',            9.891098E-01,  6.299564E+01,  'HNO3'        ], 
        ( 12,  2 ):    [    117,  'H(15N)(16O)3',            3.636429E-03,  6.399268E+01,  'HNO3'        ], 
        ( 13,  1 ):    [     48,  '(16O)H',                  9.974726E-01,  1.700274E+01,  'OH'          ], 
        ( 13,  2 ):    [     49,  '(18O)H',                  2.000138E-03,  1.900699E+01,  'OH'          ], 
        ( 13,  3 ):    [     50,  '(16O)D',                  1.553706E-04,  1.800891E+01,  'OH'          ], 
        ( 14,  1 ):    [     51,  'H(19F)',                  9.998443E-01,  2.000623E+01,  'HF'          ], 
        ( 14,  2 ):    [    110,  'D(19F)',                  1.557410E-04,  2.101240E+01,  'HF'          ], 
        ( 15,  1 ):    [     52,  'H(35Cl)',                 7.575870E-01,  3.597668E+01,  'HCl'         ], 
        ( 15,  2 ):    [     53,  'H(37Cl)',                 2.422573E-01,  3.797373E+01,  'HCl'         ], 
        ( 15,  3 ):    [    107,  'D(35Cl)',                 1.180050E-04,  3.698285E+01,  'HCl'         ], 
        ( 15,  4 ):    [    108,  'D(37Cl)',                 3.773502E-05,  3.897990E+01,  'HCl'         ], 
        ( 16,  1 ):    [     54,  'H(79Br)',                 5.067811E-01,  7.992616E+01,  'HBr'         ], 
        ( 16,  2 ):    [     55,  'H(81Br)',                 4.930632E-01,  8.192412E+01,  'HBr'         ], 
        ( 16,  3 ):    [    111,  'D(79Br)',                 7.893838E-05,  8.093234E+01,  'HBr'         ], 
        ( 16,  4 ):    [    112,  'D(81Br)',                 7.680162E-05,  8.293029E+01,  'HBr'         ], 
        ( 17,  1 ):    [     56,  'H(127I)',                 9.998443E-01,  1.279123E+02,  'HI'          ], 
        ( 17,  2 ):    [    113,  'D(127I)',                 1.557410E-04,  1.289185E+02,  'HI'          ], 
        ( 18,  1 ):    [     57,  '(35Cl)(16O)',             7.559077E-01,  5.096377E+01,  'ClO'         ], 
        ( 18,  2 ):    [     58,  '(37Cl)(16O)',             2.417203E-01,  5.296082E+01,  'ClO'         ], 
        ( 19,  1 ):    [     59,  '(16O)(12C)(32S)',         9.373947E-01,  5.996699E+01,  'OCS'         ], 
        ( 19,  2 ):    [     60,  '(16O)(12C)(34S)',         4.158284E-02,  6.196278E+01,  'OCS'         ], 
        ( 19,  3 ):    [     61,  '(16O)(13C)(32S)',         1.053146E-02,  6.097034E+01,  'OCS'         ], 
        ( 19,  4 ):    [     62,  '(16O)(12C)(33S)',         7.399083E-03,  6.096637E+01,  'OCS'         ], 
        ( 19,  5 ):    [     63,  '(18O)(12C)(32S)',         1.879670E-03,  6.197123E+01,  'OCS'         ], 
        ( 19,  6 ):    [    135,  '(16O)(13C)(34S)',         4.671757E-04,  6.296614E+01,  'OCS'         ], 
        ( 20,  1 ):    [     64,  'H2(12C)(16O)',            9.862371E-01,  3.001056E+01,  'H2CO'        ], 
        ( 20,  2 ):    [     65,  'H2(13C)(16O)',            1.108020E-02,  3.101392E+01,  'H2CO'        ], 
        ( 20,  3 ):    [     66,  'H2(12C)(18O)',            1.977609E-03,  3.201481E+01,  'H2CO'        ], 
        ( 21,  1 ):    [     67,  'H(16O)(35Cl)',            7.557900E-01,  5.197159E+01,  'HOCl'        ], 
        ( 21,  2 ):    [     68,  'H(16O)(37Cl)',            2.416826E-01,  5.396864E+01,  'HOCl'        ], 
        ( 22,  1 ):    [     69,  '(14N)2',                  9.926874E-01,  2.800615E+01,  'N2'          ], 
        ( 22,  2 ):    [    118,  '(14N)(15N)',              7.299165E-03,  2.900318E+01,  'N2'          ], 
        ( 23,  1 ):    [     70,  'H(12C)(14N)',             9.851143E-01,  2.701090E+01,  'HCN'         ], 
        ( 23,  2 ):    [     71,  'H(13C)(14N)',             1.106758E-02,  2.801425E+01,  'HCN'         ], 
        ( 23,  3 ):    [     72,  'H(12C)(15N)',             3.621740E-03,  2.800793E+01,  'HCN'         ], 
        ( 24,  1 ):    [     73,  '(12C)H3(35Cl)',           7.489369E-01,  4.999233E+01,  'CH3Cl'       ], 
        ( 24,  2 ):    [     74,  '(12C)H3(37Cl)',           2.394912E-01,  5.198938E+01,  'CH3Cl'       ], 
        ( 25,  1 ):    [     75,  'H2(16O)2',                9.949516E-01,  3.400548E+01,  'H2O2'        ], 
        ( 26,  1 ):    [     76,  '(12C)2H2',                9.775989E-01,  2.601565E+01,  'C2H2'        ], 
        ( 26,  2 ):    [     77,  '(12C)(13C)H2',            2.196629E-02,  2.701900E+01,  'C2H2'        ], 
        ( 26,  3 ):    [    105,  '(12C)2HD',                3.045499E-04,  2.702182E+01,  'C2H2'        ], 
        ( 27,  1 ):    [     78,  '(12C)2H6',                9.769900E-01,  3.004695E+01,  'C2H6'        ], 
        ( 27,  2 ):    [    106,  '(12C)H3(13C)H3',          2.195261E-02,  3.105031E+01,  'C2H6'        ], 
        ( 28,  1 ):    [     79,  '(31P)H3',                 9.995329E-01,  3.399724E+01,  'PH3'         ], 
        ( 29,  1 ):    [     80,  '(12C)(16O)(19F)2',        9.865444E-01,  6.599172E+01,  'COF2'        ], 
        ( 29,  2 ):    [    119,  '(13C)(16O)(19F)2',        1.108366E-02,  6.699508E+01,  'COF2'        ], 
        ( 30,  1 ):    [    126,  '(32S)(19F)6',             9.501800E-01,  1.459625E+02,  'SF6'         ], 
        ( 31,  1 ):    [     81,  'H2(32S)',                 9.498841E-01,  3.398772E+01,  'H2S'         ], 
        ( 31,  2 ):    [     82,  'H2(34S)',                 4.213687E-02,  3.598351E+01,  'H2S'         ], 
        ( 31,  3 ):    [     83,  'H2(33S)',                 7.497664E-03,  3.498710E+01,  'H2S'         ], 
        ( 32,  1 ):    [     84,  'H(12C)(16O)(16O)H',       9.838977E-01,  4.600548E+01,  'HCOOH'       ], 
        ( 33,  1 ):    [     85,  'H(16O)2',                 9.951066E-01,  3.299766E+01,  'HO2'         ], 
        ( 34,  1 ):    [     86,  '(16O)',                   9.976280E-01,  1.599492E+01,  'O'           ], 
        ( 35,  1 ):    [    127,  '(35Cl)(16O)(14N)(16O)2',  7.495702E-01,  9.695667E+01,  'ClONO2'      ], 
        ( 35,  2 ):    [    128,  '(37Cl)(16O)(14N)(16O)2',  2.396937E-01,  9.895372E+01,  'ClONO2'      ], 
        ( 36,  1 ):    [     87,  '(14N)(16O)+',             9.939737E-01,  2.999799E+01,  'NOp'         ], 
        ( 37,  1 ):    [     88,  'H(16O)(79Br)',            5.055790E-01,  9.592108E+01,  'HOBr'        ], 
        ( 37,  2 ):    [     89,  'H(16O)(81Br)',            4.918937E-01,  9.791903E+01,  'HOBr'        ], 
        ( 38,  1 ):    [     90,  '(12C)2H4',                9.772944E-01,  2.803130E+01,  'C2H4'        ], 
        ( 38,  2 ):    [     91,  '(12C)H2(13C)H2',          2.195946E-02,  2.903466E+01,  'C2H4'        ], 
        ( 39,  1 ):    [     92,  '(12C)H3(16O)H',           9.859299E-01,  3.202622E+01,  'CH3OH'       ], 
        ( 40,  1 ):    [     93,  '(12C)H3(79Br)',           5.009946E-01,  9.394181E+01,  'CH3Br'       ], 
        ( 40,  2 ):    [     94,  '(12C)H3(81Br)',           4.874334E-01,  9.593976E+01,  'CH3Br'       ], 
        ( 41,  1 ):    [     95,  '(12C)H3(12C)(14N)',       9.738662E-01,  4.102655E+01,  'CH3CN'       ], 
        ( 42,  1 ):    [     96,  '(12C)(19F)4',             9.888900E-01,  8.799362E+01,  'CF4'         ], 
        ( 43,  1 ):    [    116,  '(12C)4H2',                9.559980E-01,  5.001565E+01,  'C4H2'        ], 
        ( 44,  1 ):    [    109,  'H(12C)3(14N)',            9.633460E-01,  5.101090E+01,  'HC3N'        ], 
        ( 45,  1 ):    [    103,  'H2',                      9.996885E-01,  2.015650E+00,  'H2'          ], 
        ( 45,  2 ):    [    115,  'HD',                      3.114316E-04,  3.021825E+00,  'H2'          ], 
        ( 46,  1 ):    [     97,  '(12C)(32S)',              9.396236E-01,  4.397207E+01,  'CS'          ], 
        ( 46,  2 ):    [     98,  '(12C)(34S)',              4.168171E-02,  4.596787E+01,  'CS'          ], 
        ( 46,  3 ):    [     99,  '(13C)(32S)',              1.055650E-02,  4.497543E+01,  'CS'          ], 
        ( 46,  4 ):    [    100,  '(12C)(33S)',              7.416675E-03,  4.497146E+01,  'CS'          ], 
        ( 47,  1 ):    [    114,  '(32S)(16O)3',             9.434345E-01,  7.995682E+01,  'SO3'         ], 
        ( 48,  1 ):    [    123,  '(12C)2(14N)2',            9.707524E-01,  5.200615E+01,  'C2N2'        ], 
        ( 49,  1 ):    [    124,  '(12C)(16O)(35Cl)2',       5.663918E-01,  9.793262E+01,  'COCl2'       ], 
        ( 49,  2 ):    [    125,  '(12C)(16O)(35Cl)(37Cl)',  3.622350E-01,  9.992967E+01,  'COCl2'       ], 
        ( 50,  1 ):    [    146,  '(32S)(16O)',              9.479262E-01,  4.796699E+01,  'SO'          ], 
        ( 50,  2 ):    [    147,  '(34S)(16O)',              4.205002E-02,  4.996278E+01,  'SO'          ], 
        ( 50,  3 ):    [    148,  '(32S)(18O)',              1.900788E-03,  4.997123E+01,  'SO'          ], 
        ( 51,  1 ):    [    144,  '(12C)H3(19F)',            9.884280E-01,  3.402188E+01,  'CH3F'        ], 
        ( 52,  1 ):    [    139,  '(74Ge)H4',                3.651724E-01,  7.795248E+01,  'GeH4'        ], 
        ( 52,  2 ):    [    140,  '(72Ge)H4',                2.741292E-01,  7.595338E+01,  'GeH4'        ], 
        ( 52,  3 ):    [    141,  '(70Ge)H4',                2.050722E-01,  7.395555E+01,  'GeH4'        ], 
        ( 52,  4 ):    [    142,  '(73Ge)H4',                7.755167E-02,  7.695476E+01,  'GeH4'        ], 
        ( 52,  5 ):    [    143,  '(76Ge)H4',                7.755167E-02,  7.995270E+01,  'GeH4'        ], 
        ( 53,  1 ):    [    131,  '(12C)(32S)2',             8.928115E-01,  7.594414E+01,  'CS2'         ], 
        ( 53,  2 ):    [    132,  '(32S)(12C)(34S)',         7.921026E-02,  7.793994E+01,  'CS2'         ], 
        ( 53,  3 ):    [    133,  '(32S)(12C)(33S)',         1.409435E-02,  7.694353E+01,  'CS2'         ], 
        ( 53,  4 ):    [    134,  '(13C)(32S)2',             1.003057E-02,  7.694750E+01,  'CS2'         ], 
        ( 54,  1 ):    [    145,  '(12C)H3(127I)',           9.884280E-01,  1.419279E+02,  'CH3I'        ], 
        ( 55,  1 ):    [    136,  '(14N)(19F)3',             9.963370E-01,  7.099829E+01,  'NF3'         ], 
    }

    def __init__(self, q_folder=None):
        self.q_folder = q_folder
        self.molecules = {}          # name -> data dict
        self.molecule_order = []     # 保持顺序

    # ---------- 分子管理 ----------
    def add_molecule(self, par_file, concentration=1.0, name=None):
        """添加一个分子，concentration 为体积分数（摩尔分数）"""
        if self.q_folder is None:
            raise ValueError("请先设置配分函数文件夹")

        # 读取谱线数据
        db, info = self._read_par(par_file)
        if db is None or len(db) == 0:
            raise ValueError(f"文件 {par_file} 无有效数据")

        mol_id = info['molecule_id']
        iso_id = info['isotope_id']

        # 获取分子质量 (g/mol)
        mass_gmol = self._get_mass(mol_id, iso_id)
        if mass_gmol is None:
            raise ValueError(f"无法获取分子 {mol_id}-{iso_id} 的质量")

        # 分子名称
        mol_name = name or self._get_name(mol_id, iso_id)

        # 读取配分函数
        q_file = os.path.join(self.q_folder, f'q{mol_id}.txt')
        if not os.path.exists(q_file):
            q_file = os.path.join(self.q_folder, f'Q{mol_id}.txt')
        if not os.path.exists(q_file):
            raise FileNotFoundError(f"配分函数文件 {q_file} 不存在")
        q_data = self._read_q(q_file)

        # 存储
        self.molecules[mol_name] = {
            'db': db,
            'conc': concentration,
            'mass_gmol': mass_gmol,
            'q_data': q_data,
            'mol_id': mol_id,
            'iso_id': iso_id,
            'par_file': par_file,
        }
        self.molecule_order.append(mol_name)
        print(f"加载 {mol_name}: {len(db)} 条谱线, 浓度 {concentration}")

    # ---------- 光谱计算 ----------
    def cross_section(self, mol_name, T, p, wavenumber, wing=10.0):
        """
        计算单个分子的吸收截面 σ(ν) [cm²/molecule]
        参数:
            T: 温度 (K)
            p: 总压 (atm)
            wavenumber: 波数网格
            wing: 线翼截断倍数
        """
        mol = self.molecules[mol_name]
        db = mol['db']
        mass_gmol = mol['mass_gmol']
        Q_T = self._interp_q(mol['q_data'], T)
        Q_ref = self._interp_q(mol['q_data'], self.T_ref)

        # 单分子质量 (g)
        m_molecule = mass_gmol / self.cNA

        sigma_arr = np.zeros_like(wavenumber)

        for line in db:
            nu = line[0]
            # 线翼截断
            gamma_D = self.cGammaD * math.sqrt(T / mass_gmol) * nu   # HWHM
            gamma_p = line[3] * p * (self.T_ref / T) ** line[6]      # 忽略自加宽
            wing_width = wing * (gamma_D + gamma_p)
            if nu < wavenumber[0] - wing_width or nu > wavenumber[-1] + wing_width:
                continue

            # 线强温度修正
            S = line[1]
            E = line[4]
            ratio = (Q_ref / Q_T) * math.exp(-self.c2 * E * (1.0/T - 1.0/self.T_ref))
            stim = (1.0 - math.exp(-self.c2 * nu / T)) / (1.0 - math.exp(-self.c2 * nu / self.T_ref))
            intensity = S * ratio * stim

            # Voigt 线型
            profile = self._voigt(nu, wavenumber, line[3], line[6], p, T, mass_gmol)
            sigma_arr += intensity * profile

        return sigma_arr

    def coef_mixture(self, T, p, wavenumber, wing=10.0):
        """
        计算混合气体总吸收系数 k(ν) [cm⁻¹]
        k = Σ N_i * σ_i
        """
        total_k = np.zeros_like(wavenumber)
        individual_k = {}
        for name in self.molecule_order:
            mol = self.molecules[name]
            sigma = self.cross_section(name, T, p, wavenumber, wing)
            # 数密度 N_i = (p * conc_i * cP) / (kB * T)   [分子/cm³]
            N_i = p * mol['conc'] * self.cP / (self.cBolts * T)
            k_i = sigma * N_i
            individual_k[name] = k_i
            total_k += k_i
        return total_k, wavenumber, individual_k

    def OD_mixture(self, T, p, L, wavenumber=None, start=None, end=None,
                   resolution=0.01, wing=10.0):
        """计算混合气体光学深度、透射率和吸收率"""
        if wavenumber is None:
            if start is None or end is None:
                # 自动范围
                starts = [self.molecules[n]['db'][:,0].min() for n in self.molecule_order]
                ends   = [self.molecules[n]['db'][:,0].max() for n in self.molecule_order]
                start, end = min(starts), max(ends)
            wavenumber = np.arange(start, end, resolution)

        total_k, wavenumber, ind_k = self.coef_mixture(T, p, wavenumber, wing)
        OD = total_k * L
        Tr = np.exp(-OD)
        Ab = 1.0 - Tr
        return OD, Ab, Tr, wavenumber, total_k, ind_k

    # ---------- 内部工具 ----------
    def _read_par(self, filename):
        db = []
        mol_ids = set()
        iso_ids = set()
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) < 160:
                    continue
                try:
                    M = int(line[0:2])
                    I = int(line[2:3]) if line[2:3].strip() else 1
                    if I == 0: I = 1
                    nu = float(line[3:15])
                    S  = float(line[15:25])
                    # A 字段不使用
                    gamma_air = float(line[35:40])
                    gamma_self= float(line[40:45])
                    E  = float(line[45:55])
                    n_air = float(line[55:59])
                    delta  = float(line[59:67])
                    mol_ids.add(M)
                    iso_ids.add(I)
                    db.append([nu, S, 0.0, gamma_air, gamma_self, E, n_air, delta])
                except (ValueError, IndexError):
                    continue
        if not db:
            return None, None
        main_mol = max(mol_ids, key=lambda x: sum(1 for d in db if d[0]==x)) if mol_ids else None
        main_iso = max(iso_ids, key=lambda x: sum(1 for d in db if d[1]==x)) if iso_ids else 1
        info = {'molecule_id': main_mol, 'isotope_id': main_iso}
        return np.array(db), info

    def _read_q(self, filename):
        data = np.loadtxt(filename)
        return {'T': data[:, 0], 'Q': data[:, 1]}

    def _interp_q(self, q_data, T):
        return np.interp(T, q_data['T'], q_data['Q'])

    def _get_mass(self, mol_id, iso_id):
        key = (mol_id, iso_id)
        if key in self.ISO:
            return self.ISO[key][3]   # g/mol
        # 回退到主要同位素
        for (mid, iid), val in self.ISO.items():
            if mid == mol_id and iid == 1:
                print(f"警告: 未找到 ({mol_id},{iso_id})，使用 ({mol_id},1) 质量")
                return val[3]
        return None

    def _get_name(self, mol_id, iso_id):
        key = (mol_id, iso_id)
        if key in self.ISO:
            return self.ISO[key][4]
        return f"Mol_{mol_id}"

    def _voigt(self, nu0, wn_grid, gamma_air, n_air, p, T, mass_gmol):
        """计算归一化 Voigt 线型（面积=1）"""
        gamma_p = gamma_air * p * (self.T_ref / T) ** n_air   # 洛伦兹半宽
        # 多普勒标准差 σ = sqrt(kT/m) * ν0/c
        # 使用单分子质量 m = mass_gmol / NA  (g)
        m = mass_gmol / self.cNA
        sigma = (nu0 / self.cc) * math.sqrt(self.cBolts * T / m)   # CGS 单位一致
        z = (wn_grid - nu0 + 1j * gamma_p) / (sigma * math.sqrt(2.0))
        return np.real(special.wofz(z)) / (sigma * math.sqrt(2.0 * math.pi))

    # ---------- 信息输出 ----------
    def info(self):
        lines = []
        for name in self.molecule_order:
            mol = self.molecules[name]
            lines.append(f"{name}: ID={mol['mol_id']}, 浓度={mol['conc']}, 谱线={len(mol['db'])}, "
                         f"质量={mol['mass_gmol']:.2f} g/mol")
        return "\n".join(lines)

            
# ---------- 示例 ----------
if __name__ == "__main__":
    # 初始化，指定 Q 文件夹路径
    hitran = HitranSpectrum(q_folder='../hitran_database/Q')
    hitran.add_molecule('CO_1416.par', concentration=0.01, name='CO')
    hitran.add_molecule('HITRAN_2073-2074.par', concentration=0.02, name='all')

    T, p, L = 600.0, 1.0, 10.0
    OD, Ab, Tr, wn, total_k, ind_k = hitran.OD_mixture(
        T, p, L, start=2000, end=2200, resolution=0.01, wing=10
    )

    plt.plot(wn, Tr)
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Transmittance')
    plt.show()
