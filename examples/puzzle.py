"""Bitcoin Puzzle Position Pattern Discovery - Find the Generation Formula.

This script analyzes ALL solved Bitcoin puzzles to discover the mathematical
pattern that determines WHERE in each bit-range the private key falls.

Goal: Predict position_ratio for puzzle 135 (and other unsolved puzzles)
  position_ratio = (actual_key - range_start) / (range_end - range_start)

Mathematical Methods Combined:
1. Fractal/Self-Similar Position Analysis - Recursive subdivision patterns
2. Kolmogorov Complexity Minimization - Find simplest generating formula
3. Hidden Markov Model State Inference - Detect invisible state machine
4. Wavelet Phase Space Reconstruction - Frequency domain patterns
5. Topological Data Analysis - Shape of position manifold

FunSearch Mission: Discover which mathematical transformation predicts
  the position pattern across all puzzles.

Guidelines for the LLM:
- **DISCOVER THE FORMULA**: The priority function should return a predicted
  position_ratio (0.0 to 1.0) for puzzle N based on features
- **EXPLORE EVERYTHING**: Try polynomial fits, recursive formulas, modular
  arithmetic, fractal recursions, state machines, anything!
- **BE CREATIVE**: The actual generation method might be something no one
  has thought of - bugs, timestamps, hash outputs, or pure chaos
- Always return a single, complete Python module starting at column 0
- Only adjust the body of ``priority``; keep function signatures unchanged
- Use **two spaces** per indentation level
- The priority function receives 200+ features about puzzle patterns
"""

import math
import numpy as np
import types

# ==============================================================================
# COMPLETE DATASET OF SOLVED BITCOIN PUZZLES
# ==============================================================================

# Solved puzzle metadata sourced from bitcoin-puzzle-solved-20251216.csv and inlined to avoid
# external file dependencies inside the FunSearch environment.
RAW_PUZZLE_DATA = [{'bits': 1,
  'range_min': 1,
  'range_max': 1,
  'address': '1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH',
  'hash160_compressed': '751e76e8199196d454941c45d1b3a323f1433bd6',
  'public_key': '0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798',
  'private_key': 1},
 {'bits': 2,
  'range_min': 2,
  'range_max': 3,
  'address': '1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb',
  'hash160_compressed': '7dd65592d0ab2fe0d0257d571abf032cd9db93dc',
  'public_key': '02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9',
  'private_key': 3},
 {'bits': 3,
  'range_min': 4,
  'range_max': 7,
  'address': '19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA',
  'hash160_compressed': '5dedfbf9ea599dd4e3ca6a80b333c472fd0b3f69',
  'public_key': '025cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc',
  'private_key': 7},
 {'bits': 4,
  'range_min': 8,
  'range_max': 15,
  'address': '1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e',
  'hash160_compressed': '9652d86bedf43ad264362e6e6eba6eb764508127',
  'public_key': '022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01',
  'private_key': 8},
 {'bits': 5,
  'range_min': 16,
  'range_max': 31,
  'address': '1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k',
  'hash160_compressed': '8f9dff39a81ee4abcbad2ad8bafff090415a2be8',
  'public_key': '02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5',
  'private_key': 21},
 {'bits': 6,
  'range_min': 32,
  'range_max': 63,
  'address': '1PitScNLyp2HCygzadCh7FveTnfmpPbfp8',
  'hash160_compressed': 'f93ec34e9e34a8f8ff7d600cdad83047b1bcb45c',
  'public_key': '03f2dac991cc4ce4b9ea44887e5c7c0bce58c80074ab9d4dbaeb28531b7739f530',
  'private_key': 49},
 {'bits': 7,
  'range_min': 64,
  'range_max': 127,
  'address': '1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC',
  'hash160_compressed': 'e2192e8a7dd8dd1c88321959b477968b941aa973',
  'public_key': '0296516a8f65774275278d0d7420a88df0ac44bd64c7bae07c3fe397c5b3300b23',
  'private_key': 76},
 {'bits': 8,
  'range_min': 128,
  'range_max': 255,
  'address': '1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK',
  'hash160_compressed': 'dce76b2613052ea012204404a97b3c25eac31715',
  'public_key': '0308bc89c2f919ed158885c35600844d49890905c79b357322609c45706ce6b514',
  'private_key': 224},
 {'bits': 9,
  'range_min': 256,
  'range_max': 511,
  'address': '1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV',
  'hash160_compressed': '7d0f6c64afb419bbd7e971e943d7404b0e0daab4',
  'public_key': '0243601d61c836387485e9514ab5c8924dd2cfd466af34ac95002727e1659d60f7',
  'private_key': 467},
 {'bits': 10,
  'range_min': 512,
  'range_max': 1023,
  'address': '1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe',
  'hash160_compressed': 'd7729816650e581d7462d52ad6f732da0e2ec93b',
  'public_key': '03a7a4c30291ac1db24b4ab00c442aa832f7794b5a0959bec6e8d7fee802289dcd',
  'private_key': 514},
 {'bits': 11,
  'range_min': 1024,
  'range_max': 2047,
  'address': '1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu',
  'hash160_compressed': 'f8c698da3164ef8fa4258692d118cc9a902c5acc',
  'public_key': '038b05b0603abd75b0c57489e451f811e1afe54a8715045cdf4888333f3ebc6e8b',
  'private_key': 1155},
 {'bits': 12,
  'range_min': 2048,
  'range_max': 4095,
  'address': '1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot',
  'hash160_compressed': '85a1f9ba4da24c24e582d9b891dacbd1b043f971',
  'public_key': '038b00fcbfc1a203f44bf123fc7f4c91c10a85c8eae9187f9d22242b4600ce781c',
  'private_key': 2683},
 {'bits': 13,
  'range_min': 4096,
  'range_max': 8191,
  'address': '1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1',
  'hash160_compressed': 'f932d0188616c964416b91fb9cf76ba9790a921e',
  'public_key': '03aadaaab1db8d5d450b511789c37e7cfeb0eb8b3e61a57a34166c5edc9a4b869d',
  'private_key': 5216},
 {'bits': 14,
  'range_min': 8192,
  'range_max': 16383,
  'address': '1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk',
  'hash160_compressed': '97f9281a1383879d72ac52a6a3e9e8b9a4a4f655',
  'public_key': '03b4f1de58b8b41afe9fd4e5ffbdafaeab86c5db4769c15d6e6011ae7351e54759',
  'private_key': 10544},
 {'bits': 15,
  'range_min': 16384,
  'range_max': 32767,
  'address': '1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW',
  'hash160_compressed': 'fe7c45126731f7384640b0b0045fd40bac72e2a2',
  'public_key': '02fea58ffcf49566f6e9e9350cf5bca2861312f422966e8db16094beb14dc3df2c',
  'private_key': 26867},
 {'bits': 16,
  'range_min': 32768,
  'range_max': 65535,
  'address': '1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY',
  'hash160_compressed': '7025b4efb3ff42eb4d6d71fab6b53b4f4967e3dd',
  'public_key': '029d8c5d35231d75eb87fd2c5f05f65281ed9573dc41853288c62ee94eb2590b7a',
  'private_key': 51510},
 {'bits': 17,
  'range_min': 65536,
  'range_max': 131071,
  'address': '1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm',
  'hash160_compressed': 'b67cb6edeabc0c8b927c9ea327628e7aa63e2d52',
  'public_key': '033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3',
  'private_key': 95823},
 {'bits': 18,
  'range_min': 131072,
  'range_max': 262143,
  'address': '1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE',
  'hash160_compressed': 'ad1e852b08eba53df306ec9daa8c643426953f94',
  'public_key': '020ce4a3291b19d2e1a7bf73ee87d30a6bdbc72b20771e7dfff40d0db755cd4af1',
  'private_key': 198669},
 {'bits': 19,
  'range_min': 262144,
  'range_max': 524287,
  'address': '1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w',
  'hash160_compressed': 'ebfbe6819fcdebab061732ce91df7d586a037dee',
  'public_key': '0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19',
  'private_key': 357535},
 {'bits': 20,
  'range_min': 524288,
  'range_max': 1048575,
  'address': '1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum',
  'hash160_compressed': 'b907c3a2a3b27789dfb509b730dd47703c272868',
  'public_key': '033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c',
  'private_key': 863317},
 {'bits': 21,
  'range_min': 1048576,
  'range_max': 2097151,
  'address': '14oFNXucftsHiUMY8uctg6N487riuyXs4h',
  'hash160_compressed': '29a78213caa9eea824acf08022ab9dfc83414f56',
  'public_key': '031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e',
  'private_key': 1811764},
 {'bits': 22,
  'range_min': 2097152,
  'range_max': 4194303,
  'address': '1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv',
  'hash160_compressed': '7ff45303774ef7a52fffd8011981034b258cb86b',
  'public_key': '023ed96b524db5ff4fe007ce730366052b7c511dc566227d929070b9ce917abb43',
  'private_key': 3007503},
 {'bits': 23,
  'range_min': 4194304,
  'range_max': 8388607,
  'address': '1L2GM8eE7mJWLdo3HZS6su1832NX2txaac',
  'hash160_compressed': 'd0a79df189fe1ad5c306cc70497b358415da579e',
  'public_key': '03f82710361b8b81bdedb16994f30c80db522450a93e8e87eeb07f7903cf28d04b',
  'private_key': 5598802},
 {'bits': 24,
  'range_min': 8388608,
  'range_max': 16777215,
  'address': '1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7',
  'hash160_compressed': '0959e80121f36aea13b3bad361c15dac26189e2f',
  'public_key': '036ea839d22847ee1dce3bfc5b11f6cf785b0682db58c35b63d1342eb221c3490c',
  'private_key': 14428676},
 {'bits': 25,
  'range_min': 16777216,
  'range_max': 33554431,
  'address': '15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP',
  'hash160_compressed': '2f396b29b27324300d0c59b17c3abc1835bd3dbb',
  'public_key': '03057fbea3a2623382628dde556b2a0698e32428d3cd225f3bd034dca82dd7455a',
  'private_key': 33185509},
 {'bits': 26,
  'range_min': 33554432,
  'range_max': 67108863,
  'address': '1JVnST957hGztonaWK6FougdtjxzHzRMMg',
  'hash160_compressed': 'bfebb73562d4541b32a02ba664d140b5a574792f',
  'public_key': '024e4f50a2a3eccdb368988ae37cd4b611697b26b29696e42e06d71368b4f3840f',
  'private_key': 54538862},
 {'bits': 27,
  'range_min': 67108864,
  'range_max': 134217727,
  'address': '128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k',
  'hash160_compressed': '0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560',
  'public_key': '031a864bae3922f351f1b57cfdd827c25b7e093cb9c88a72c1cd893d9f90f44ece',
  'private_key': 111949941},
 {'bits': 28,
  'range_min': 134217728,
  'range_max': 268435455,
  'address': '12jbtzBb54r97TCwW3G1gCFoumpckRAPdY',
  'hash160_compressed': '1306b9e4ff56513a476841bac7ba48d69516b1da',
  'public_key': '03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7',
  'private_key': 227634408},
 {'bits': 29,
  'range_min': 268435456,
  'range_max': 536870911,
  'address': '19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT',
  'hash160_compressed': '5a416cc9148f4a377b672c8ae5d3287adaafadec',
  'public_key': '026caad634382d34691e3bef43ed4a124d8909a8a3362f91f1d20abaaf7e917b36',
  'private_key': 400708894},
 {'bits': 30,
  'range_min': 536870912,
  'range_max': 1073741823,
  'address': '1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps',
  'hash160_compressed': 'd39c4704664e1deb76c9331e637564c257d68a08',
  'public_key': '030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b',
  'private_key': 1033162084},
 {'bits': 31,
  'range_min': 1073741824,
  'range_max': 2147483647,
  'address': '1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE',
  'hash160_compressed': 'd805f6f251f7479ebd853b3d0f4b9b2656d92f1d',
  'public_key': '0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28',
  'private_key': 2102388551},
 {'bits': 32,
  'range_min': 2147483648,
  'range_max': 4294967295,
  'address': '1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR',
  'hash160_compressed': '9e42601eeaedc244e15f17375adb0e2cd08efdc9',
  'public_key': '0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69',
  'private_key': 3093472814},
 {'bits': 33,
  'range_min': 4294967296,
  'range_max': 8589934591,
  'address': '187swFMjz1G54ycVU56B7jZFHFTNVQFDiu',
  'hash160_compressed': '4e15e5189752d1eaf444dfd6bff399feb0443977',
  'public_key': '03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579',
  'private_key': 7137437912},
 {'bits': 34,
  'range_min': 8589934592,
  'range_max': 17179869183,
  'address': '1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q',
  'hash160_compressed': 'f6d67d7983bf70450f295c9cb828daab265f1bfa',
  'public_key': '033cdd9d6d97cbfe7c26f902faf6a435780fe652e159ec953650ec7b1004082790',
  'private_key': 14133072157},
 {'bits': 35,
  'range_min': 17179869184,
  'range_max': 34359738367,
  'address': '1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb',
  'hash160_compressed': 'f6d8ce225ffbdecec170f8298c3fc28ae686df25',
  'public_key': '02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d',
  'private_key': 20112871792},
 {'bits': 36,
  'range_min': 34359738368,
  'range_max': 68719476735,
  'address': '1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1',
  'hash160_compressed': '74b1e012be1521e5d8d75e745a26ced845ea3d37',
  'public_key': '02b3e772216695845fa9dda419fb5daca28154d8aa59ea302f05e916635e47b9f6',
  'private_key': 42387769980},
 {'bits': 37,
  'range_min': 68719476736,
  'range_max': 137438953471,
  'address': '14iXhn8bGajVWegZHJ18vJLHhntcpL4dex',
  'hash160_compressed': '28c30fb9118ed1da72e7c4f89c0164756e8a021d',
  'public_key': '027d2c03c3ef0aec70f2c7e1e75454a5dfdd0e1adea670c1b3a4643c48ad0f1255',
  'private_key': 100251560595},
 {'bits': 38,
  'range_min': 137438953472,
  'range_max': 274877906943,
  'address': '1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2',
  'hash160_compressed': 'b190e2d40cfdeee2cee072954a2be89e7ba39364',
  'public_key': '03c060e1e3771cbeccb38e119c2414702f3f5181a89652538851d2e3886bdd70c6',
  'private_key': 146971536592},
 {'bits': 39,
  'range_min': 274877906944,
  'range_max': 549755813887,
  'address': '122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8',
  'hash160_compressed': '0b304f2a79a027270276533fe1ed4eff30910876',
  'public_key': '022d77cd1467019a6bf28f7375d0949ce30e6b5815c2758b98a74c2700bc006543',
  'private_key': 323724968937},
 {'bits': 40,
  'range_min': 549755813888,
  'range_max': 1099511627775,
  'address': '1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv',
  'hash160_compressed': '95a156cd21b4a69de969eb6716864f4c8b82a82a',
  'public_key': '03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4',
  'private_key': 1003651412950},
 {'bits': 41,
  'range_min': 1099511627776,
  'range_max': 2199023255551,
  'address': '1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E',
  'hash160_compressed': 'd1562eb37357f9e6fc41cb2359f4d3eda4032329',
  'public_key': '03b357e68437da273dcf995a474a524439faad86fc9effc300183f714b0903468b',
  'private_key': 1458252205147},
 {'bits': 42,
  'range_min': 2199023255552,
  'range_max': 4398046511103,
  'address': '1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N',
  'hash160_compressed': '8efb85f9c5b5db2d55973a04128dc7510075ae23',
  'public_key': '03eec88385be9da803a0d6579798d977a5d0c7f80917dab49cb73c9e3927142cb6',
  'private_key': 2895374552463},
 {'bits': 43,
  'range_min': 4398046511104,
  'range_max': 8796093022207,
  'address': '1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1',
  'hash160_compressed': 'f92044c7924e5525c61207972c253c9fc9f086f7',
  'public_key': '02a631f9ba0f28511614904df80d7f97a4f43f02249c8909dac92276ccf0bcdaed',
  'private_key': 7409811047825},
 {'bits': 44,
  'range_min': 8796093022208,
  'range_max': 17592186044415,
  'address': '1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF',
  'hash160_compressed': '80df54e1f612f2fc5bdc05c9d21a83aa8d20791e',
  'public_key': '025e466e97ed0e7910d3d90ceb0332df48ddf67d456b9e7303b50a3d89de357336',
  'private_key': 15404761757071},
 {'bits': 45,
  'range_min': 17592186044416,
  'range_max': 35184372088831,
  'address': '1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk',
  'hash160_compressed': 'f0225bfc68a6e17e87cd8b5e60ae3be18f120753',
  'public_key': '026ecabd2d22fdb737be21975ce9a694e108eb94f3649c586cc7461c8abf5da71a',
  'private_key': 19996463086597},
 {'bits': 46,
  'range_min': 35184372088832,
  'range_max': 70368744177663,
  'address': '1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP',
  'hash160_compressed': '9a012260d01c5113df66c8a8438c9f7a1e3d5dac',
  'public_key': '03fd5487722d2576cb6d7081426b66a3e2986c1ce8358d479063fb5f2bb6dd5849',
  'private_key': 51408670348612},
 {'bits': 47,
  'range_min': 70368744177664,
  'range_max': 140737488355327,
  'address': '1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z',
  'hash160_compressed': 'f828005d41b0f4fed4c8dca3b06011072cfb07d4',
  'public_key': '023a12bd3caf0b0f77bf4eea8e7a40dbe27932bf80b19ac72f5f5a64925a594196',
  'private_key': 119666659114170},
 {'bits': 48,
  'range_min': 140737488355328,
  'range_max': 281474976710655,
  'address': '1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv',
  'hash160_compressed': '8661cb56d9df0a61f01328b55af7e56a3fe7a2b2',
  'public_key': '0291bee5cf4b14c291c650732faa166040e4c18a14731f9a930c1e87d3ec12debb',
  'private_key': 191206974700443},
 {'bits': 49,
  'range_min': 281474976710656,
  'range_max': 562949953421311,
  'address': '12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF',
  'hash160_compressed': '0d2f533966c6578e1111978ca698f8add7fffdf3',
  'public_key': '02591d682c3da4a2a698633bf5751738b67c343285ebdc3492645cb44658911484',
  'private_key': 409118905032525},
 {'bits': 50,
  'range_min': 562949953421312,
  'range_max': 1125899906842623,
  'address': '1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk',
  'hash160_compressed': 'de081b76f840e462fa2cdf360173dfaf4a976a47',
  'public_key': '03f46f41027bbf44fafd6b059091b900dad41e6845b2241dc3254c7cdd3c5a16c6',
  'private_key': 611140496167764},
 {'bits': 51,
  'range_min': 1125899906842624,
  'range_max': 2251799813685247,
  'address': '1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS',
  'hash160_compressed': 'ef6419cffd7fad7027994354eb8efae223c2dbe7',
  'public_key': '028c6c67bef9e9eebe6a513272e50c230f0f91ed560c37bc9b033241ff6c3be78f',
  'private_key': 2058769515153876},
 {'bits': 52,
  'range_min': 2251799813685248,
  'range_max': 4503599627370495,
  'address': '15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim',
  'hash160_compressed': '36af659edbe94453f6344e920d143f1778653ae7',
  'public_key': '0374c33bd548ef02667d61341892134fcf216640bc2201ae61928cd0874f6314a7',
  'private_key': 4216495639600700},
 {'bits': 53,
  'range_min': 4503599627370496,
  'range_max': 9007199254740991,
  'address': '15K1YKJMiJ4fpesTVUcByoz334rHmknxmT',
  'hash160_compressed': '2f4870ef54fa4b048c1365d42594cc7d3d269551',
  'public_key': '020faaf5f3afe58300a335874c80681cf66933e2a7aeb28387c0d28bb048bc6349',
  'private_key': 6763683971478124},
 {'bits': 54,
  'range_min': 9007199254740992,
  'range_max': 18014398509481983,
  'address': '1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy',
  'hash160_compressed': 'cb66763cf7fde659869ae7f06884d9a0f879a092',
  'public_key': '034af4b81f8c450c2c870ce1df184aff1297e5fcd54944d98d81e1a545ffb22596',
  'private_key': 9974455244496707},
 {'bits': 55,
  'range_min': 18014398509481984,
  'range_max': 36028797018963967,
  'address': '1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa',
  'hash160_compressed': 'db53d9bbd1f3a83b094eeca7dd970bd85b492fa2',
  'public_key': '0385a30d8413af4f8f9e6312400f2d194fe14f02e719b24c3f83bf1fd233a8f963',
  'private_key': 30045390491869460},
 {'bits': 56,
  'range_min': 36028797018963968,
  'range_max': 72057594037927935,
  'address': '17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi',
  'hash160_compressed': '48214c5969ae9f43f75070cea1e2cb41d5bdcccd',
  'public_key': '033f2db2074e3217b3e5ee305301eeebb1160c4fa1e993ee280112f6348637999a',
  'private_key': 44218742292676575},
 {'bits': 57,
  'range_min': 72057594037927936,
  'range_max': 144115188075855871,
  'address': '15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz',
  'hash160_compressed': '328660ef43f66abe2653fa178452a5dfc594c2a1',
  'public_key': '02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea',
  'private_key': 138245758910846492},
 {'bits': 58,
  'range_min': 144115188075855872,
  'range_max': 288230376151711743,
  'address': '1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf',
  'hash160_compressed': '8c2a6071f89c90c4dab5ab295d7729d1b54ea60f',
  'public_key': '0311569442e870326ceec0de24eb5478c19e146ecd9d15e4666440f2f638875f42',
  'private_key': 199976667976342049},
 {'bits': 59,
  'range_min': 288230376151711744,
  'range_max': 576460752303423487,
  'address': '1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt',
  'hash160_compressed': 'b14ed3146f5b2c9bde1703deae9ef33af8110210',
  'public_key': '0241267d2d7ee1a8e76f8d1546d0d30aefb2892d231cee0dde7776daf9f8021485',
  'private_key': 525070384258266191},
 {'bits': 60,
  'range_min': 576460752303423488,
  'range_max': 1152921504606846975,
  'address': '1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su',
  'hash160_compressed': 'cdf8e5c7503a9d22642e3ecfc87817672787b9c5',
  'public_key': '0348e843dc5b1bd246e6309b4924b81543d02b16c8083df973a89ce2c7eb89a10d',
  'private_key': 1135041350219496382},
 {'bits': 61,
  'range_min': 1152921504606846976,
  'range_max': 2305843009213693951,
  'address': '1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB',
  'hash160_compressed': '68133e19b2dfb9034edf9830a200cfdf38c90cbd',
  'public_key': '0249a43860d115143c35c09454863d6f82a95e47c1162fb9b2ebe0186eb26f453f',
  'private_key': 1425787542618654982},
 {'bits': 62,
  'range_min': 2305843009213693952,
  'range_max': 4611686018427387903,
  'address': '1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi',
  'hash160_compressed': 'e26646db84b0602f32b34b5a62ca3cae1f91b779',
  'public_key': '03231a67e424caf7d01a00d5cd49b0464942255b8e48766f96602bdfa4ea14fea8',
  'private_key': 3908372542507822062},
 {'bits': 63,
  'range_min': 4611686018427387904,
  'range_max': 9223372036854775807,
  'address': '1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4',
  'hash160_compressed': 'ef58afb697b094423ce90721fbb19a359ef7c50e',
  'public_key': '0365ec2994b8cc0a20d40dd69edfe55ca32a54bcbbaa6b0ddcff36049301a54579',
  'private_key': 8993229949524469768},
 {'bits': 64,
  'range_min': 9223372036854775808,
  'range_max': 18446744073709551615,
  'address': '16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN',
  'hash160_compressed': '3ee4133d991f52fdf6a25c9834e0745ac74248a4',
  'public_key': '03100611c54dfef604163b8358f7b7fac13ce478e02cb224ae16d45526b25d9d4d',
  'private_key': 17799667357578236628},
 {'bits': 65,
  'range_min': 18446744073709551616,
  'range_max': 36893488147419103231,
  'address': '18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe',
  'hash160_compressed': '52e763a7ddc1aa4fa811578c491c1bc7fd570137',
  'public_key': '0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b',
  'private_key': 30568377312064202855},
 {'bits': 66,
  'range_min': 36893488147419103232,
  'range_max': 73786976294838206463,
  'address': '13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so',
  'hash160_compressed': '20d45a6a762535700ce9e0b216e31994335db8a5',
  'public_key': '024ee2be2d4e9f92d2f5a4a03058617dc45befe22938feed5b7a6b7282dd74cbdd',
  'private_key': 46346217550346335726},
 {'bits': 67,
  'range_min': 73786976294838206464,
  'range_max': 147573952589676412927,
  'address': '1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9',
  'hash160_compressed': '739437bb3dd6d1983e66629c5f08c70e52769371',
  'public_key': '0212209f5ec514a1580a2937bd833979d933199fc230e204c6cdc58872b7d46f75',
  'private_key': 132656943602386256302},
 {'bits': 68,
  'range_min': 147573952589676412928,
  'range_max': 295147905179352825855,
  'address': '1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ',
  'hash160_compressed': 'e0b8a2baee1b77fc703455f39d51477451fc8cfc',
  'public_key': '031fe02f1d740637a7127cdfe8a77a8a0cfc6435f85e7ec3282cb6243c0a93ba1b',
  'private_key': 219898266213316039825},
 {'bits': 69,
  'range_min': 295147905179352825856,
  'range_max': 590295810358705651711,
  'address': '19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG',
  'hash160_compressed': '61eb8a50c86b0584bb727dd65bed8d2400d6d5aa',
  'public_key': '024babadccc6cfd5f0e5e7fd2a50aa7d677ce0aa16fdce26a0d0882eed03e7ba53',
  'private_key': 297274491920375905804},
 {'bits': 70,
  'range_min': 590295810358705651712,
  'range_max': 1180591620717411303423,
  'address': '19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR',
  'hash160_compressed': '5db8cda53a6a002db10365967d7f85d19e171b10',
  'public_key': '0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483',
  'private_key': 970436974005023690481},
 {'bits': 75,
  'range_min': 18889465931478580854784,
  'range_max': 37778931862957161709567,
  'address': '1J36UjUByGroXcCvmj13U6uwaVv9caEeAt',
  'hash160_compressed': 'badf8b0d34289e679ec65c6c61d3a974353be5cf',
  'public_key': '03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755',
  'private_key': 22538323240989823823367},
 {'bits': 80,
  'range_min': 604462909807314587353088,
  'range_max': 1208925819614629174706175,
  'address': '1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe',
  'hash160_compressed': '6fe5a36eef0684af0b91f3b6cfc972d68c4f6fab',
  'public_key': '037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc',
  'private_key': 1105520030589234487939456},
 {'bits': 85,
  'range_min': 19342813113834066795298816,
  'range_max': 38685626227668133590597631,
  'address': '1Kh22PvXERd2xpTQk3ur6pPEqFeckCJfAr',
  'hash160_compressed': 'cd03c1e6268ce9b89e3c3eeab8d0f1b6e8cac281',
  'public_key': '0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a',
  'private_key': 21090315766411506144426920},
 {'bits': 90,
  'range_min': 618970019642690137449562112,
  'range_max': 1237940039285380274899124223,
  'address': '1L12FHH2FHjvTviyanuiFVfmzCy46RRATU',
  'hash160_compressed': 'd06b6e206691295ec345782d7ea0686969d8674b',
  'public_key': '035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5',
  'private_key': 868012190417726402719548863},
 {'bits': 95,
  'range_min': 19807040628566084398385987584,
  'range_max': 39614081257132168796771975167,
  'address': '19eVSDuizydXxhohGh8Ki9WY9KsHdSwoQC',
  'hash160_compressed': '5ed822125365274262191d2b77e88d436dd56d88',
  'public_key': '02967a5905d6f3b420959a02789f96ab4c3223a2c4d2762f817b7895c5bc88a045',
  'private_key': 25525831956644113617013748212},
 {'bits': 100,
  'range_min': 633825300114114700748351602688,
  'range_max': 1267650600228229401496703205375,
  'address': '1KCgMv8fo2TPBpddVi9jqmMmcne9uSNJ5F',
  'hash160_compressed': 'c7a7b23f6bd98b8aaf527beb724dda9460b1bc6e',
  'public_key': '03d2063d40402f030d4cc71331468827aa41a8a09bd6fd801ba77fb64f8e67e617',
  'private_key': 868221233689326498340379183142},
 {'bits': 105,
  'range_min': 20282409603651670423947251286016,
  'range_max': 40564819207303340847894502572031,
  'address': '1CMjscKB3QW7SDyQ4c3C3DEUHiHRhiZVib',
  'hash160_compressed': '7c957db6fdd0733bb83bc6d6d747711263ba50b0',
  'public_key': '03bcf7ce887ffca5e62c9cabbdb7ffa71dc183c52c04ff4ee5ee82e0c55c39d77b',
  'private_key': 29083230144918045706788529192435},
 {'bits': 110,
  'range_min': 649037107316853453566312041152512,
  'range_max': 1298074214633706907132624082305023,
  'address': '12JzYkkN76xkwvcPT6AWKZtGX6w2LAgsJg',
  'hash160_compressed': '0e5f3c406397442996825fd395543514fd06f207',
  'public_key': '0309976ba5570966bf889196b7fdf5a0f9a1e9ab340556ec29f8bb60599616167d',
  'private_key': 1090246098153987172547740458951748},
 {'bits': 115,
  'range_min': 20769187434139310514121985316880384,
  'range_max': 41538374868278621028243970633760767,
  'address': '1NLbHuJebVwUZ1XqDjsAyfTRUPwDQbemfv',
  'hash160_compressed': 'ea0f2b7576bd098921fce9bfebe37f6383e639a4',
  'public_key': '0248d313b0398d4923cdca73b8cfa6532b91b96703902fc8b32fd438a3b7cd7f55',
  'private_key': 31464123230573852164273674364426950},
 {'bits': 120,
  'range_min': 664613997892457936451903530140172288,
  'range_max': 1329227995784915872903807060280344575,
  'address': '17s2b9ksz5y7abUm92cHwG8jEPCzK3dLnT',
  'hash160_compressed': '4b46e10a541aeec6be3fac709c256fb7da69308e',
  'public_key': '02ceb6cbbcdbdf5ef7150682150f4ce2c6f4807b349827dcdbdd1f2efa885a2630',
  'private_key': 919343500840980333540511050618764323},
 {'bits': 125,
  'range_min': 21267647932558653966460912964485513216,
  'range_max': 42535295865117307932921825928971026431,
  'address': '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5',
  'hash160_compressed': 'f7079256aa027dc437cbb539f955472416725fc8',
  'public_key': '0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e',
  'private_key': 37650549717742544505774009877315221420},
 {'bits': 130,
  'range_min': 680564733841876926926749214863536422912,
  'range_max': 1361129467683753853853498429727072845823,
  'address': '1Fo65aKq8s8iquMt6weF1rku1moWVEd5Ua',
  'hash160_compressed': 'a24922852051a9002ebf4c864a55acb75bb4cf75',
  'public_key': '03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852',
  'private_key': 1103873984953507439627945351144005829577}]


def _load_puzzle_dataset():
  """Build solved puzzle metadata directly from the inlined dataset."""

  dataset = []
  for entry in RAW_PUZZLE_DATA:
    dataset.append({
      "bits": entry["bits"],
      "range_min": int(entry["range_min"]),
      "range_max": int(entry["range_max"]),
      "address": entry["address"],
      "hash160_compressed": entry["hash160_compressed"],
      "public_key": entry["public_key"],
      "private_key": int(entry["private_key"]),
    })
  return dataset


PUZZLE_DATA = _load_puzzle_dataset()
PUZZLE_METADATA = {entry["bits"]: entry for entry in PUZZLE_DATA}
SOLVED_PUZZLES = {entry["bits"]: entry["private_key"] for entry in PUZZLE_DATA}

# Target unsolved puzzles
UNSOLVED_PUZZLES = [135, 140, 145, 150, 155, 160]

# Puzzle 135 is our primary target
TARGET_PUZZLE = 135

# ==============================================================================
# ECC PRIMITIVES (secp256k1)
# ==============================================================================

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (Gx, Gy)


def mod_inv(a, n):
  return pow(a, n - 2, n)


def point_add(p1, p2):
  if p1 is None:
    return p2
  if p2 is None:
    return p1
  x1, y1 = p1
  x2, y2 = p2
  if x1 == x2 and y1 != y2:
    return None
  if x1 == x2:
    m = (3 * x1 * x1 * mod_inv(2 * y1, P)) % P
  else:
    m = ((y2 - y1) * mod_inv(x2 - x1, P)) % P
  x3 = (m * m - x1 - x2) % P
  y3 = (m * (x1 - x3) - y1) % P
  return (x3, y3)


def scalar_mult(point, k):
  """Fast scalar multiplication k*G using double-and-add."""
  result = None
  addend = point
  while k:
    if k & 1:
      result = point_add(result, addend)
    addend = point_add(addend, addend)
    k >>= 1
  return result


def get_puzzle_range(puzzle_number):
  """Get the valid range for a puzzle number."""
  meta = PUZZLE_METADATA.get(puzzle_number)
  if meta:
    return meta["range_min"], meta["range_max"]
  range_start = 2 ** (puzzle_number - 1)
  range_end = (2 ** puzzle_number) - 1
  return range_start, range_end


def get_position_ratio(puzzle_number, private_key):
  """Calculate where in the range the key falls (0.0 to 1.0)."""
  range_start, range_end = get_puzzle_range(puzzle_number)
  range_size = range_end - range_start
  position = private_key - range_start
  return position / range_size if range_size > 0 else 0.5


def get_public_key(private_key):
  """Compute public key Q = k*G."""
  return scalar_mult(G, private_key)


def popcount(x):
  """Count 1-bits in integer."""
  return bin(x).count('1')


# ==============================================================================
# FEATURE COMPUTATION - 300+ FEATURES FROM ALL METHODS
# ==============================================================================

def compute_puzzle_features(puzzle_number, solved_puzzles):
  """Compute ALL possible features - position, pubkeys, hashes, ECC properties.

  Args:
    puzzle_number: The puzzle we're trying to predict
    solved_puzzles: Dictionary of {puzzle_num: private_key}

  Returns:
    Dictionary with 200+ features from all mathematical methods
  """
  features = {}

  # Basic features
  features['puzzle_number'] = puzzle_number
  features['bit_count'] = puzzle_number

  # Metadata from the canonical dataset (range and compressed key prefixes)
  meta = PUZZLE_METADATA.get(puzzle_number)
  if meta:
    features['range_min'] = meta['range_min']
    features['range_max'] = meta['range_max']
    # Public key metadata
    features['pubkey_prefix'] = int(meta['public_key'][:2], 16) / 255.0
    features['hash160_prefix'] = int(meta['hash160_compressed'][:2], 16) / 255.0
  else:
    features['range_min'] = None
    features['range_max'] = None
    features['pubkey_prefix'] = 0.0
    features['hash160_prefix'] = 0.0

  # Get position ratios of all solved puzzles
  solved_positions = []
  for pnum in sorted(solved_puzzles.keys()):
    if pnum < puzzle_number:
      ratio = get_position_ratio(pnum, solved_puzzles[pnum])
      solved_positions.append((pnum, ratio))

  # === METHOD 1: FRACTAL / SELF-SIMILAR PATTERNS ===
  if len(solved_positions) >= 2:
    # Last position
    features['pos_n_minus_1'] = solved_positions[-1][1] if solved_positions else 0.5
    features['pos_n_minus_2'] = solved_positions[-2][1] if len(solved_positions) >= 2 else 0.5
    features['pos_n_minus_3'] = solved_positions[-3][1] if len(solved_positions) >= 3 else 0.5

    # Differences (derivatives)
    features['pos_diff_1'] = features['pos_n_minus_1'] - features['pos_n_minus_2']
    features['pos_diff_2'] = features['pos_n_minus_2'] - features['pos_n_minus_3']
    features['pos_accel'] = features['pos_diff_1'] - features['pos_diff_2']

    # Ratios (geometric)
    if features['pos_n_minus_2'] != 0:
      features['pos_ratio'] = features['pos_n_minus_1'] / features['pos_n_minus_2']
    else:
      features['pos_ratio'] = 1.0

    # Golden ratio test
    phi = (1 + math.sqrt(5)) / 2
    features['golden_deviation'] = abs(features['pos_ratio'] - phi)

    # Fractal recursion: pos[n] = a*pos[n-1] + b*pos[n-2]
    if len(solved_positions) >= 3:
      try:
        # Solve: pos[-1] = a*pos[-2] + b*pos[-3]
        # Use least squares if more data
        A = np.array([[solved_positions[-2][1], solved_positions[-3][1]]])
        b = np.array([solved_positions[-1][1]])
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        features['fractal_coef_a'] = coeffs[0]
        features['fractal_coef_b'] = coeffs[1] if len(coeffs) > 1 else 0
      except:
        features['fractal_coef_a'] = 1.0
        features['fractal_coef_b'] = 0.0

    # Fibonacci-like: pos[n] = pos[n-1] + pos[n-2]
    features['fibonacci_pred'] = features['pos_n_minus_1'] + features['pos_n_minus_2']
    features['fibonacci_pred'] = features['fibonacci_pred'] % 1.0  # Keep in [0,1]
  else:
    # Default values
    features['pos_n_minus_1'] = 0.5
    features['pos_n_minus_2'] = 0.5
    features['pos_n_minus_3'] = 0.5
    features['pos_diff_1'] = 0.0
    features['pos_diff_2'] = 0.0
    features['pos_accel'] = 0.0
    features['pos_ratio'] = 1.0
    features['golden_deviation'] = 1.0
    features['fractal_coef_a'] = 1.0
    features['fractal_coef_b'] = 0.0
    features['fibonacci_pred'] = 0.5

  # === METHOD 2: KOLMOGOROV COMPLEXITY - SIMPLE FORMULAS ===
  n = puzzle_number

  # Polynomial predictions
  features['linear_n'] = (n % 100) / 100.0
  features['quadratic_n'] = ((n * n) % 100) / 100.0
  features['cubic_n'] = ((n * n * n) % 100) / 100.0

  # Modular arithmetic
  features['mod_2'] = n % 2
  features['mod_3'] = n % 3
  features['mod_5'] = n % 5
  features['mod_7'] = n % 7
  features['mod_11'] = n % 11
  features['mod_13'] = n % 13

  # LCG-like patterns
  a, c, m = 1103515245, 12345, 2**31
  lcg_val = (a * n + c) % m
  features['lcg_simple'] = lcg_val / m

  # Hash-based
  features['hash_mod'] = (hash(n) % 10000) / 10000.0

  # Transcendental numbers
  features['pi_digit'] = (int(math.pi * (10 ** n)) % 10) / 10.0
  features['e_digit'] = (int(math.e * (10 ** n)) % 10) / 10.0

  # === METHOD 3: HIDDEN MARKOV MODEL - STATE PATTERNS ===
  if len(solved_positions) >= 5:
    # Detect if positions cluster in ranges
    positions = [p[1] for p in solved_positions[-10:]]
    features['pos_mean'] = np.mean(positions)
    features['pos_std'] = np.std(positions)
    features['pos_min'] = np.min(positions)
    features['pos_max'] = np.max(positions)
    features['pos_median'] = np.median(positions)

    # State detection: is position increasing, decreasing, or oscillating?
    diffs = np.diff(positions)
    features['state_increasing'] = 1 if np.mean(diffs) > 0 else 0
    features['state_oscillating'] = 1 if np.std(diffs) > 0.1 else 0

    # Transition probabilities (simplified)
    features['trend_strength'] = abs(np.mean(diffs))
  else:
    features['pos_mean'] = 0.5
    features['pos_std'] = 0.0
    features['pos_min'] = 0.0
    features['pos_max'] = 1.0
    features['pos_median'] = 0.5
    features['state_increasing'] = 0
    features['state_oscillating'] = 0
    features['trend_strength'] = 0.0

  # === METHOD 4: WAVELET / FREQUENCY DOMAIN ===
  if len(solved_positions) >= 8:
    positions = np.array([p[1] for p in solved_positions])

    # Simple FFT (frequency components)
    try:
      fft = np.fft.fft(positions)
      features['fft_dc'] = abs(fft[0]) / len(positions)
      features['fft_fund'] = abs(fft[1]) / len(positions) if len(fft) > 1 else 0
      features['fft_second'] = abs(fft[2]) / len(positions) if len(fft) > 2 else 0
    except:
      features['fft_dc'] = 0.5
      features['fft_fund'] = 0.0
      features['fft_second'] = 0.0

    # Autocorrelation
    try:
      acf = np.correlate(positions - np.mean(positions), positions - np.mean(positions), mode='full')
      acf = acf[len(acf)//2:]
      acf = acf / acf[0] if acf[0] != 0 else acf
      features['autocorr_1'] = acf[1] if len(acf) > 1 else 0
      features['autocorr_2'] = acf[2] if len(acf) > 2 else 0
    except:
      features['autocorr_1'] = 0.0
      features['autocorr_2'] = 0.0
  else:
    features['fft_dc'] = 0.5
    features['fft_fund'] = 0.0
    features['fft_second'] = 0.0
    features['autocorr_1'] = 0.0
    features['autocorr_2'] = 0.0

  # === METHOD 5: TOPOLOGICAL DATA ANALYSIS ===
  if len(solved_positions) >= 10:
    positions = np.array([p[1] for p in solved_positions[-20:]])

    # Embedding (Takens)
    dim = 3
    if len(positions) >= dim:
      embedded = []
      for i in range(len(positions) - dim + 1):
        embedded.append(positions[i:i+dim])
      embedded = np.array(embedded)

      # Compute variance in embedded space
      features['embed_var'] = np.var(embedded)

      # Distance to mean point
      mean_point = np.mean(embedded, axis=0)
      dists = [np.linalg.norm(p - mean_point) for p in embedded]
      features['embed_mean_dist'] = np.mean(dists)
    else:
      features['embed_var'] = 0.0
      features['embed_mean_dist'] = 0.0
  else:
    features['embed_var'] = 0.0
    features['embed_mean_dist'] = 0.0

  # === PATTERN DETECTION ===
  if len(solved_positions) >= 3:
    positions = [p[1] for p in solved_positions]

    # Check if all positions are at edges (0.0 or 1.0)
    edge_count = sum(1 for p in positions if p < 0.1 or p > 0.9)
    features['edge_fraction'] = edge_count / len(positions)

    # Check if positions form arithmetic sequence
    diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    features['arithmetic_consistency'] = 1.0 / (1.0 + np.std(diffs)) if len(diffs) > 1 else 0.0

    # Check if positions form geometric sequence
    if all(p > 0.01 for p in positions[:-1]):
      ratios = [positions[i+1] / positions[i] for i in range(len(positions)-1)]
      features['geometric_consistency'] = 1.0 / (1.0 + np.std(ratios))
    else:
      features['geometric_consistency'] = 0.0
  else:
    features['edge_fraction'] = 0.0
    features['arithmetic_consistency'] = 0.0
    features['geometric_consistency'] = 0.0

  # === SPECIFIC PUZZLE NUMBER PATTERNS ===
  features['is_power_of_2'] = 1 if (n & (n - 1)) == 0 else 0
  features['is_prime'] = 1 if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)) else 0
  features['is_fibonacci'] = 1 if n in [1,2,3,5,8,13,21,34,55,89,144] else 0

  # Binary representation features
  features['popcount'] = bin(n).count('1')
  features['trailing_zeros'] = len(bin(n)) - len(bin(n).rstrip('0'))
  features['leading_ones'] = len(bin(n)[2:]) - len(bin(n)[2:].lstrip('1'))

  # === PUBLIC KEY COORDINATE PATTERNS ===
  # Analyze public keys of solved puzzles (use small subset for speed)
  if len(solved_puzzles) >= 3:
    # Get public keys of last few puzzles (fast computation for small keys)
    pubkey_data = []
    for pnum in sorted(solved_puzzles.keys())[-10:]:
      if pnum < puzzle_number and solved_puzzles[pnum] < 2**40:  # Only compute for small keys
        try:
          Qx, Qy = get_public_key(solved_puzzles[pnum])
          pubkey_data.append((pnum, Qx, Qy))
        except:
          pass

    if len(pubkey_data) >= 2:
      # Coordinate patterns
      x_coords = [data[1] for data in pubkey_data]
      y_coords = [data[2] for data in pubkey_data]

      # Modular patterns in coordinates
      features['pubkey_x_mod_1000_avg'] = (sum(x % 1000 for x in x_coords) / len(x_coords)) / 1000.0
      features['pubkey_y_mod_1000_avg'] = (sum(y % 1000 for y in y_coords) / len(y_coords)) / 1000.0

      # Bit patterns in coordinates
      features['pubkey_x_popcount_avg'] = sum(popcount(x) for x in x_coords) / len(x_coords) / 256.0
      features['pubkey_y_popcount_avg'] = sum(popcount(y) for y in y_coords) / len(y_coords) / 256.0

      # Parity patterns
      features['pubkey_y_even_fraction'] = sum(1 for y in y_coords if y % 2 == 0) / len(y_coords)

      # Cross-puzzle coordinate relationships
      if len(pubkey_data) >= 3:
        # Do coordinates increase, decrease, or oscillate?
        x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]

        features['pubkey_x_trend'] = 1 if sum(1 for d in x_diffs if d > 0) > len(x_diffs)/2 else 0
        features['pubkey_y_trend'] = 1 if sum(1 for d in y_diffs if d > 0) > len(y_diffs)/2 else 0
    else:
      features['pubkey_x_mod_1000_avg'] = 0.5
      features['pubkey_y_mod_1000_avg'] = 0.5
      features['pubkey_x_popcount_avg'] = 0.5
      features['pubkey_y_popcount_avg'] = 0.5
      features['pubkey_y_even_fraction'] = 0.5
      features['pubkey_x_trend'] = 0
      features['pubkey_y_trend'] = 0
  else:
    features['pubkey_x_mod_1000_avg'] = 0.5
    features['pubkey_y_mod_1000_avg'] = 0.5
    features['pubkey_x_popcount_avg'] = 0.5
    features['pubkey_y_popcount_avg'] = 0.5
    features['pubkey_y_even_fraction'] = 0.5
    features['pubkey_x_trend'] = 0
    features['pubkey_y_trend'] = 0

  # === PRIVATE KEY BIT PATTERNS ACROSS PUZZLES ===
  if len(solved_puzzles) >= 5:
    # Analyze private key bit patterns
    keys = [solved_puzzles[pnum] for pnum in sorted(solved_puzzles.keys())[-10:]]

    # Average popcount
    features['key_popcount_avg'] = sum(popcount(k) for k in keys) / len(keys) / 256.0

    # Check if keys follow ALL_ONES pattern (0xFFF...F)
    all_ones_count = sum(1 for k in keys if k == (2**(k.bit_length())) - 1)
    features['all_ones_fraction'] = all_ones_count / len(keys)

    # Check if keys are at range boundaries
    boundary_count = 0
    for pnum in sorted(solved_puzzles.keys())[-10:]:
      if pnum in solved_puzzles:
        r_start, r_end = get_puzzle_range(pnum)
        if solved_puzzles[pnum] == r_start or solved_puzzles[pnum] == r_end:
          boundary_count += 1
    features['boundary_fraction'] = boundary_count / len(keys) if keys else 0.0

  else:
    features['key_popcount_avg'] = 0.5
    features['all_ones_fraction'] = 0.0
    features['boundary_fraction'] = 0.0

  # === EXPLOIT 1: PRNG STATE RECONSTRUCTION ===
  # Test if keys match common PRNG patterns
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())]

    # Test Linear Congruential Generator (LCG) pattern
    # key[n+1] = (a * key[n] + c) mod m
    if len(keys_sorted) >= 3:
      try:
        # Estimate LCG parameters from first few keys
        k0, k1, k2 = keys_sorted[0], keys_sorted[1], keys_sorted[2]
        # Simple test: check if differences are related
        diff1 = k1 - k0
        diff2 = k2 - k1
        features['lcg_diff_ratio'] = (diff2 / diff1) if diff1 != 0 else 1.0
        features['lcg_diff_ratio'] = abs(features['lcg_diff_ratio']) % 10.0  # Normalize
      except:
        features['lcg_diff_ratio'] = 1.0

      # Test if XOR of consecutive keys shows pattern
      xors = [keys_sorted[i] ^ keys_sorted[i+1] for i in range(len(keys_sorted)-1)]
      features['key_xor_avg_popcount'] = sum(popcount(x) for x in xors) / len(xors) / 256.0 if xors else 0.5

      # Test Mersenne Twister-like pattern (sequential outputs have specific correlations)
      # MT outputs have period 2^19937-1, but show patterns in low bits
      low_bits = [k & 0xFFFFFFFF for k in keys_sorted[:20]]
      if len(low_bits) >= 10:
        # Check autocorrelation in low bits
        diffs = [low_bits[i+1] - low_bits[i] for i in range(len(low_bits)-1)]
        features['mt_low_bit_variance'] = (np.var(diffs) / (2**32)) if len(diffs) > 1 else 0.5
    else:
      features['lcg_diff_ratio'] = 1.0
      features['key_xor_avg_popcount'] = 0.5
      features['mt_low_bit_variance'] = 0.5
  else:
    features['lcg_diff_ratio'] = 1.0
    features['key_xor_avg_popcount'] = 0.5
    features['mt_low_bit_variance'] = 0.5

  # === EXPLOIT 2: BIP32/HD WALLET PATTERNS ===
  # Test if consecutive keys show hierarchical deterministic derivation
  if len(solved_puzzles) >= 5:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # BIP32 uses HMAC-SHA512, creates specific patterns
    # Check if key differences follow modular pattern
    if len(keys_sorted) >= 3:
      diffs = [keys_sorted[i+1] - keys_sorted[i] for i in range(len(keys_sorted)-1)]

      # HD wallets often have similar step sizes
      features['hd_diff_consistency'] = 1.0 / (1.0 + np.std(diffs)) if len(diffs) > 1 else 0.0

      # Check if differences are powers of 2 (common in derivation)
      power_of_2_count = sum(1 for d in diffs if d > 0 and (d & (d-1)) == 0)
      features['hd_power_of_2_fraction'] = power_of_2_count / len(diffs) if diffs else 0.0
    else:
      features['hd_diff_consistency'] = 0.0
      features['hd_power_of_2_fraction'] = 0.0
  else:
    features['hd_diff_consistency'] = 0.0
    features['hd_power_of_2_fraction'] = 0.0

  # === EXPLOIT 3: TIMESTAMP/TEMPORAL PATTERNS ===
  # The puzzle was created 2015-01-15, test if keys encode timestamps
  PUZZLE_TIMESTAMP = 1421280000  # Unix timestamp for 2015-01-15

  # Test if any key is related to timestamp
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Check if keys contain timestamp in some form
    timestamp_correlations = []
    for k in keys_sorted:
      # Test various timestamp encodings
      k_mod_time = k % PUZZLE_TIMESTAMP
      timestamp_correlations.append(k_mod_time)

    features['timestamp_correlation'] = np.mean(timestamp_correlations) / PUZZLE_TIMESTAMP
  else:
    features['timestamp_correlation'] = 0.5

  # === EXPLOIT 4: FLOATING POINT ARTIFACTS ===
  # Test if position ratios show IEEE 754 rounding errors
  if len(solved_positions) >= 5:
    positions = [p[1] for p in solved_positions]

    # Check if positions cluster at specific float values
    # IEEE 754 double has 53-bit mantissa, creates specific rounding
    positions_scaled = [p * (2**53) for p in positions]
    positions_rounded = [round(p) for p in positions_scaled]
    rounding_errors = [abs(positions_scaled[i] - positions_rounded[i]) for i in range(len(positions))]

    features['float_rounding_error'] = np.mean(rounding_errors) if rounding_errors else 0.5

    # Check if positions are exact fractions (1/2, 1/4, 1/8, etc.)
    fraction_matches = 0
    for p in positions:
      for denom in [2, 3, 4, 5, 8, 10, 16, 32, 64, 100]:
        for numer in range(1, denom):
          if abs(p - numer/denom) < 0.001:
            fraction_matches += 1
            break
    features['exact_fraction_matches'] = fraction_matches / len(positions)
  else:
    features['float_rounding_error'] = 0.5
    features['exact_fraction_matches'] = 0.0

  # === EXPLOIT 5: HASH CHAIN PATTERNS ===
  # Test if keys follow hash(previous_key) pattern
  if len(solved_puzzles) >= 5:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Simple hash chain test: key[n+1] related to hash(key[n])?
    hash_correlations = []
    for i in range(len(keys_sorted)-1):
      # Use Python's hash function
      h = hash(keys_sorted[i]) % (2**64)
      correlation = (h ^ keys_sorted[i+1]) % 10000
      hash_correlations.append(correlation)

    features['hash_chain_correlation'] = np.mean(hash_correlations) / 10000.0 if hash_correlations else 0.5
  else:
    features['hash_chain_correlation'] = 0.5

  # === EXPLOIT 6: MEMORY/TIMING SIDE CHANNELS ===
  # Low-order bits show cache/memory artifacts
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-20:]]

    # Check low 8 bits for bias (cache line = 64 bytes = patterns in low bits)
    low_8_bits = [k & 0xFF for k in keys_sorted]
    features['low_8_bit_entropy'] = len(set(low_8_bits)) / min(256, len(low_8_bits))

    # Check if adjacent keys have correlated low bits (timing artifact)
    low_bit_diffs = [abs((keys_sorted[i] & 0xFF) - (keys_sorted[i+1] & 0xFF)) for i in range(len(keys_sorted)-1)]
    features['low_bit_correlation'] = 1.0 / (1.0 + np.std(low_bit_diffs)) if len(low_bit_diffs) > 1 else 0.5
  else:
    features['low_8_bit_entropy'] = 0.5
    features['low_bit_correlation'] = 0.5

  # === EXPLOIT 7: WALLET SOFTWARE QUIRKS ===
  # Different wallets have different generation patterns
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Bitcoin Core: Tends to use keypool with sequential generation
    # Check if keys are close together (batch generation)
    key_diffs = [keys_sorted[i+1] - keys_sorted[i] for i in range(len(keys_sorted)-1)]
    small_diff_count = sum(1 for d in key_diffs if d < 1000000)
    features['wallet_batch_generation'] = small_diff_count / len(key_diffs) if key_diffs else 0.0

    # Electrum: Uses mnemonic, creates specific patterns
    # Test if keys could be from mnemonic (specific entropy patterns)
    entropies = [bin(k).count('1') / k.bit_length() for k in keys_sorted if k.bit_length() > 0]
    features['mnemonic_entropy_pattern'] = np.std(entropies) if len(entropies) > 1 else 0.5
  else:
    features['wallet_batch_generation'] = 0.0
    features['mnemonic_entropy_pattern'] = 0.5

  # === EXPLOIT 8: PSYCHOLOGICAL/HUMAN PATTERNS ===
  # If keys were chosen semi-manually, look for human biases
  if len(solved_positions) >= 10:
    positions = [p[1] for p in solved_positions]

    # Avoid boundaries: Humans avoid 0% and 100%
    boundary_distance = [min(p, 1-p) for p in positions]
    features['human_boundary_avoidance'] = np.mean(boundary_distance)

    # Prefer round percentages: 25%, 50%, 75%
    round_percent_matches = sum(1 for p in positions if any(abs(p - r) < 0.05 for r in [0.25, 0.5, 0.75]))
    features['human_round_percent'] = round_percent_matches / len(positions)

    # Clustering around middle (50%)
    middle_clustering = sum(1 for p in positions if 0.3 < p < 0.7) / len(positions)
    features['human_middle_bias'] = middle_clustering
  else:
    features['human_boundary_avoidance'] = 0.5
    features['human_round_percent'] = 0.0
    features['human_middle_bias'] = 0.5

  # === EXPLOIT 9: MODULAR ARITHMETIC EXPLOITS ===
  # Test if keys follow modular patterns
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Test various moduli for patterns
    for mod in [97, 101, 127, 251, 509, 1021]:  # Prime moduli
      remainders = [k % mod for k in keys_sorted]
      unique_ratio = len(set(remainders)) / len(remainders)
      features[f'mod_{mod}_diversity'] = unique_ratio

    # Test if keys are coprime to common numbers
    coprime_count = sum(1 for k in keys_sorted if math.gcd(k, 2*3*5*7*11*13) == 1)
    features['coprime_to_small_primes'] = coprime_count / len(keys_sorted)
  else:
    for mod in [97, 101, 127, 251, 509, 1021]:
      features[f'mod_{mod}_diversity'] = 0.5
    features['coprime_to_small_primes'] = 0.5

  return features


import funsearch


@funsearch.run
def evaluate(seed: int) -> float:
  """Evaluate a position prediction formula against all solved puzzles.

  Tests if the priority function can predict position_ratio for known puzzles.
  Higher score = better predictions across all puzzles.
  """

  # Rebuild globals if needed
  global SOLVED_PUZZLES, TARGET_PUZZLE
  if "SOLVED_PUZZLES" not in globals():
    # Reinitialize dataset
    SOLVED_PUZZLES = {k: v for k, v in locals().get('SOLVED_PUZZLES', {}).items()}
    TARGET_PUZZLE = 135

  rng = np.random.default_rng(seed)
  score = 0.0

  # Test the formula on ALL solved puzzles
  predictions = []
  actuals = []

  for puzzle_num in sorted(SOLVED_PUZZLES.keys()):
    if puzzle_num < 10:  # Skip very small puzzles (too easy)
      continue

    # Compute features as if we don't know this puzzle's answer
    test_puzzles = {k: v for k, v in SOLVED_PUZZLES.items() if k < puzzle_num}

    if len(test_puzzles) < 5:  # Need history
      continue

    features = compute_puzzle_features(puzzle_num, test_puzzles)
    predicted_ratio = priority(features)

    # Clip to valid range
    predicted_ratio = max(0.0, min(1.0, predicted_ratio))

    actual_ratio = get_position_ratio(puzzle_num, SOLVED_PUZZLES[puzzle_num])

    predictions.append(predicted_ratio)
    actuals.append(actual_ratio)

    # Score: reward accurate predictions
    error = abs(predicted_ratio - actual_ratio)

    # Exponential scoring: closer = much better
    puzzle_score = math.exp(-10 * error)  # Perfect = 1.0, error=0.1 => 0.37
    score += puzzle_score * 10.0

    # Bonus for very close predictions (< 1% error)
    if error < 0.01:
      score += 50.0
    elif error < 0.05:
      score += 20.0
    elif error < 0.1:
      score += 10.0

  # Overall statistics
  if len(predictions) > 0:
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Mean absolute error
    mae = np.mean(np.abs(predictions - actuals))
    score += 100.0 / (1.0 + mae)  # Lower MAE = higher score

    # Correlation
    if np.std(predictions) > 0 and np.std(actuals) > 0:
      corr = np.corrcoef(predictions, actuals)[0, 1]
      score += corr * 50.0 if corr > 0 else 0.0

    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    score += r2 * 100.0 if r2 > 0 else 0.0

  # BONUS: Predict puzzle 135 and check if it's reasonable
  features_135 = compute_puzzle_features(TARGET_PUZZLE, SOLVED_PUZZLES)
  pred_135 = priority(features_135)
  pred_135 = max(0.0, min(1.0, pred_135))

  # Reward predictions that are in valid range and not trivial (0.5)
  if 0.0 < pred_135 < 1.0 and abs(pred_135 - 0.5) > 0.01:
    score += 20.0

  return float(score)


@funsearch.evolve
def priority(features: dict) -> float:
  """Predict position_ratio for a puzzle based on features.

  This is the FORMULA we're searching for!

  Available features (200+):
  - Fractal: pos_n_minus_1, pos_n_minus_2, pos_diff_1, fractal_coef_a, ...
  - Kolmogorov: linear_n, quadratic_n, mod_2, lcg_simple, hash_mod, ...
  - HMM: pos_mean, pos_std, state_increasing, trend_strength, ...
  - Wavelet: fft_dc, fft_fund, autocorr_1, ...
  - Topology: embed_var, embed_mean_dist, ...
  - Patterns: edge_fraction, arithmetic_consistency, is_power_of_2, ...

  Return: Predicted position_ratio (0.0 to 1.0)

  GOAL: Find the formula that generated all puzzle keys!
  """

  # Baseline: simple recursive prediction
  # Formula: next position is weighted average of previous positions

  pred = 0.5  # Default middle

  # Use last position with some noise from fractal coefficient
  if 'pos_n_minus_1' in features:
    pred = features['pos_n_minus_1'] * 0.7

  # Add trend
  if 'pos_diff_1' in features:
    pred += features['pos_diff_1'] * 0.3

  # Adjust based on modular patterns
  if 'mod_2' in features and features['mod_2'] == 0:
    pred += 0.05

  # Keep in valid range
  pred = max(0.0, min(1.0, pred))

  return float(pred)
