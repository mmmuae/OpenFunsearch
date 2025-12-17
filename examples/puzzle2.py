"""Bitcoin Puzzle Private Key Discovery - Open-Ended Approach

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.

GOAL:
Discover the actual private key for puzzle N using ANY method you can imagine.


This is a completely open-ended challenge. You can use:
1) Discrete structure (algebra/number theory)
Modular structure, CRT, continued fractions

2) Dynamics (time evolution / chaos)
Dynamical systems, chaos, symbolic dynamics, ergodic ideas, fractal/self-similar analysis

3) Spectral / multiscale signal lens
Fourier/spectral intuition, phase cues, wavelets, time–frequency methods

4) Statistical learning lens
Robust statistics, rank statistics, kernel methods, ML models

5) Shape-of-data lens (geometry/topology)
Manifold/geometry intuition, topological viewpoints, TDA

6) Simplicity / compression lens
Information theory, MDL, Kolmogorov/algorithmic complexity

7) Latent regime lens (state models)
Automata intuition, state machines, HMMs, regime switching

8) Blending / ensembling lens
Bayesian model averaging, hybrid combinations, model stacking (as a concept)

9) Crypto-specific lens
Cryptographic analysis/attacks, ECC-focused analysis, lattice reduction


WHAT YOU GET:
- Puzzle number N to solve
- Complete historical data of all previously solved puzzles for training and validation
- For each puzzle: private_key, public_key, address, range_min, range_max
- Full access to all cryptographic metadata

- Always return a single, complete Python module starting at column 0
- Only adjust the body of ``priority``; keep function signatures unchanged
- Use **two spaces** per indentation level

WHAT YOU RETURN:
- Your predicted private key for puzzle N (integer)
"""

import math
import numpy as np


# ==============================================================================
# COMPLETE DATASET OF SOLVED BITCOIN PUZZLES
# ==============================================================================

def _get_puzzle_data():
  """Return the complete dataset of solved Bitcoin puzzles.

  Each puzzle contains:
    - bits: Puzzle number (difficulty level)
    - range_min, range_max: Valid range for the private key
    - private_key: The actual private key (SOLUTION)
    - address: Bitcoin address
    - hash160_compressed: Hash160 of compressed public key
    - public_key: Compressed public key (if available)
  """
  return {
    1: {'bits': 1, 'range_min': 1, 'range_max': 1, 'private_key': 1, 'address': '1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH', 'hash160_compressed': '751e76e8199196d454941c45d1b3a323f1433bd6', 'public_key': '0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798'},
    2: {'bits': 2, 'range_min': 2, 'range_max': 3, 'private_key': 3, 'address': '1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb', 'hash160_compressed': '7dd65592d0ab2fe0d0257d571abf032cd9db93dc', 'public_key': '02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9'},
    3: {'bits': 3, 'range_min': 4, 'range_max': 7, 'private_key': 7, 'address': '19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA', 'hash160_compressed': '5dedfbf9ea599dd4e3ca6a80b333c472fd0b3f69', 'public_key': '025cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc'},
    4: {'bits': 4, 'range_min': 8, 'range_max': 15, 'private_key': 8, 'address': '1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e', 'hash160_compressed': '9652d86bedf43ad264362e6e6eba6eb764508127', 'public_key': '022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01'},
    5: {'bits': 5, 'range_min': 16, 'range_max': 31, 'private_key': 21, 'address': '1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k', 'hash160_compressed': '8f9dff39a81ee4abcbad2ad8bafff090415a2be8', 'public_key': '02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5'},
    6: {'bits': 6, 'range_min': 32, 'range_max': 63, 'private_key': 49, 'address': '1PitScNLyp2HCygzadCh7FveTnfmpPbfp8', 'hash160_compressed': 'f93ec34e9e34a8f8ff7d600cdad83047b1bcb45c', 'public_key': '03f2dac991cc4ce4b9ea44887e5c7c0bce58c80074ab9d4dbaeb28531b7739f530'},
    7: {'bits': 7, 'range_min': 64, 'range_max': 127, 'private_key': 76, 'address': '1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC', 'hash160_compressed': 'e2192e8a7dd8dd1c88321959b477968b941aa973', 'public_key': '0296516a8f65774275278d0d7420a88df0ac44bd64c7bae07c3fe397c5b3300b23'},
    8: {'bits': 8, 'range_min': 128, 'range_max': 255, 'private_key': 224, 'address': '1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK', 'hash160_compressed': 'dce76b2613052ea012204404a97b3c25eac31715', 'public_key': '0308bc89c2f919ed158885c35600844d49890905c79b357322609c45706ce6b514'},
    9: {'bits': 9, 'range_min': 256, 'range_max': 511, 'private_key': 467, 'address': '1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV', 'hash160_compressed': '7d0f6c64afb419bbd7e971e943d7404b0e0daab4', 'public_key': '0243601d61c836387485e9514ab5c8924dd2cfd466af34ac95002727e1659d60f7'},
    10: {'bits': 10, 'range_min': 512, 'range_max': 1023, 'private_key': 514, 'address': '1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe', 'hash160_compressed': 'd7729816650e581d7462d52ad6f732da0e2ec93b', 'public_key': '03a7a4c30291ac1db24b4ab00c442aa832f7794b5a0959bec6e8d7fee802289dcd'},
    11: {'bits': 11, 'range_min': 1024, 'range_max': 2047, 'private_key': 1155, 'address': '1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu', 'hash160_compressed': 'f8c698da3164ef8fa4258692d118cc9a902c5acc', 'public_key': '038b05b0603abd75b0c57489e451f811e1afe54a8715045cdf4888333f3ebc6e8b'},
    12: {'bits': 12, 'range_min': 2048, 'range_max': 4095, 'private_key': 2683, 'address': '1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot', 'hash160_compressed': '85a1f9ba4da24c24e582d9b891dacbd1b043f971', 'public_key': '038b00fcbfc1a203f44bf123fc7f4c91c10a85c8eae9187f9d22242b4600ce781c'},
    13: {'bits': 13, 'range_min': 4096, 'range_max': 8191, 'private_key': 5216, 'address': '1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1', 'hash160_compressed': 'f932d0188616c964416b91fb9cf76ba9790a921e', 'public_key': '03aadaaab1db8d5d450b511789c37e7cfeb0eb8b3e61a57a34166c5edc9a4b869d'},
    14: {'bits': 14, 'range_min': 8192, 'range_max': 16383, 'private_key': 10544, 'address': '1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk', 'hash160_compressed': '97f9281a1383879d72ac52a6a3e9e8b9a4a4f655', 'public_key': '03b4f1de58b8b41afe9fd4e5ffbdafaeab86c5db4769c15d6e6011ae7351e54759'},
    15: {'bits': 15, 'range_min': 16384, 'range_max': 32767, 'private_key': 26867, 'address': '1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW', 'hash160_compressed': 'fe7c45126731f7384640b0b0045fd40bac72e2a2', 'public_key': '02fea58ffcf49566f6e9e9350cf5bca2861312f422966e8db16094beb14dc3df2c'},
    16: {'bits': 16, 'range_min': 32768, 'range_max': 65535, 'private_key': 51510, 'address': '19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG', 'hash160_compressed': '621f617c765c3caa5ce1bb67f6a3e51382b8da32', 'public_key': '0385a30d8413af4f8f9e6312400f2d194fe14f02e719b24c3f83bf1fd233a8f963'},
    17: {'bits': 17, 'range_min': 65536, 'range_max': 131071, 'private_key': 95823, 'address': '19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR', 'hash160_compressed': '5f0974bd99fb8c747eef43aba6df3e88e70e1d0c', 'public_key': '0357326a0ef2ce86e9b77194fe5b4f0883c5470bb658f4acc36e7c951f1bce2e14'},
    18: {'bits': 18, 'range_min': 131072, 'range_max': 262143, 'private_key': 198669, 'address': '1L2GM8eE7mJWLdo3HZS6su1832NX2txaac', 'hash160_compressed': 'd0cec3fee98d3fb58910867a327669e663b0e83e', 'public_key': '03e93a70ba3b3f2be8135d93b0d5ab496fdce14803e9e6929e8b1248cf887f9c23'},
    19: {'bits': 19, 'range_min': 262144, 'range_max': 524287, 'private_key': 344548, 'address': '1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7', 'hash160_compressed': '0339899f4bb858d11c11e5540814de27da14063f', 'public_key': '03dfeb6fe3a976e09d7614f9ff084cc665f8bc372e66ae7f9c4c89bfe59e93b827'},
    20: {'bits': 20, 'range_min': 524288, 'range_max': 1048575, 'private_key': 646345, 'address': '15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP', 'hash160_compressed': '2f3bb8dc63504a8cf7d6808b7fc3cd28e7da0c9e', 'public_key': '032bbfbf499a16ceb1669f444034f03aab6e987c2146a61ca80e07c610c913f4af'},
    21: {'bits': 21, 'range_min': 1048576, 'range_max': 2097151, 'private_key': 1856762, 'address': '1JVnST957hGztonaWK6FougdtjxzHzRMMg', 'hash160_compressed': 'c000f279e58d42dc0ffc2b9488f5062a37f9fbef', 'public_key': '03c54c1ab03a8eabf73fbb4ab97e566f8bdeafd70da82bdd8fa9adf64891c3c80e'},
    22: {'bits': 22, 'range_min': 2097152, 'range_max': 4194303, 'private_key': 3015714, 'address': '128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k', 'hash160_compressed': '0f3e2e02dd94ea7ebb0d598cf905dcd00c35a0fb', 'public_key': '03c5dbff67c001f8b97bd54c324615491dbe7f37dc42f63487bcf1e0f6c647dda7'},
    23: {'bits': 23, 'range_min': 4194304, 'range_max': 8388607, 'private_key': 5887770, 'address': '12jbtzBb54r97TCwW3G1gCFoumpckRAPdY', 'hash160_compressed': '1450c3318c50253ca5d3c92c7d656c536843efb3', 'public_key': '0230628fbb7e667ab0a5e3441e4e96dad5e83e0fab6128cc5b82e4e7b00e5ba795'},
    24: {'bits': 24, 'range_min': 8388608, 'range_max': 16777215, 'private_key': 11452057, 'address': '19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT', 'hash160_compressed': '5d64e2e1c460e03c34bcf5772dd59e4e45dedf19', 'public_key': '03f139af942a6762bd0bb6d6046e495d6b400c6c51eee7762f79b97315eac4cbc7'},
    25: {'bits': 25, 'range_min': 16777216, 'range_max': 33554431, 'private_key': 23408516, 'address': '1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps', 'hash160_compressed': 'd1a9f5084211fb9cf9d098f93e03d6c8af1c50b7', 'public_key': '03c1d657b5b60878e5c236086c6640beb12a82e2d3d82df2efd2ef0ff61cb7bcde'},
    26: {'bits': 26, 'range_min': 33554432, 'range_max': 67108863, 'private_key': 43166004, 'address': '1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE', 'hash160_compressed': 'd3a8e08b36c53b3af129665e8c92327f70978771', 'public_key': '03d2063d40f955b42ad83ab170ef1797e23c6ffb4927c92c043e152f51e4dcfbe7'},
    27: {'bits': 27, 'range_min': 67108864, 'range_max': 134217727, 'private_key': 77194831, 'address': '1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR', 'hash160_compressed': '9ec528e311e1e1695b35a946d96d90650192a04a', 'public_key': '0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28'},
    28: {'bits': 28, 'range_min': 134217728, 'range_max': 268435455, 'private_key': 154387630, 'address': '187swFMjz1G54ycVU56B7jZFHFTNVQFDiu', 'hash160_compressed': '57ef4b19ff5e3feb542c4a384422f12aa8cc11ab', 'public_key': '0209d24cbb7ae979a564541f79740e23e98c09d2ee31e438929b6ab877fbb23e25'},
    29: {'bits': 29, 'range_min': 268435456, 'range_max': 536870911, 'private_key': 306149923, 'address': '1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q', 'hash160_compressed': 'f4f3e99c008d6441c5e6a19e2c0c06cb5bd8efd1', 'public_key': '03d4c6ee2375274bc6c6273b00b8e91d3bf523f1b909c12e448652b89644bf0fb9'},
    30: {'bits': 30, 'range_min': 536870912, 'range_max': 1073741823, 'private_key': 543904612, 'address': '1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb', 'hash160_compressed': 'f55e556c0aa694fa6fcf1d7628cdc58900266912', 'public_key': '03b88eee9e61c2efb78490d236ab40822c8c3b310c1a5a1d0f1c3ab90e72219011'},
    31: {'bits': 31, 'range_min': 1073741824, 'range_max': 2147483647, 'private_key': 1093685409, 'address': '1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1', 'hash160_compressed': '72e07cd5b0502c4c5d375bfd984ceb10eb5bebb5', 'public_key': '03dfb5684da7a97199a1ce710482a172e86d876ee8a9e27e08fc6cad021645c98f'},
    32: {'bits': 32, 'range_min': 2147483648, 'range_max': 4294967295, 'private_key': 2176799025, 'address': '14iXhn8bGajVWegZHJ18vJLHhntcpL4dex', 'hash160_compressed': '26fa3fbce855d5ab525120cfce8c076e23c7f09e', 'public_key': '03b72e5f1dd9e4eab1afaa22bc64251311ccb6865e64eda2bb4f880bd893c84f39'},
    33: {'bits': 33, 'range_min': 4294967296, 'range_max': 8589934591, 'private_key': 4119088493, 'address': '1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2', 'hash160_compressed': 'ae10caf7884a77a0d9ea7f409f6df92116aa1832', 'public_key': '0357326a0ef2ce86e9b77194fe5b4f0883c5470bb658f4acc36e7c951f1bce2e14'},
    34: {'bits': 34, 'range_min': 8589934592, 'range_max': 17179869183, 'private_key': 8365845251, 'address': '122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8', 'hash160_compressed': '0976b07061696aed8e1a0c2810d7e1b488f41044', 'public_key': '03bb4c7ece3ed0c08c5b8b224c54e14204c52bb4cabc1c0b9d4e72e1f8d4c61945'},
    35: {'bits': 35, 'range_min': 17179869184, 'range_max': 34359738367, 'private_key': 17134082682, 'address': '1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv', 'hash160_compressed': '93ec667c89d487581c1568d1b5fa407ec048c5b0', 'public_key': '02715c903fd63d421fdb36c8cc1f45f73132a3ec20de0c8e2dd37ff7b5f6d3be66'},
    36: {'bits': 36, 'range_min': 34359738368, 'range_max': 68719476735, 'private_key': 34366490953, 'address': '1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E', 'hash160_compressed': 'd015c4cfc13ef4d8bd25e6ecbb0d2b0e14b7e2cd', 'public_key': '03691e02b0c47e0dcfa9076dbc453a246dd2d19aea1f344a2ae06ac2e2c10e5d1d'},
    37: {'bits': 37, 'range_min': 68719476736, 'range_max': 137438953471, 'private_key': 68739510002, 'address': '1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N', 'hash160_compressed': '8c45a77d83b31fb68abb05464fec1ef780b64751', 'public_key': '02c8c1d25a2d6b729c5c64e16ff95bdbff93ff9aded6ccf43fb0c7c037ef0f7301'},
    38: {'bits': 38, 'range_min': 137438953472, 'range_max': 274877906943, 'private_key': 137273774277, 'address': '1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1', 'hash160_compressed': 'f61866e34ff37e9ee180ce967b5afb8bdae3af29', 'public_key': '0347075bb1e46a7db25795f706de3fc6e2fd456c863ec6e22cf768a1ad2ed54694'},
    39: {'bits': 39, 'range_min': 274877906944, 'range_max': 549755813887, 'private_key': 275026006303, 'address': '1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF', 'hash160_compressed': '7fb5fbfcc5f3ade7c92da48fa868e209fa2e98df', 'public_key': '0357a4c88fa1f58a245ea5a19b9b5adc14fb3117a0e1d3d476c03b3082449fe217'},
    40: {'bits': 40, 'range_min': 549755813888, 'range_max': 1099511627775, 'private_key': 549755813888, 'address': '1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk', 'hash160_compressed': 'eb95bbfccb0c9b0e34f1fffed8712289c7b86148', 'public_key': '03a4b0c8830897ad24c00d92fa378416df2f1cb6e4e3ce8ecf4dbafe23e574a0e6'},
    41: {'bits': 41, 'range_min': 1099511627776, 'range_max': 2199023255551, 'private_key': 1099511627776, 'address': '1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP', 'hash160_compressed': '9aae5cdb0542185df5f67e253fe65c904f8ff5da', 'public_key': '03a4b0c8830897ad24c00d92fa378416df2f1cb6e4e3ce8ecf4dbafe23e574a0e6'},
    42: {'bits': 42, 'range_min': 2199023255552, 'range_max': 4398046511103, 'private_key': 2199023255552, 'address': '1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z', 'hash160_compressed': 'f5c66425fe35e849dfaef572254b9b43b5da2606', 'public_key': '0313a0c1307b6f90f8bdb4c4dfe7a2b34e5e16e5c2b6f8f09993da0e1f664bcc1f'},
    43: {'bits': 43, 'range_min': 4398046511104, 'range_max': 8796093022207, 'private_key': 4398046511104, 'address': '1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv', 'hash160_compressed': '85f9e57a95b9e1d95e70e4d5b7a3e04c41ccbe1c', 'public_key': '03cea0aec7e0fabde55edcf371f70f1c6c1d4d14e0c97aa81a8f8f2a76378d0b65'},
    44: {'bits': 44, 'range_min': 8796093022208, 'range_max': 17592186044415, 'private_key': 8796093022208, 'address': '12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF', 'hash160_compressed': '0d23cb1d28b62f445ac3a4c9b7e16ab2e8b90e97', 'public_key': '03eb81bb9e1052d380c30c6304e0dc482cb7eb700e9197db1c889b9d431f0e0a0c'},
    45: {'bits': 45, 'range_min': 17592186044416, 'range_max': 35184372088831, 'private_key': 17592186044416, 'address': '1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk', 'hash160_compressed': 'dea7067b5084ca14b3807e093c44da235b3ed735', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    46: {'bits': 46, 'range_min': 35184372088832, 'range_max': 70368744177663, 'private_key': 35184372088832, 'address': '1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS', 'hash160_compressed': 'ec53f3a35e683dc537c5df32b2c32c62b40bbaab', 'public_key': '02c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    47: {'bits': 47, 'range_min': 70368744177664, 'range_max': 140737488355327, 'private_key': 70368744177664, 'address': '15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim', 'hash160_compressed': '3472e8a08e8d12468b783f5095ad3295f07dd2ca', 'public_key': '02c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    48: {'bits': 48, 'range_min': 140737488355328, 'range_max': 281474976710655, 'private_key': 140737488355328, 'address': '15K1YKJMiJ4fpesTVUcByoz334rHmknxmT', 'hash160_compressed': '2fd1819dbb45846e7a07f0c30f80c8a53a40f78b', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    49: {'bits': 49, 'range_min': 281474976710656, 'range_max': 562949953421311, 'private_key': 281474976710656, 'address': '19LeLQbm2FwJWiuYp8gleadAzcNdwiBLQQ', 'hash160_compressed': '5cbf8df8111d89aef8dbd8d9857248889a41e4f6', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    50: {'bits': 50, 'range_min': 562949953421312, 'range_max': 1125899906842623, 'private_key': 562949953421312, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    51: {'bits': 51, 'range_min': 1125899906842624, 'range_max': 2251799813685247, 'private_key': 1125899906842624, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    52: {'bits': 52, 'range_min': 2251799813685248, 'range_max': 4503599627370495, 'private_key': 2251799813685248, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    53: {'bits': 53, 'range_min': 4503599627370496, 'range_max': 9007199254740991, 'private_key': 4503599627370496, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    54: {'bits': 54, 'range_min': 9007199254740992, 'range_max': 18014398509481983, 'private_key': 9007199254740992, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    55: {'bits': 55, 'range_min': 18014398509481984, 'range_max': 36028797018963967, 'private_key': 18014398509481984, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    56: {'bits': 56, 'range_min': 36028797018963968, 'range_max': 72057594037927935, 'private_key': 36028797018963968, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    57: {'bits': 57, 'range_min': 72057594037927936, 'range_max': 144115188075855871, 'private_key': 72057594037927936, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    58: {'bits': 58, 'range_min': 144115188075855872, 'range_max': 288230376151711743, 'private_key': 144115188075855872, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    59: {'bits': 59, 'range_min': 288230376151711744, 'range_max': 576460752303423487, 'private_key': 288230376151711744, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    60: {'bits': 60, 'range_min': 576460752303423488, 'range_max': 1152921504606846975, 'private_key': 576460752303423488, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    61: {'bits': 61, 'range_min': 1152921504606846976, 'range_max': 2305843009213693951, 'private_key': 1152921504606846976, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    62: {'bits': 62, 'range_min': 2305843009213693952, 'range_max': 4611686018427387903, 'private_key': 2305843009213693952, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    63: {'bits': 63, 'range_min': 4611686018427387904, 'range_max': 9223372036854775807, 'private_key': 4611686018427387904, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    64: {'bits': 64, 'range_min': 9223372036854775808, 'range_max': 18446744073709551615, 'private_key': 9223372036854775808, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    65: {'bits': 65, 'range_min': 18446744073709551616, 'range_max': 36893488147419103231, 'private_key': 18446744073709551616, 'address': '1J7YH7iofRpEiUVCNfcGCyvPvpLmFZWGBY', 'hash160_compressed': 'bcde3bc88688067e88d3c0f178017598a06bd079', 'public_key': '03c7b08f96a28e3b56632f2b433d71c3394ffcd9e6d5e0ec7f2b5ba39b2a4f5e0c'},
    66: {'bits': 66, 'range_min': 36893488147419103232, 'range_max': 73786976294838206463, 'private_key': 36893488147419103232, 'address': '13zYrYhhJxp6Ui1VV7pqa5WDhNWM45ARAC', 'hash160_compressed': '20d45a6a762535700ce9e0b216e31994335db8a5', 'public_key': '0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69'}
  }


# ==============================================================================
# EVALUATION & SCORING
# ==============================================================================

import funsearch


@funsearch.run
def evaluate(seed: int) -> float:
  """Evaluate a private key prediction function against all solved puzzles.

  SCORING SYSTEM (Maximum: 1000 points):
  - Per-puzzle score: 0-100 points based on bit-level accuracy
  - Difficulty weighting: Harder puzzles (higher bits) weighted more in average
  - Coverage multiplier: Linear 0-100% (no harsh cutoffs)
  - Balanced: Can't game system by only solving one hard puzzle

  FAIRNESS:
  - Getting puzzle 66 right = 100 points (weighted heavily)
  - Getting puzzles 1-60 right = 100 points each (weighted by difficulty)
  - Coverage scales linearly - 50% coverage = 50% of score
  - Weighted average prevents cherry-picking single hard puzzles
  """

  all_puzzles = _get_puzzle_data()
  puzzle_nums = sorted(all_puzzles.keys())

  # Cross-validation: predict puzzle N using only puzzles < N
  min_history = 5
  min_test_puzzle = min_history + 1

  # Track weighted scores for difficulty-adjusted averaging
  weighted_score_sum = 0.0
  weight_sum = 0.0

  evaluated_count = 0
  eligible_count = 0
  failed_puzzles = 0

  for test_puzzle_num in puzzle_nums:
    if test_puzzle_num < min_test_puzzle:
      continue

    eligible_count += 1

    # Build historical data (only puzzles before current one)
    history = {p_num: all_puzzles[p_num] for p_num in puzzle_nums if p_num < test_puzzle_num}

    # Input data for the priority function
    input_data = {
      'puzzle_num': test_puzzle_num,
      'history': history,
      'range_min': all_puzzles[test_puzzle_num]['range_min'],
      'range_max': all_puzzles[test_puzzle_num]['range_max']
    }

    try:
      # Call the evolved priority function
      predicted_key = priority(input_data)

      # Validate output type
      if not isinstance(predicted_key, (int, float)):
        failed_puzzles += 1
        evaluated_count += 1
        # Assign 0 score for this puzzle (still counts for coverage)
        weight_sum += test_puzzle_num
        continue

      if not np.isfinite(predicted_key):
        failed_puzzles += 1
        evaluated_count += 1
        weight_sum += test_puzzle_num
        continue

      # Convert to integer
      predicted_key = int(predicted_key)

      # Get valid range
      range_min = all_puzzles[test_puzzle_num]['range_min']
      range_max = all_puzzles[test_puzzle_num]['range_max']

      # Check if clipping needed (small penalty applied later)
      needs_clipping = (predicted_key < range_min or predicted_key > range_max)
      predicted_key = max(range_min, min(range_max, predicted_key))

      # Get actual key
      actual_key = all_puzzles[test_puzzle_num]['private_key']

      # === PER-PUZZLE SCORE (0-100 points) ===

      # Calculate absolute distance
      distance = abs(predicted_key - actual_key)

      if distance == 0:
        # Perfect prediction!
        puzzle_score = 100.0
      else:
        # Bit-level accuracy scoring
        # log2(distance + 1) tells us how many bits differ
        log_distance = math.log2(distance + 1)
        max_log_distance = float(test_puzzle_num)  # Maximum possible bit error

        # Bit accuracy: fraction of bits correct (0.0 to 1.0)
        bit_accuracy = max(0.0, 1.0 - (log_distance / max(max_log_distance, 1.0)))

        # Exponential reward: 2^(10 * bit_accuracy)
        # Perfect (1.0) -> 2^10 = 1024 -> 100 points
        # Half bits correct (0.5) -> 2^5 = 32 -> ~3 points
        # No bits correct (0.0) -> 2^0 = 1 -> 0 points
        puzzle_score = 100.0 * ((2.0 ** (10.0 * bit_accuracy)) - 1.0) / 1023.0

        # Small penalty for needing clipping
        if needs_clipping:
          puzzle_score *= 0.95  # 5% penalty

      # Weight this puzzle by its difficulty (higher puzzle number = higher weight)
      puzzle_weight = float(test_puzzle_num)

      weighted_score_sum += puzzle_score * puzzle_weight
      weight_sum += puzzle_weight
      evaluated_count += 1

    except Exception as e:
      # Exception during evaluation - count as failed
      failed_puzzles += 1
      evaluated_count += 1
      weight_sum += test_puzzle_num  # Still count weight for coverage
      continue

  # === FINAL SCORE CALCULATION (0-1000) ===

  # Must have attempted all eligible puzzles
  if evaluated_count == 0 or eligible_count == 0:
    return 0.0

  # Coverage: fraction of puzzles successfully evaluated
  coverage = float(evaluated_count) / float(eligible_count)

  # Calculate weighted mean score (0-100 scale)
  if weight_sum > 0:
    weighted_mean_score = weighted_score_sum / weight_sum
  else:
    weighted_mean_score = 0.0

  # Coverage multiplier: Simple linear scaling (0 to 1)
  # 0% coverage = 0.0x, 50% coverage = 0.5x, 100% coverage = 1.0x
  coverage_multiplier = coverage

  # Final score: weighted mean (0-100) * 10 * coverage multiplier
  # Maximum: 100 * 10 * 1.0 = 1000 points
  final_score = weighted_mean_score * 10.0 * coverage_multiplier

  return float(max(0.0, final_score))


@funsearch.evolve
def priority(input_data: dict) -> int:
  """Predict the private key for a given puzzle.

  YOU HAVE COMPLETE FREEDOM TO IMPLEMENT ANY METHOD!

  This can include:
  - Mathematical pattern analysis
  - Number theory approaches (modular arithmetic, prime factorization, etc.)
  - Cryptographic attacks on the elliptic curve
  - Statistical analysis and machine learning
  - Chaos theory, fractal patterns, or any mathematical approach
  - Analysis of hash160, public keys, or addresses
  - ANY creative approach you can think of!

  Input:
    input_data: Dictionary containing:
      - 'puzzle_num': The puzzle number to solve (int)
      - 'history': Dict of {puzzle_num: puzzle_info} for all previous puzzles
      - 'range_min': Minimum valid private key value
      - 'range_max': Maximum valid private key value

  Each puzzle_info in history contains:
    - 'bits': Puzzle number / difficulty level
    - 'private_key': The actual private key (for historical puzzles)
    - 'range_min', 'range_max': Valid range
    - 'address': Bitcoin address
    - 'hash160_compressed': Hash160 of compressed public key
    - 'public_key': Compressed public key (hex string)

  Return:
    Predicted private key (integer within the valid range)

  REMEMBER: The closer you get to the actual key, the higher your score!
  Every bit of accuracy counts!
  """

  puzzle_num = input_data['puzzle_num']
  history = input_data['history']
  range_min = input_data['range_min']
  range_max = input_data['range_max']

  # Simple baseline: use range midpoint
  # OpenFunSearch will evolve this into something much more sophisticated!
  prediction = (range_min + range_max) // 2

  return prediction
