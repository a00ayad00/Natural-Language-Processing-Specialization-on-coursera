# import textwrap
import itertools
import numpy as np

import trax
from trax import layers as tl

# from trax.supervised import decoding

def testing_rnd():
    def dummy_generator():
        vals = np.linspace(0, 1, 10)
        cyclic_vals = itertools.cycle(vals)
        for _ in range(100):
            yield next(cyclic_vals)

    dumr = itertools.cycle(dummy_generator())

    def dummy_randomizer():
        return next(dumr)

    return dummy_randomizer


def test_tokenize_and_mask(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "mocked randomizer input 0",
            "input": {
                "text": b"Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.",
                "randomizer": testing_rnd(),
            },
            "expected": (
                [
                    31999,
                    15068,
                    4501,
                    3,
                    12297,
                    3399,
                    16,
                    5964,
                    7115,
                    31998,
                    531,
                    25,
                    241,
                    12,
                    129,
                    394,
                    44,
                    492,
                    31997,
                    58,
                    148,
                    56,
                    43,
                    8,
                    1004,
                    6,
                    474,
                    31996,
                    39,
                    4793,
                    230,
                    5,
                    2721,
                    6,
                    1600,
                    1630,
                    31995,
                    1150,
                    4501,
                    15068,
                    16127,
                    6,
                    9137,
                    2659,
                    5595,
                    31994,
                    782,
                    3624,
                    14627,
                    15,
                    12612,
                    277,
                    5,
                    216,
                    31993,
                    2119,
                    3,
                    9,
                    19529,
                    593,
                    853,
                    21,
                    921,
                    31992,
                    12,
                    129,
                    394,
                    28,
                    70,
                    17712,
                    1098,
                    5,
                    31991,
                    3884,
                    25,
                    762,
                    25,
                    174,
                    12,
                    214,
                    12,
                    31990,
                    3,
                    9,
                    3,
                    23405,
                    4547,
                    15068,
                    2259,
                    6,
                    31989,
                    6,
                    5459,
                    6,
                    13618,
                    7,
                    6,
                    3604,
                    1801,
                    31988,
                    6,
                    303,
                    24190,
                    11,
                    1472,
                    251,
                    5,
                    37,
                    31987,
                    36,
                    16,
                    8,
                    853,
                    19,
                    25264,
                    399,
                    568,
                    31986,
                    21,
                    21380,
                    7,
                    34,
                    19,
                    339,
                    5,
                    15746,
                    31985,
                    8,
                    583,
                    56,
                    36,
                    893,
                    3,
                    9,
                    3,
                    31984,
                    9486,
                    42,
                    3,
                    9,
                    1409,
                    29,
                    11,
                    25,
                    31983,
                    12246,
                    5977,
                    13,
                    284,
                    3604,
                    24,
                    19,
                    2657,
                    31982,
                ],
                [
                    31999,
                    12847,
                    277,
                    31998,
                    9,
                    55,
                    31997,
                    3326,
                    15068,
                    31996,
                    48,
                    30,
                    31995,
                    727,
                    1715,
                    31994,
                    45,
                    301,
                    31993,
                    56,
                    36,
                    31992,
                    113,
                    2746,
                    31991,
                    216,
                    56,
                    31990,
                    5978,
                    16,
                    31989,
                    379,
                    2097,
                    31988,
                    11,
                    27856,
                    31987,
                    583,
                    12,
                    31986,
                    6,
                    11,
                    31985,
                    26,
                    16,
                    31984,
                    17,
                    18,
                    31983,
                    56,
                    36,
                    31982,
                    5,
                ],
            ),
            "error_message": "Incorrect output tuple",
        },
        {
            "name": "numpy randomizer input 0",
            "input": {
                "text": b"Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.",
            },
            "expected": (
                [
                    12847,
                    277,
                    15068,
                    4501,
                    3,
                    12297,
                    3399,
                    16,
                    5964,
                    7115,
                    9,
                    55,
                    31999,
                    241,
                    12,
                    129,
                    394,
                    44,
                    492,
                    3326,
                    15068,
                    58,
                    148,
                    56,
                    43,
                    8,
                    1004,
                    6,
                    474,
                    31998,
                    30,
                    39,
                    31997,
                    230,
                    5,
                    2721,
                    31996,
                    1600,
                    1630,
                    727,
                    1715,
                    1150,
                    31995,
                    15068,
                    16127,
                    6,
                    9137,
                    2659,
                    5595,
                    45,
                    301,
                    782,
                    3624,
                    14627,
                    15,
                    12612,
                    277,
                    5,
                    216,
                    56,
                    36,
                    2119,
                    3,
                    31994,
                    19529,
                    593,
                    853,
                    21,
                    921,
                    113,
                    2746,
                    12,
                    129,
                    394,
                    28,
                    70,
                    17712,
                    1098,
                    5,
                    216,
                    56,
                    3884,
                    25,
                    762,
                    25,
                    174,
                    12,
                    214,
                    12,
                    5978,
                    16,
                    3,
                    9,
                    31993,
                    23405,
                    4547,
                    15068,
                    2259,
                    31992,
                    379,
                    2097,
                    6,
                    31991,
                    13618,
                    7,
                    6,
                    3604,
                    1801,
                    11,
                    27856,
                    6,
                    303,
                    31990,
                    11,
                    1472,
                    251,
                    5,
                    37,
                    583,
                    12,
                    31989,
                    16,
                    8,
                    853,
                    19,
                    25264,
                    31988,
                    568,
                    6,
                    31987,
                    21,
                    21380,
                    7,
                    34,
                    31986,
                    339,
                    5,
                    15746,
                    26,
                    16,
                    8,
                    583,
                    56,
                    36,
                    893,
                    3,
                    9,
                    3,
                    17,
                    18,
                    9486,
                    31985,
                    3,
                    9,
                    1409,
                    29,
                    31984,
                    25,
                    56,
                    31983,
                    12246,
                    5977,
                    13,
                    284,
                    3604,
                    24,
                    19,
                    2657,
                    5,
                ],
                [
                    31999,
                    531,
                    25,
                    31998,
                    48,
                    31997,
                    4793,
                    31996,
                    6,
                    31995,
                    4501,
                    31994,
                    9,
                    31993,
                    3,
                    31992,
                    6,
                    31991,
                    5459,
                    6,
                    31990,
                    24190,
                    31989,
                    36,
                    31988,
                    399,
                    31987,
                    11,
                    31986,
                    19,
                    31985,
                    42,
                    31984,
                    11,
                    31983,
                    36,
                ],
            ),
            "error_message": "Incorrect output tuple",
        },
        {"name": "mocked randomizer input 2",
            "input": {
                "text": b'Foil plaid lycra and spandex shortall with metallic slinky insets. Attached metallic elastic belt with O-ring. Headband included. Great hip hop or jazz dance costume. Made in the USA.',
                "randomizer": testing_rnd(),
            },
            "expected": ([31999, 30772, 3, 120, 2935, 11, 8438, 26, 994, 31998, 28, 18813, 3, 7, 4907, 63, 16, 2244, 31997, 28416, 15, 26, 18813, 15855, 6782, 28, 411, 31996, 5, 3642, 3348, 1285, 5, 1651, 5436, 13652, 31995, 2595, 11594, 5, 6465, 16, 8, 2312, 5],
                         [31999, 4452, 173, 31998, 710, 1748, 31997, 7, 5, 31996, 18, 1007, 31995, 42, 9948]
                        ),
           "error_message": "Incorrect output tuple",
        },
        {"name": "numpy randomizer input 2",
            "input": {
                "text": b'Foil plaid lycra and spandex shortall with metallic slinky insets. Attached metallic elastic belt with O-ring. Headband included. Great hip hop or jazz dance costume. Made in the USA.',
            },
            "expected": ([4452, 173, 30772, 3, 120, 2935, 11, 8438, 26, 994, 710, 1748, 31999, 3, 7, 4907, 63, 16, 2244, 7, 5, 28416, 15, 26, 18813, 15855, 6782, 28, 411, 31998, 1007, 5, 31997, 3348, 1285, 5, 31996, 5436, 13652, 42, 9948, 2595, 31995, 5, 6465, 16, 8, 2312, 5],
                         [31999, 28, 18813, 31998, 18, 31997, 3642, 31996, 1651, 31995, 11594]
                        ),
             "error_message": "Incorrect output tuple",
        }
    ]

    for test_case in test_cases:
        if "randomizer" not in test_case["input"]:
            np.random.seed(12345)

        output = target(**test_case["input"])

        try:
            assert len(output) == len(test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]),
                    "got": len(output),
                }
            )
            print(
                f"Output has incorrect size. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert output[0] == test_case["expected"][0]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][0],
                    "got": output[0],
                }
            )
            print(
                f"Element with index 0 in output tuple is incorrect. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert output[1] == test_case["expected"][1]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][1],
                    "got": output[1],
                }
            )
            print(
                f"Element with index 1 in output tuple is incorrect. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_FeedForwardBlock(target):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [{'name': 'default_feedforwardblock'
                   , 'input': {'d_model': 512, 'd_ff': 2048, 'dropout': 0.8, 'dropout_shared_axes': 0, 'mode': "train", 'activation': tl.Relu}
                   , 'expected': {'expected_str': f'[LayerNorm, Dense_2048, Serial[\n  Relu\n], Dropout, Dense_512, Dropout]',
                                'expected_types': [
                                                    trax.layers.normalization.LayerNorm,
                                                    trax.layers.core.Dense,
                                                    trax.layers.combinators.Serial,
                                                    trax.layers.core.Dropout,
                                                    trax.layers.core.Dense,
                                                    trax.layers.core.Dropout,
                                                  ]

                               }
                  },
                  {'name': 'check_feedforwardblock'
                   ,'input': {'d_model': 16, 'd_ff': 64, 'dropout': 0.1, 'dropout_shared_axes': 0, 'mode': "train", 'activation': tl.Relu}
                   ,'expected': {'expected_str': f'[LayerNorm, Dense_64, Serial[\n  Relu\n], Dropout, Dense_16, Dropout]',
                                'expected_types': [trax.layers.normalization.LayerNorm,
                                                     trax.layers.core.Dense,
                                                     trax.layers.combinators.Serial,
                                                     trax.layers.core.Dropout,
                                                     trax.layers.core.Dense,
                                                     trax.layers.core.Dropout]

                               }
                  }
                 ]

    for test_case in test_cases:
    
        output = target(**test_case['input'])    

        try:
            assert str(output).replace(" ", "") == test_case['expected']['expected_str'].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "str_check", "expected": test_case['expected']['expected_str'], "got": str(output),}
            )
            print(
                f"Wrong model. \nProposed:\n {failed_cases[-1].get('got')}. \nExpected:\n {failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(output, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "type_check", "expected": list, "got": type(output),}
            )
            print(
                f"FeedForwardBlock does not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert len(output) == len(test_case['expected']['expected_types'])
            successful_cases += 1

            test_func = lambda x: list((map(type, x)))
            output_sublayers_type = test_func(output)

            try:
                for output_elem, expected_elem in zip(output, test_case['expected']['expected_types']):
                    assert isinstance(output_elem, expected_elem)
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "sublayers_type_check",
                        "expected": test_case['expected']['expected_types'],
                        "got": output_sublayers_type,
                    }
                )
                print(
                    f"Sublayer elements in FeedForwardBlock do not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
                )

        except:
            failed_cases.append(
                {"name": "n_elements_check", "expected": len(test_case['expected']['expected_types']), "got": len(output),}
            )
            print(
                f"Number of elements in FeedForwardBlock is not correct. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_EncoderBlock(target):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [{'name': 'default_encoderblock'
                   , 'input': {'d_model': 512, 'd_ff': 2048, 'n_heads': 6, 'dropout': 0.8, 'dropout_shared_axes': 0, 'mode': "train", 'ff_activation': tl.Relu}
                   , 'expected': {'expected_str': '[Serial_in2_out2[\n  Branch_in2_out3[\n    None\n    Serial_in2_out2[\n      LayerNorm\n      Serial_in2_out2[\n        _in2_out2\n        Serial_in2_out2[\n          Select[0,0,0]_out3\n          Serial_in4_out2[\n            _in4_out4\n            Serial_in4_out2[\n              Parallel_in3_out3[\n                Dense_512\n                Dense_512\n                Dense_512\n              ]\n              PureAttention_in4_out2\n              Dense_512\n            ]\n            _in2_out2\n          ]\n        ]\n        _in2_out2\n      ]\n      Dropout\n    ]\n  ]\n  Add_in2\n], Serial[\n  Branch_out2[\n    None\n    Serial[\n      LayerNorm\n      Dense_2048\n      Serial[\n        Relu\n      ]\n      Dropout\n      Dense_512\n      Dropout\n    ]\n  ]\n  Add_in2\n]]', 
                                  'expected_types': (trax.layers.combinators.Serial,
                                                        [trax.layers.combinators.Serial, trax.layers.base.PureLayer],
                                                    )
                                 }
                  },
                  {'name': 'check_encoderblock'
                   ,'input': {'d_model': 16, 'd_ff': 64, 'n_heads': 2, 'dropout': 0.1, 'dropout_shared_axes': 0, 'mode': "train", 'ff_activation': tl.Relu}
                   ,'expected': {'expected_str': '[Serial_in2_out2[\n  Branch_in2_out3[\n    None\n    Serial_in2_out2[\n      LayerNorm\n      Serial_in2_out2[\n        _in2_out2\n        Serial_in2_out2[\n          Select[0,0,0]_out3\n          Serial_in4_out2[\n            _in4_out4\n            Serial_in4_out2[\n              Parallel_in3_out3[\n                Dense_16\n                Dense_16\n                Dense_16\n              ]\n              PureAttention_in4_out2\n              Dense_16\n            ]\n            _in2_out2\n          ]\n        ]\n        _in2_out2\n      ]\n      Dropout\n    ]\n  ]\n  Add_in2\n], Serial[\n  Branch_out2[\n    None\n    Serial[\n      LayerNorm\n      Dense_64\n      Serial[\n        Relu\n      ]\n      Dropout\n      Dense_16\n      Dropout\n    ]\n  ]\n  Add_in2\n]]'
                                , 'expected_types': (trax.layers.combinators.Serial,
                                                        [trax.layers.combinators.Serial, trax.layers.base.PureLayer],
                                                    )

                               }
                  }
                 ]

    for test_case in test_cases:
        output = target(**test_case['input'])
    
        try:
            assert str(output).replace(" ", "") == test_case['expected']['expected_str'].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "str_check", "expected": test_case['expected']['expected_str'], "got": str(output),}
            )
            print(
                f"Wrong model. \nProposed:\n {failed_cases[-1].get('got')}. \n\nExpected:\n {failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(output, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "type_check", "expected": list, "got": type(output),}
            )
            print(
                f"EncoderBlock does not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert len(output) == 2
            successful_cases += 1
            
            test_func = lambda x: (type(x), list(map(type, x.sublayers)))

            try:
                assert test_func(output[0]) == test_case['expected']['expected_types']
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "model_sublayers0_type_check",
                        "expected": test_case['expected']['expected_types'],
                        "got": test_func(output[0]),
                    }
                )
                print(
                    f"EncoderBlock list element 0 has sublayers with incorrect type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
                )

            try:
                assert test_func(output[1]) == test_case['expected']['expected_types']
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "model_sublayers1_type_check",
                        "expected": test_case['expected']['expected_types'],
                        "got": test_func(output[1]),
                    }
                )
                print(
                    f"EncoderBlock list element 1 has sublayers with incorrect type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
                )

        except:
            failed_cases.append(
                {"name": "n_elements_check", "expected": len(test_case['expected']['expected_types']), "got": len(output),}
            )
            print(
                f"Number of elements in EncoderBlock is not correct. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_TransformerEncoder(target):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [{'name': 'default_TransformerEncoder'
                   , 'input': {'vocab_size':32000, 'n_classes':10, 'd_model':512, 'd_ff':2048, 'n_layers':1
                               , 'n_heads':8, 'dropout':0.1, 'dropout_shared_axes':None, 'max_len':2048
                               , 'mode':'train', 'ff_activation':tl.Relu}
                   , 'expected': {'expected_str': '''Serial[\n  Branch_out2[\n    [Embedding_32000_512, Dropout, PositionalEncoding]\n    Serial[\n      PaddingMask(0)\n    ]\n  ]\n  Serial_in2_out2[\n    Branch_in2_out3[\n      None\n      Serial_in2_out2[\n        LayerNorm\n        Serial_in2_out2[\n          _in2_out2\n          Serial_in2_out2[\n            Select[0,0,0]_out3\n            Serial_in4_out2[\n              _in4_out4\n              Serial_in4_out2[\n                Parallel_in3_out3[\n                  Dense_512\n                  Dense_512\n                  Dense_512\n                ]\n                PureAttention_in4_out2\n                Dense_512\n              ]\n              _in2_out2\n            ]\n          ]\n          _in2_out2\n        ]\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  Serial[\n    Branch_out2[\n      None\n      Serial[\n        LayerNorm\n        Dense_2048\n        Serial[\n          Relu\n        ]\n        Dropout\n        Dense_512\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  Select[0]_in2\n  LayerNorm\n  Mean\n  Dense_10\n  LogSoftmax\n]'''
                                  , 'expected_types': (
                                                        trax.layers.combinators.Serial,
                                                        [
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.base.PureLayer,
                                                            trax.layers.normalization.LayerNorm,
                                                            trax.layers.base.PureLayer,
                                                            trax.layers.core.Dense,
                                                            trax.layers.base.PureLayer,
                                                        ],
                                                    )
                                 }
                  },
                  {'name': 'check_TransformerEncoder'
                   ,'input': {'vocab_size':100, 'n_classes':2, 'd_model':16, 'd_ff':32
                              , 'n_layers':1, 'n_heads':2, 'dropout':0.05, 'dropout_shared_axes':None
                              , 'max_len':256, 'mode':'train', 'ff_activation':tl.Relu}
                   ,'expected': {'expected_str': '''Serial[\n  Branch_out2[\n    [Embedding_100_16, Dropout, PositionalEncoding]\n    Serial[\n      PaddingMask(0)\n    ]\n  ]\n  Serial_in2_out2[\n    Branch_in2_out3[\n      None\n      Serial_in2_out2[\n        LayerNorm\n        Serial_in2_out2[\n          _in2_out2\n          Serial_in2_out2[\n            Select[0,0,0]_out3\n            Serial_in4_out2[\n              _in4_out4\n              Serial_in4_out2[\n                Parallel_in3_out3[\n                  Dense_16\n                  Dense_16\n                  Dense_16\n                ]\n                PureAttention_in4_out2\n                Dense_16\n              ]\n              _in2_out2\n            ]\n          ]\n          _in2_out2\n        ]\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  Serial[\n    Branch_out2[\n      None\n      Serial[\n        LayerNorm\n        Dense_32\n        Serial[\n          Relu\n        ]\n        Dropout\n        Dense_16\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  Select[0]_in2\n  LayerNorm\n  Mean\n  Dense_2\n  LogSoftmax\n]'''
                                , 'expected_types': (trax.layers.combinators.Serial,
                                                     [trax.layers.combinators.Serial,
                                                      trax.layers.combinators.Serial,
                                                      trax.layers.combinators.Serial,
                                                      trax.layers.base.PureLayer,
                                                      trax.layers.normalization.LayerNorm,
                                                      trax.layers.base.PureLayer,
                                                      trax.layers.core.Dense,
                                                      trax.layers.base.PureLayer])

                               }
                  }
                 ]


    for test_case in test_cases:
        output = target(**test_case['input'])

        try:
            assert str(output).replace(" ", "") == test_case['expected']['expected_str'].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "str_check", "expected": test_case['expected']['expected_str'], "got": str(output),}
            )
            print(
                f"Wrong model. \nProposed:\n {failed_cases[-1].get('got')}. \n\nExpected:\n {failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(output, trax.layers.combinators.Serial)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "type_check",
                    "expected": trax.layers.combinators.Serial,
                    "got": type(output),
                }
            )
            print(
                f"TransformerEncoder does not have the correct type.\nExpected:\n{failed_cases[-1].get('expected')}.\nGot:\n{failed_cases[-1].get('got')}."
            )

        try:
            assert len(output.sublayers) == len(test_case['expected']['expected_types'][1])
            successful_cases += 1

            test_func = lambda x: (type(x), list(map(type, x.sublayers)))

            try:
                for i in range(len(output.sublayers)):
                    assert isinstance(output.sublayers[i], test_case['expected']['expected_types'][1][i])
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "model_sublayers_type_check",
                        "expected": test_case['expected']['expected_types'][1],
                        "got": test_func(output),
                    }
                )
                print(
                    f"TransformerEncoder has sublayers with incorrect type.\nExpected:\n{failed_cases[-1].get('expected')}.\nGot:\n{failed_cases[-1].get('got')}."
                )

        except:
            failed_cases.append(
                {"name": "n_elements_check", "expected": len(test_case['expected']['expected_types'][1]), "got": len(output.sublayers),}
            )
            print(
                f"Number of sublayers in TransformerEncoder is not correct. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
