import jax
import trax
import pickle
import numpy as np
from trax import layers as tl
from trax.fastmath import numpy as fastnp
#from trax.supervised import training

# UNIT TEST for UNQ_C1
def test_get_conversation(target):

    data = {'file1.json': {'log':[{'text': 'hi'},
                                  {'text': 'hello'},
                                  {'text': 'nice'}]},
            'file2.json':{'log':[{'text': 'a b'}, 
                                 {'text': ''}, 
                                 {'text': 'good '}, 
                                 {'text': 'no?'}]}}
    
    res1 = target('file1.json', data)
    res2 = target('file2.json', data)
    
    expected1 = ' Person 1: hi Person 2: hello Person 1: nice'
    expected2 = ' Person 1: a b Person 2:  Person 1: good  Person 2: no?'

    successful_cases = 0
    failed_cases = 0
    
    try:
        assert res1 == expected1
        successful_cases += 1
    except ValueError:
        print('Error in test 1 \nResult  : ', res1, 'x \nExpected: ', expected1)
        failed_cases += 1
    try:
        assert res2 == expected2
        successful_cases += 1
    except:
        print('Error in test 2 \nResult  : ', res2, ' \nExpected: ', expected2)
        failed_cases += 1
            
    if failed_cases == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', successful_cases," Tests passed")
        print('\033[91m', failed_cases, " Tests failed")


# UNIT TEST for UNQ_C2
def test_reversible_layer_forward(target):
    
    successful_cases = 0
    failed_cases = 0
    
    test_cases = [{'name': 'input1'
                   ,'input': {'x': np.array([1, 2, 3, 4, 5, 6, 7, 8])
                             , 'f': lambda x: x + 2                              
                             , 'g': lambda x: x * 3
                             }
                   , 'expected': np.array([8, 10, 12, 14, 29, 36, 43, 50])
                  },
                  {'name': 'input2'
                   ,'input': {'x': np.array([1] * 128)
                             , 'f': lambda x: x + 1
                             , 'g': lambda x: x * 2
                             }
                   , 'expected': np.array([3] * 64 + [7] * 64)
                  },
                  {'name': 'input3'
                   ,'input': {'x': np.array([1] * 20)
                             , 'f': lambda x: 3*x + 1
                             , 'g': lambda x: x * 2./5
                             }
                   , 'expected': np.array([5] * 10 + [3] * 10)
                  },
                 ]
    
    for test_case in test_cases:
        res = target(**test_case['input'])
    
        try:            
            assert isinstance(res, np.ndarray)
            successful_cases += 1
        except:
            print('Wrong type! Output is not of type np.ndarray')
            failed_cases += 1
        try:            
            assert np.allclose(res, test_case['expected'])
            successful_cases += 1
        except ValueError:
            print('Error in test 1 \nResult  : ', res, 'x \nExpected: ', test_case['expected'])
            failed_cases += 1

    if failed_cases == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', successful_cases," Tests passed")
        print('\033[91m', failed_cases, " Tests failed")


# UNIT TEST for UNQ_C3
def test_reversible_layer_reverse(target):
    
    successful_cases = 0
    failed_cases = 0
   
    
    test_cases = [{'name': 'input1'
                   ,'input': {'y': np.array([1, 2, 3, 4, 5, 6, 7, 8])
                             , 'f': lambda x: x + 2                              
                             , 'g': lambda x: x * 3
                             }
                   , 'expected': np.array([-3,  0,  3,  6,  2,  0, -2, -4])
                  },
                  {'name': 'input2'
                   ,'input': {'y': np.array([1] * 128)
                             , 'f': lambda x: x + 1
                             , 'g': lambda x: x * 2
                             }
                   , 'expected': np.array([1] * 64 + [-1] * 64)
                  },
                  {'name': 'input3'
                   ,'input': {'y': np.array([5] * 10 + [3] * 10)
                             , 'f': lambda x: 3*x + 1
                             , 'g': lambda x: x * 2./5
                             }
                   , 'expected': np.array([1] * 20)
                  },
                 ]
    
    for test_case in test_cases:
        res = target(**test_case['input'])
        
        try:        
            assert isinstance(res, np.ndarray)
            successful_cases += 1
        except:
            print('Wrong type! Output is not of type np.ndarray')
            failed_cases += 1
        try:        
            assert np.allclose(res, test_case['expected'])
            successful_cases += 1
        except ValueError:
            print('Error in test 1 \nResult  : ', res, 'x \nExpected: ', test_case['expected'])
            failed_cases += 1
            
    if failed_cases == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', successful_cases," Tests passed")
        print('\033[91m', failed_cases, " Tests failed")


# +
#def test_reversible_layer_forward_reverse(target):    
# -

def test_ReformerLM(target):
    test_cases = [
        {
            "name": "default_reformerlm"
            , "input": {'vocab_size':33000, 'n_layers':2, 'mode':'train', 'attention_type':tl.SelfAttention}
            ,"expected": {
                "expected_str": "Serial[\n  Serial[\n    Serial[\n      ShiftRight(1)\n    ]\n    Embedding_33000_512\n    Dropout\n    Serial[\n      PositionalEncoding\n    ]\n    Dup_out2\n    ReversibleSerial_in2_out2[\n      ReversibleHalfResidualDecoderAttn_in2_out2[\n        Serial[\n          LayerNorm\n        ]\n        SelfAttention\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderFF_in2_out2[\n        Serial[\n          LayerNorm\n          Dense_2048\n          Dropout\n          Serial[\n            FastGelu\n          ]\n          Dense_512\n          Dropout\n        ]\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderAttn_in2_out2[\n        Serial[\n          LayerNorm\n        ]\n       SelfAttention\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderFF_in2_out2[\n        Serial[\n         LayerNorm\n          Dense_2048\n          Dropout\n          Serial[\n            FastGelu\n          ]\n          Dense_512\n          Dropout\n        ]\n      ]\n      ReversibleSwap_in2_out2\n    ]\n    Concatenate_in2\n    LayerNorm\n    Dropout\n    Serial[\n      Dense_33000\n    ]\n  ]\n  LogSoftmax\n]",
                "n_sublayers": 11,
            },
        },
        
        {
            "name": "check_reformerlm",
            "input": {
                "vocab_size": 100,
                "n_layers": 3,
                "mode": "train",
                "attention_type": tl.SelfAttention,
            },
            "expected": {
                "expected_str": "Serial[\n  Serial[\n    Serial[\n      ShiftRight(1)\n    ]\n    Embedding_100_512\n    Dropout\n    Serial[\n      PositionalEncoding\n    ]\n    Dup_out2\n    ReversibleSerial_in2_out2[\n      ReversibleHalfResidualDecoderAttn_in2_out2[\n        Serial[\n          LayerNorm\n        ]\n        SelfAttention\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderFF_in2_out2[\n        Serial[\n          LayerNorm\n          Dense_2048\n          Dropout\n          Serial[\n            FastGelu\n          ]\n          Dense_512\n          Dropout\n        ]\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderAttn_in2_out2[\n        Serial[\n          LayerNorm\n        ]\n        SelfAttention\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderFF_in2_out2[\n        Serial[\n          LayerNorm\n          Dense_2048\n          Dropout\n          Serial[\n            FastGelu\n          ]\n          Dense_512\n          Dropout\n        ]\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderAttn_in2_out2[\n        Serial[\n          LayerNorm\n        ]\n        SelfAttention\n      ]\n      ReversibleSwap_in2_out2\n      ReversibleHalfResidualDecoderFF_in2_out2[\n        Serial[\n          LayerNorm\n          Dense_2048\n          Dropout\n          Serial[\n            FastGelu\n          ]\n          Dense_512\n          Dropout\n        ]\n      ]\n      ReversibleSwap_in2_out2\n    ]\n    Concatenate_in2\n    LayerNorm\n    Dropout\n    Serial[\n      Dense_100\n    ]\n  ]\n  LogSoftmax\n]",
                "n_sublayers": 11,
            },
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:        
        temp_model = target(**test_case["input"])

        try:

            assert test_case["expected"]["expected_str"].replace(" ", "").replace(
                "\n", ""
            ) == str(temp_model).replace(" ", "").replace("\n", "")
            successful_cases += 1

            try:
                assert test_case["expected"]["n_sublayers"] == len(
                    temp_model.sublayers
                ) - 1 + len(temp_model.sublayers[0].sublayers)
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "n_sublayers_check",
                        "expected": test_case["expected"]["n_sublayers"],
                        "got": len(temp_model.sublayers)
                        - 1
                        + len(temp_model.sublayers[0].sublayers),
                    }
                )
                print(
                    f"Wrong number of sublayers.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot{failed_cases[-1].get('got')}."
                )

        except:
            failed_cases.append(
                {
                    "name": "str_rep_check",
                    "expected": test_case["expected"]["expected_str"],
                    "got": str(temp_model),
                }
            )
            print(
                f"Wrong model.\n Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}"
            )
    
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


# UNIT TEST for UNQ_C5
def test_tasks(target):
    train_task = target._tasks
    eval_task = target._eval_tasks
    
    successful_cases = 0
    failed_cases = []
    
    try:
        assert isinstance(train_task, list) 
        successful_cases += 1
        
        try:
            assert len(train_task) == 1
            successful_cases += 1
        except:
            failed_cases.append({
                    "name": "train_task_elements",
                    "expected": 1,
                    "got": len(train_task),
                })
            print(f'train_task list contains more than one task.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')                    
        
    except:
        failed_cases.append({
                    "name": "train_task_type",
                    "expected": list,
                    "got": type(train_task),
                })
        print(f'train_task object has the wrong type.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')
        
        
    try:
        assert isinstance(eval_task, list) 
        successful_cases += 1
        
        try:
            assert len(eval_task) == 1
            successful_cases += 1
        except:
            failed_cases.append({
                    "name": "eval_task_elements",
                    "expected": 1,
                    "got": len(eval_task),
                })
            print(f'eval_task list contains more than one task.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')

        
    except:
        failed_cases.append({
                    "name": "eval_task_type",
                    "expected": list,
                    "got": type(eval_task),
                })
        print(f'eval_task object has the wrong type.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')
        
     
    # Test the labeled data parameter for train_task
    strlabel = str(train_task[0]._labeled_data)
    try:        
        assert ("generator" in strlabel) and ("add_loss_weights" in  strlabel)
        successful_cases += 1
    except:
        failed_cases.append({
                    "name": "train_labeled_data",
                    "expected": 'generator object add_loss_weights',
                    "got": str(train_task[0]._labeled_data),
                })
        print(f'Wrong labeled data parameter in train_task.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')
    
    # Test the cross entropy loss data parameter
    strlabel = str(train_task[0]._loss_layer)
    try:        
        assert(strlabel == "CrossEntropyLoss_in3")
        successful_cases += 1
    except:
        failed_cases.append({
            "name": "train_loss_layer",
            "expected": "CrossEntropyLoss_in3",
            "got": str(train_task[0]._loss_layer),
        })
        print(f'Wrong loss function.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')

    # Test the optimizer parameter
    try:
        assert(isinstance(train_task[0].optimizer, trax.optimizers.adam.Adam))
        successful_cases += 1
    except:
        failed_cases.append({
            "name": "train_optimizer",
            "expected": trax.optimizers.adam.Adam,
            "got": type(train_task[0].optimizer),
        })
        print(f'Wrong optimizer.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')
        
    opt_params_dict = {'weight_decay_rate': fastnp.array(1.e-5),
                         'b1': fastnp.array(0.9),
                         'b2': fastnp.array(0.999),
                         'eps': fastnp.array(1.e-5),
                         'learning_rate': fastnp.array(0.01)}
    
    try: 
        assert train_task[0]._optimizer.opt_params == opt_params_dict
        successful_cases += 1
    except:
        failed_cases.append({"name": "optimizer_parameters",
                            "expected": opt_params_dict,
                            "got": train_task[0].optimizer.opt_params,})
        print(f"Optimizer has the wrong parameters.\n\tExpected {opt_params_dict}.\n\tGot {train_task[0]._optimizer.opt_params}.")


    # Test the schedule parameter
    try:
        assert(isinstance(train_task[0]._lr_schedule, trax.supervised.lr_schedules._BodyAndTail))
        successful_cases += 1
    except:
        failed_cases.append({
            "name": "train_optimizer",
            "expected": trax.supervised.lr_schedules._BodyAndTail,
            "got": type(train_task[0]._lr_schedule),
        })
        print(f'Wrong learning rate schedule type.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')

    try: 
        assert train_task[0]._lr_schedule._body_value == 0.01
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "lr_check",
                "expected": 0.01,
                "got": train_task[0]._lr_schedule._body_value,
            }
        )
        print(f'Wrong learning rate value.\n\tExpected 0.01.\n\tGot {output_loop._tasks[0]._lr_schedule._body_value}.')

    
    # Test the labeled data parameter for eval_task
    strlabel = str(eval_task[0]._labeled_data)
    try:        
        assert ("generator" in strlabel) and ("add_loss_weights" in  strlabel)
        successful_cases += 1
    except:
        failed_cases.append({
                    "name": "eval_labeled_data",
                    "expected": 'generator object add_loss_weights',
                    "got": str(eval_task[0]._labeled_data),
                })
        print(f'Wrong labeled data parameter in eval_task.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')
    
    # Test the metrics in eval_task     
    try:        
        assert eval_task[0]._metric_names == ["CrossEntropyLoss", "Accuracy"]
        successful_cases += 1
    except:
        failed_cases.append({
                    "name": "eval_metrics",
                    "expected": ["CrossEntropyLoss", "Accuracy"],
                    "got": eval_task[0]._metric_names,
                })
        print(f'Wrong metrics in eval_task.\n Expected: {failed_cases[-1].get("expected")}.\n Got: {failed_cases[-1].get("got")}')
        
        
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', successful_cases," Tests passed")
        print('\033[91m', len(failed_cases), " Tests failed")


def test_ReformerLM_output_gen(test_model, test_output_gen):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [{
        "name": "ReformerLM_output",
        "expected": jax.device_put([1,0,4,3,0,4]),
        "error_message": "",
        },]    
    
    expected_output = jax.device_put([1,0,4,3,0,4])
    
    WEIGHTS_FROM_FILE = ()


    with open('weights', 'rb') as file:
        WEIGHTS_FROM_FILE = pickle.load(file)


    shape11 = trax.shapes.ShapeDtype((1, 1), dtype=np.int32)
    
    test_model.init_weights_and_state(shape11)

    test_model.weights = WEIGHTS_FROM_FILE

    output = []

    for i in range(6):
        output.append(next(test_output_gen)[0])
        
    try:
        assert int(len(output)) == int(len(expected_output))
        successful_cases += 1
    except:
        failed_cases.append({"name": "ReformerLM_output_len",
                             "expected": len(expected_output),
                             "got": len(output),})
        print(f"Length of generated output does not match with expected.\n Expected {failed_cases[-1].get('expected')}.\n Got {failed_cases[-1].get('got')}")
    
    
    try:
        for elem_out, elem_exp in zip(output, expected_output):
            assert elem_out == elem_exp
        successful_cases += 1
    except:
        failed_cases.append({"name": "ReformerLM_output",
                             "expected": expected_output,
                             "got": output,})
        print(f"Generated output does not match expected.\n Expected {failed_cases[-1].get('expected')}.\n Got {failed_cases[-1].get('got')}")
    
    print('Generated output', output)
    
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', successful_cases," Tests passed")
        print('\033[91m', len(failed_cases), " Tests failed")
    
    del WEIGHTS_FROM_FILE



