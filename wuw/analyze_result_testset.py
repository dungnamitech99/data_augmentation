import json
import numpy as np

if __name__ == "__main__":
    with open("nami_record/result_testset_easy.json", "r") as f:
        result_testset_easy = json.load(f)

    # Analyze result from rule 1
    lstm_rule1 = []
    dscnn_rule1 = []
    result_rule1 = result_testset_easy["rule1"]
    for key in result_rule1:
        lstm_rule1.append(result_rule1[key]["lstm"]["false_alarm"])
        dscnn_rule1.append(result_rule1[key]["dscnn"]["false_alarm"])

    print("lstm rule1:", lstm_rule1)
    print("lstm rule1 fa_rate:", np.array(lstm_rule1)/60480*100)
    print("dscnn rule1:", dscnn_rule1)
    print("dscnn rule1 fa_rate:", np.array(dscnn_rule1)/60480*100)
    
    # Analyze result from rule 2
    lstm_rule2 = []
    dscnn_rule2 = []
    result_rule2 = result_testset_easy["rule2"]
    for key in result_rule2:
        lstm_rule2.append(result_rule2[key]["lstm"]["false_alarm"])
        dscnn_rule2.append(result_rule2[key]["dscnn"]["false_alarm"])

    print("lstm rule2:", lstm_rule2)
    print("lstm rule2 fa_rate:", np.array(lstm_rule2)/60480*100)
    print("dscnn rule2:", dscnn_rule2)
    print("dscnn rule2 fa_rate:", np.array(dscnn_rule2)/60480*100)
    # Analyze result from rule 3 
    fa_rule3 = []
    result_rule3 = result_testset_easy["rule3"]
    for key in result_rule3:
        fa_rule3.append(result_rule3[key]["false_alarm"])
    
    print("rule3 result:", fa_rule3)
    print("rule3 fa_rate:", np.array(fa_rule3)/60480*100)
    # Analyze result from rule 4 
    fa_rule4 = []
    result_rule4 = result_testset_easy["rule4"]
    for key in result_rule4:
        fa_rule4.append(result_rule4[key]["false_alarm"])
    
    print("rule4 result:", fa_rule4)
    print("rule4 fa_rate:", np.array(fa_rule4)/60480*100)